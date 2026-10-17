//! The daemon: owns ComfyUI, runs the jobs in order one child at a time, imports every result, answers the
//! socket and draws the screen. One thread, this one, makes every decision; the others only feed it messages.

use crate::comfy::{self, Http, Process};
use crate::convert::{self, Converted};
use crate::import;
use crate::runner::{self, Tools};
use crate::screen::{RowData, RowKind, Screen, View};
use crate::socket::{self, Responder};
use anyhow::{Context, Result, bail};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rope_core::hash::short;
use rope_core::library::Lock;
use rope_core::queue::protocol::{self, ChildEvent, Event, Request, ResultEvent, Submission, SubmitRequest};
use rope_core::queue::{self, Input, Job, NewGroup, State, Submitted};
use rope_core::store::{self, NewImage};
use rope_core::{Library, time};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::Write;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

pub struct Options {
    pub lib: Library,
    pub repo: PathBuf,
    /// `host:port`.
    pub server: String,
    pub port: u16,
    /// GB left to the desktop, `--reserve-vram`; none to leave ComfyUI's default.
    pub reserve_vram: Option<String>,
    pub comfy_args: Vec<String>,
    /// A child silent this long is killed.
    pub job_timeout: Duration,
    /// comfyconvert.py.
    pub converter: PathBuf,
    pub screen: bool,
    /// The ComfyUI (pid, log) the daemon this one replaces through `reload` hands over.
    pub adopt: Option<(i32, PathBuf)>,
}

pub enum Msg {
    Request(Request, Responder),
    Child { job: u64, what: ChildMsg },
    Converted { input: String, result: Converted },
    Key(KeyEvent),
}

pub enum ChildMsg {
    Event(ChildEvent),
    /// A line of stderr.
    Line(String),
    /// A line of stdout that is not an event.
    Garbage(String),
    /// Sent last, after every line.
    Exited(ExitStatus),
}

const COMFY_BOOT: Duration = Duration::from_secs(300);
const STATS_EVERY: Duration = Duration::from_secs(5);
const STATS_MISSES: u32 = 6;
const IDLE_FREE: Duration = Duration::from_secs(60);
const KILL_AFTER: Duration = Duration::from_secs(30);
const COMFY_STOP: Duration = Duration::from_secs(30);
const DEATHS_WITHIN: Duration = Duration::from_secs(600);
const CONVERTERS: usize = 2;
const NOTES_KEPT: usize = 200;

static SIGNAL: AtomicI32 = AtomicI32::new(0);

extern "C" fn on_signal(sig: libc::c_int) {
    SIGNAL.store(sig, Ordering::SeqCst);
}

fn trap_signals() {
    for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        // SAFETY: installing an async-signal-safe handler that only stores an integer.
        unsafe {
            libc::signal(sig, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
        }
    }
}

fn is_tty() -> bool {
    // SAFETY: a plain libc query.
    unsafe { libc::isatty(1) == 1 }
}

fn kill(pid: i32, sig: i32) {
    // SAFETY: a plain libc call on a pid this process started.
    unsafe {
        libc::kill(pid, sig);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Comfy {
    Booting { since: Instant, polled: Instant },
    Up { polled: Instant, misses: u32 },
    Down,
}

/// What the running job's end should bring, decided before the child exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Judged from the exit and what the child delivered.
    Normal,
    /// The user cancelled it.
    Cancel,
    /// Interrupted by the daemon: back in line.
    Requeue,
    /// Killed by the daemon for cause: `why` says.
    Fail,
}

struct Running {
    id: u64,
    pid: i32,
    client_id: String,
    started: Instant,
    /// The last time the child said anything.
    last: Instant,
    stage: String,
    progress: Option<(u32, u32)>,
    result: Option<ResultEvent>,
    error: Option<String>,
    last_line: Option<String>,
    ended: bool,
    seconds: Option<f64>,
    outcome: Outcome,
    why: Option<String>,
    /// When SIGINT went to it; SIGKILL follows after `KILL_AFTER`.
    killed_at: Option<Instant>,
    killed: bool,
}

struct Daemon {
    opts: Options,
    lib: Library,
    tools: Tools,
    http: Http,
    proc: Option<Process>,
    comfy: Comfy,
    comfy_version: String,
    /// What the subscribers were last told about ComfyUI, so they hear of every change and nothing else.
    comfy_told: Option<(String, bool)>,
    deaths: VecDeque<Instant>,
    _lock: Lock,
    jobs: BTreeMap<u64, Job>,
    running: Option<Running>,
    converting: HashSet<String>,
    converted: HashMap<String, Converted>,
    paused: bool,
    pause_reason: Option<String>,
    stopping: bool,
    reload: bool,
    idle_since: Option<Instant>,
    /// Whether ComfyUI may hold models in VRAM since the last job.
    vram_held: bool,
    subs: Vec<(Responder, Vec<u64>)>,
    notes: VecDeque<String>,
    screen: Option<Screen>,
    rx: Receiver<Msg>,
    tx: Sender<Msg>,
}

/// Starts the daemon and runs it until it is stopped. Everything that can be checked is checked before ComfyUI
/// starts, so a mistake costs no GPU time.
pub fn run(mut opts: Options) -> Result<()> {
    if let Ok(v) = socket::call(&Request::Ping) {
        bail!("a rope-queue is running already (pid {}, library {})", v["pid"], v["library"].as_str().unwrap_or("?"));
    }
    let tools = Tools::new(&opts.repo, &opts.server);
    tools.check()?;
    if !opts.converter.is_file() {
        bail!("no converter at {}; pass --converter FILE", opts.converter.display());
    }
    if !opts.repo.join("run_comfyui.sh").is_file() {
        bail!("{} is not the comfyrope repo (no run_comfyui.sh)", opts.repo.display());
    }
    let lock = opts.lib.try_queue_lock()?.context("another rope-queue holds queue/lock in this library")?;
    opts.lib.set_repo(&opts.repo)?;
    let http = Http::new(&opts.server);
    let (tx, rx) = channel();
    let lib = opts.lib.clone();
    let adopt = opts.adopt.take();
    let mut d = Daemon {
        lib,
        tools,
        http,
        proc: None,
        comfy: Comfy::Down,
        comfy_version: String::from("?"),
        comfy_told: None,
        deaths: VecDeque::new(),
        _lock: lock,
        jobs: BTreeMap::new(),
        running: None,
        converting: HashSet::new(),
        converted: HashMap::new(),
        paused: false,
        pause_reason: None,
        stopping: false,
        reload: false,
        idle_since: None,
        vram_held: false,
        subs: Vec::new(),
        notes: VecDeque::new(),
        screen: None,
        rx,
        tx: tx.clone(),
        opts,
    };
    match adopt {
        Some((pid, log)) => {
            let mut p = Process::adopt(pid, log);
            let stats = d.http.stats();
            if !p.alive() || stats.is_err() {
                bail!("the ComfyUI handed over by the daemon before (pid {pid}) is gone; start again");
            }
            d.comfy_version = Http::version(&stats?);
            d.comfy = Comfy::Up { polled: Instant::now(), misses: 0 };
            d.proc = Some(p);
            d.vram_held = true;
            d.note(format!("reloaded; keeping ComfyUI {} (pid {pid})", d.comfy_version));
        }
        None => {
            if d.http.answers() {
                bail!(
                    "a ComfyUI already answers at {}; stop it first: rope-queue starts and owns its own",
                    d.opts.server
                );
            }
            d.boot()?;
        }
    }
    let out = d.serve(tx).and_then(|()| d.run_loop());
    d.shutdown();
    out
}

impl Daemon {
    /// Settles the jobs left behind, opens the socket, traps the signals and takes the screen.
    fn serve(&mut self, tx: Sender<Msg>) -> Result<()> {
        self.recover();
        let listener = socket::listen()?;
        let stx = tx.clone();
        std::thread::spawn(move || socket::serve(listener, stx));
        trap_signals();
        if self.opts.screen && is_tty() {
            self.screen = Some(Screen::open(tx)?);
        }
        self.note(format!(
            "rope-queue (pid {}) listening on {}; library {}",
            std::process::id(),
            socket::path().display(),
            self.lib.root().display()
        ));
        Ok(())
    }

    fn run_loop(&mut self) -> Result<()> {
        loop {
            match self.rx.recv_timeout(Duration::from_millis(200)) {
                Ok(msg) => self.handle(msg),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => bail!("every thread is gone"),
            }
            while let Ok(msg) = self.rx.try_recv() {
                self.handle(msg);
            }
            let sig = SIGNAL.swap(0, Ordering::SeqCst);
            if sig != 0 {
                let name = match sig {
                    libc::SIGINT => "SIGINT",
                    libc::SIGTERM => "SIGTERM",
                    libc::SIGHUP => "SIGHUP",
                    _ => "a signal",
                };
                self.note(format!("{name}: stopping"));
                self.begin_stop();
            }
            self.tick();
            if self.stopping && self.running.is_none() {
                return Ok(());
            }
            if self.reload && self.running.is_none() && !self.stopping {
                self.exec_reload();
            }
            self.draw()?;
        }
    }

    fn tick(&mut self) {
        self.check_comfy();
        self.tell_comfy();
        self.conversions();
        self.watch_child();
        self.start_next();
        self.idle();
    }

    // ------------------------------------------------------------ notes and the screen

    /// Says something: on the screen, on stdout when there is none, and in `queue/rope-queue.log`.
    fn note(&mut self, text: impl Into<String>) {
        let text = text.into();
        let stamp = time::now();
        if let Ok(mut f) =
            std::fs::OpenOptions::new().create(true).append(true).open(self.lib.queue_dir().join("rope-queue.log"))
        {
            let _ = writeln!(f, "{stamp} {text}");
        }
        if self.screen.is_none() {
            println!("{} {text}", &stamp[11..19]);
        }
        self.notes.push_back(format!("{} {text}", &stamp[11..19]));
        while self.notes.len() > NOTES_KEPT {
            self.notes.pop_front();
        }
    }

    fn draw(&mut self) -> Result<()> {
        if self.screen.is_none() {
            return Ok(());
        }
        let view = self.view();
        if let Some(screen) = &mut self.screen {
            screen.draw(&view)?;
        }
        Ok(())
    }

    fn view(&self) -> View {
        let comfy = match self.comfy {
            Comfy::Booting { since, .. } => format!("ComfyUI starting ({})", hms(since.elapsed())),
            Comfy::Up { .. } => match &self.proc {
                Some(p) => format!("ComfyUI {} up {} (pid {})", self.comfy_version, hms(p.started.elapsed()), p.pid),
                None => format!("ComfyUI {}", self.comfy_version),
            },
            Comfy::Down => "ComfyUI down".to_owned(),
        };
        let log = self.proc.as_ref().map(|p| format!("   log {}", p.log.display())).unwrap_or_default();
        let header = format!("{comfy}{log}");
        let status = if self.stopping {
            Some("stopping".to_owned())
        } else if self.reload {
            Some("reload after this job".to_owned())
        } else if self.paused {
            Some(format!("PAUSED: {}", self.pause_reason.as_deref().unwrap_or("by request")))
        } else {
            None
        };
        let expanded = self.screen.as_ref().map(|s| s.expanded.clone()).unwrap_or_default();
        let selected = self.screen.as_ref().and_then(|s| s.selected());
        let mut groups: BTreeMap<u64, Vec<&Job>> = BTreeMap::new();
        for j in self.jobs.values() {
            groups.entry(j.group.id).or_default().push(j);
        }
        let mut rows = Vec::new();
        for (gid, jobs) in &groups {
            let (first, last) = (jobs[0].id, jobs[jobs.len() - 1].id);
            let count = |s: State| jobs.iter().filter(|j| j.state == s).count();
            let done = count(State::Done);
            let failed = count(State::Failed);
            let cancelled = count(State::Cancelled);
            let mut progress = format!("{done}/{} done", jobs.len());
            if failed > 0 {
                progress += &format!(", {failed} failed");
            }
            if cancelled > 0 {
                progress += &format!(", {cancelled} cancelled");
            }
            let all_done = jobs.iter().all(|j| j.is_terminal());
            let now = match &self.running {
                Some(r) if jobs.iter().any(|j| j.id == r.id) => running_line(r),
                _ if all_done => String::new(),
                _ if jobs.iter().any(|j| j.state == State::Converting) => "converting inputs".to_owned(),
                _ if self.paused => "paused".to_owned(),
                _ => "waiting".to_owned(),
            };
            let open = expanded.contains(gid);
            rows.push(RowData {
                kind: RowKind::Group(*gid),
                cells: vec![
                    if open { "▾" } else { "▸" }.to_owned(),
                    format!("#{gid}"),
                    if first == last { format!("#{first}") } else { format!("#{first}–#{last}") },
                    jobs[0].tool.trim_end_matches(".py").to_owned(),
                    jobs[0].into.clone().unwrap_or_default(),
                    progress,
                    now,
                ],
                dim: all_done && failed == 0,
                warn: false,
            });
            if open {
                for j in jobs {
                    let now = match (&self.running, j.state) {
                        (Some(r), State::Running) if r.id == j.id => running_line(r),
                        (_, State::Done) => describe_result(j.result.as_ref()),
                        (_, State::Failed) => j.error.clone().unwrap_or_default(),
                        _ => String::new(),
                    };
                    rows.push(RowData {
                        kind: RowKind::Job(j.id),
                        cells: vec![
                            String::new(),
                            format!("#{}", j.id),
                            j.state.to_string(),
                            String::new(),
                            j.inputs.first().map(|i| i.name.clone()).unwrap_or_default(),
                            seed_of(j),
                            now,
                        ],
                        dim: j.is_terminal() && j.state != State::Failed,
                        warn: j.state == State::Failed,
                    });
                }
            }
        }
        let detail = match selected {
            Some(RowKind::Job(id)) => self.jobs.get(&id).map(|j| self.job_detail(j)).unwrap_or_default(),
            Some(RowKind::Group(g)) => groups.get(&g).map(|jobs| group_detail(jobs[0])).unwrap_or_default(),
            None => vec!["no jobs; submit some with ./img2img.py or ./img2vid.py".to_owned()],
        };
        let notes: Vec<String> = self.notes.iter().rev().take(6).rev().cloned().collect();
        View { header, status, rows, detail, notes }
    }

    fn job_detail(&self, j: &Job) -> Vec<String> {
        let mut lines = vec![format!(
            "#{}  {}  {} {}  → {}",
            j.id,
            j.state,
            j.tool,
            j.flags.join(" "),
            j.into.as_deref().unwrap_or("(unfiled)")
        )];
        let inputs: Vec<String> = j.inputs.iter().map(|i| format!("{} ({})", i.name, short(&i.id))).collect();
        lines.push(format!("inputs: {}   prompt: {}", inputs.join(", "), j.prompt));
        match (&self.running, j.state) {
            (Some(r), State::Running) if r.id == j.id => {
                lines.push(format!("{}   {}", running_line(r), r.last_line.as_deref().unwrap_or("")));
            }
            (_, State::Done) => lines.push(describe_result(j.result.as_ref())),
            (_, State::Failed) => lines.push(format!("error: {}", j.error.as_deref().unwrap_or("?"))),
            _ => {}
        }
        lines
    }

    // ------------------------------------------------------------ ComfyUI

    fn boot(&mut self) -> Result<()> {
        let log = comfy::new_log(&self.lib)?;
        let p = Process::start(
            &self.opts.repo,
            self.opts.port,
            self.opts.reserve_vram.as_deref(),
            &self.opts.comfy_args,
            &log,
        )?;
        self.note(format!("starting ComfyUI (pid {}); its log is {}", p.pid, log.display()));
        self.proc = Some(p);
        let now = Instant::now();
        self.comfy = Comfy::Booting { since: now, polled: now };
        self.vram_held = false;
        Ok(())
    }

    fn check_comfy(&mut self) {
        let now = Instant::now();
        match self.comfy {
            Comfy::Booting { since, polled } => {
                if self.proc.as_mut().is_some_and(|p| !p.alive()) {
                    let tail = self.log_tail();
                    self.comfy = Comfy::Down;
                    self.pause(format!("ComfyUI exited while starting; its log ends with:\n{tail}"));
                } else if now.duration_since(polled) >= Duration::from_secs(1) {
                    self.comfy = Comfy::Booting { since, polled: now };
                    if let Ok(stats) = self.http.stats() {
                        self.comfy_version = Http::version(&stats);
                        self.comfy = Comfy::Up { polled: now, misses: 0 };
                        let pid = self.proc.as_ref().map(|p| p.pid).unwrap_or(0);
                        self.note(format!(
                            "ComfyUI {} is up (pid {pid}) after {}",
                            self.comfy_version,
                            hms(since.elapsed())
                        ));
                    } else if now.duration_since(since) > COMFY_BOOT {
                        if let Some(p) = self.proc.as_mut() {
                            p.stop(Duration::from_secs(10));
                        }
                        let tail = self.log_tail();
                        self.comfy = Comfy::Down;
                        self.pause(format!("ComfyUI did not answer within 5 minutes; its log ends with:\n{tail}"));
                    }
                }
            }
            Comfy::Up { polled, misses } => {
                if self.proc.as_mut().is_some_and(|p| !p.alive()) {
                    self.comfy_died("ComfyUI exited");
                } else if now.duration_since(polled) >= STATS_EVERY {
                    let misses = if self.http.answers() { 0 } else { misses + 1 };
                    self.comfy = Comfy::Up { polled: now, misses };
                    if misses >= STATS_MISSES {
                        self.comfy_died("ComfyUI stopped answering");
                    }
                }
            }
            Comfy::Down => {}
        }
    }

    fn log_tail(&self) -> String {
        self.proc.as_ref().map(|p| comfy::tail(&p.log, 15)).unwrap_or_default()
    }

    /// ComfyUI is gone: the running job goes back in line, and ComfyUI is started again, once. A second death
    /// within ten minutes pauses the queue instead: a restart loop on the card that drives the display is not
    /// acceptable.
    fn comfy_died(&mut self, why: &str) {
        let tail = self.log_tail();
        if let Some(p) = self.proc.as_mut() {
            p.stop(Duration::from_secs(10));
        }
        self.comfy = Comfy::Down;
        let now = Instant::now();
        self.deaths.retain(|t| now.duration_since(*t) < DEATHS_WITHIN);
        self.deaths.push_back(now);
        if let Some(r) = &mut self.running
            && r.outcome == Outcome::Normal
        {
            r.outcome = Outcome::Requeue;
            r.why = Some(why.to_owned());
            kill(r.pid, libc::SIGINT);
            r.killed_at = Some(now);
        }
        if self.deaths.len() >= 2 {
            self.pause(format!("{why} twice within 10 minutes; not starting it again. Its log ends with:\n{tail}"));
        } else {
            self.note(format!("{why}; starting it again"));
            if let Err(e) = self.boot() {
                self.pause(format!("cannot start ComfyUI: {e:#}"));
            }
        }
    }

    fn release_vram(&mut self) {
        if !self.vram_held || !matches!(self.comfy, Comfy::Up { .. }) {
            return;
        }
        match self.http.free() {
            Ok(()) => {
                self.vram_held = false;
                self.note("ComfyUI handed its VRAM back");
            }
            Err(e) => self.note(format!("cannot ask ComfyUI to free its VRAM: {e:#}")),
        }
    }

    fn idle(&mut self) {
        if self.running.is_some() || !matches!(self.comfy, Comfy::Up { .. }) {
            return;
        }
        let waiting = !self.paused && self.jobs.values().any(|j| j.state == State::Ready);
        if waiting {
            self.idle_since = None;
            return;
        }
        let since = *self.idle_since.get_or_insert_with(Instant::now);
        if self.vram_held && (self.paused || since.elapsed() >= IDLE_FREE) {
            self.release_vram();
        }
    }

    // ------------------------------------------------------------ pause, stop, reload

    fn pause(&mut self, reason: String) {
        self.note(format!("paused: {reason}"));
        self.paused = true;
        self.pause_reason = Some(reason.clone());
        self.broadcast(Event::Paused { paused: true, reason: Some(reason) }, None);
    }

    fn resume(&mut self) {
        self.paused = false;
        self.pause_reason = None;
        self.note("resumed");
        self.broadcast(Event::Paused { paused: false, reason: None }, None);
        if self.comfy == Comfy::Down
            && let Err(e) = self.boot()
        {
            self.pause(format!("cannot start ComfyUI: {e:#}"));
        }
    }

    fn begin_stop(&mut self) {
        if self.stopping {
            return;
        }
        self.stopping = true;
        self.broadcast(Event::Stopping, None);
        match &mut self.running {
            Some(r) => {
                if r.outcome == Outcome::Normal {
                    r.outcome = Outcome::Requeue;
                    r.why = Some("rope-queue stopped".to_owned());
                }
                kill(r.pid, libc::SIGINT);
                r.killed_at = Some(Instant::now());
                let id = r.id;
                self.note(format!("stopping: interrupting #{id}"));
            }
            None => self.note("stopping"),
        }
    }

    fn shutdown(&mut self) {
        if let Some(screen) = self.screen.take() {
            screen.close();
        }
        if let Some(p) = self.proc.as_mut()
            && p.alive()
        {
            println!("stopping ComfyUI (pid {})", p.pid);
            p.stop(COMFY_STOP);
        }
        let _ = std::fs::remove_file(socket::path());
    }

    /// Becomes a fresh rope-queue, the same command line, ComfyUI kept: how a rebuilt binary is picked up.
    fn exec_reload(&mut self) {
        let Some(p) = &self.proc else {
            self.reload = false;
            return;
        };
        let handover = format!("{}:{}", p.pid, p.log.display());
        // a binary rebuilt underneath shows as "... (deleted)": the new one is at the plain path
        let exe = match std::env::current_exe() {
            Ok(exe) => PathBuf::from(exe.to_string_lossy().trim_end_matches(" (deleted)")),
            Err(e) => {
                self.reload = false;
                self.note(format!("reload failed: {e}"));
                return;
            }
        };
        if let Some(s) = self.screen.take() {
            s.close();
        }
        println!("reloading rope-queue, keeping ComfyUI (pid {})", p.pid);
        let err =
            std::process::Command::new(exe).args(std::env::args_os().skip(1)).env("ROPE_QUEUE_ADOPT", handover).exec();
        self.reload = false;
        self.note(format!("reload failed: {err}"));
        if self.opts.screen && is_tty() {
            self.screen = Screen::open(self.tx.clone()).ok();
        }
    }

    // ------------------------------------------------------------ jobs

    /// Reads the jobs left by the last daemon and settles the ones it left in the middle.
    fn recover(&mut self) {
        let (jobs, problems) = match queue::scan(&self.lib) {
            Ok(x) => x,
            Err(e) => {
                self.note(format!("cannot read the queue: {e:#}"));
                return;
            }
        };
        for p in problems {
            self.note(format!("cannot read a job: {p}"));
        }
        self.jobs = jobs.into_iter().map(|j| (j.id, j)).collect();
        let ids: Vec<u64> = self.jobs.keys().copied().collect();
        for id in ids {
            let state = self.jobs[&id].state;
            if !matches!(state, State::Running | State::Importing) {
                continue;
            }
            match delivered(&self.lib, id) {
                Some(ev) => {
                    self.note(format!("#{id}: importing the result the last daemon left behind"));
                    self.import_result(id, &ev, None, None);
                }
                None if state == State::Running => {
                    self.note(format!("#{id}: was running when the last daemon ended; back in line"));
                    self.apply(queue::requeue(&self.lib, id));
                }
                None => self.apply(queue::fail(&self.lib, id, "the result was lost before rope took it", None)),
            }
        }
    }

    fn apply(&mut self, r: Result<Job>) {
        match r {
            Ok(j) => self.set(j),
            Err(e) => self.note(format!("{e:#}")),
        }
    }

    /// Takes a job as saved into the cache; a change of state goes to the subscribers.
    fn set(&mut self, job: Job) {
        let changed = self.jobs.get(&job.id).is_none_or(|old| old.state != job.state);
        let group = job.group.id;
        self.jobs.insert(job.id, job.clone());
        if changed {
            self.broadcast(Event::State { job: Box::new(job) }, Some(group));
        }
    }

    /// Tells the subscribers when ComfyUI comes up or goes down, whichever way that happened.
    fn tell_comfy(&mut self) {
        let now = (self.comfy_version.clone(), matches!(self.comfy, Comfy::Up { .. }));
        if self.comfy_told.as_ref() != Some(&now) {
            self.comfy_told = Some(now.clone());
            self.broadcast(Event::Comfy { version: now.0, up: now.1 }, None);
        }
    }

    fn broadcast(&mut self, ev: Event, group: Option<u64>) {
        let v = serde_json::to_value(&ev).unwrap_or(Value::Null);
        self.subs.retain_mut(|(resp, groups)| {
            if let Some(g) = group
                && !groups.is_empty()
                && !groups.contains(&g)
            {
                return true;
            }
            resp.send(&v).is_ok()
        });
    }

    /// Moves queued jobs on: converts what needs converting, once per input, and makes a job ready when every
    /// input of it is settled.
    fn conversions(&mut self) {
        let ids: Vec<u64> =
            self.jobs.values().filter(|j| matches!(j.state, State::Queued | State::Converting)).map(|j| j.id).collect();
        for id in ids {
            let job = self.jobs[&id].clone();
            let mut inputs = job.inputs.clone();
            let (mut changed, mut pending, mut failed) = (false, false, None);
            for input in &mut inputs {
                if input.rendition.is_some() {
                    continue;
                }
                match self.converted.get(&input.id) {
                    Some(Converted::AsIs) => {}
                    Some(Converted::Rendition(r)) => {
                        input.rendition = Some(r.clone());
                        changed = true;
                    }
                    Some(Converted::Failed(e)) => {
                        failed = Some(format!("cannot convert input {} ({}): {e}", input.name, short(&input.id)));
                    }
                    None => {
                        pending = true;
                        if !self.converting.contains(&input.id) && self.converting.len() < CONVERTERS {
                            self.converting.insert(input.id.clone());
                            convert::spawn(
                                &self.tools,
                                &self.lib,
                                &self.opts.converter,
                                input.id.clone(),
                                self.tx.clone(),
                            );
                        }
                    }
                }
            }
            if let Some(e) = failed {
                self.note(format!("#{id} failed: {e}"));
                self.apply(queue::fail(&self.lib, id, &e, None));
            } else if !pending {
                self.apply(queue::update(&self.lib, id, |j| {
                    j.inputs = inputs;
                    j.state = State::Ready;
                    Ok(j.clone())
                }));
            } else if changed {
                self.apply(queue::update(&self.lib, id, |j| {
                    j.inputs = inputs;
                    Ok(j.clone())
                }));
            } else if job.state == State::Queued {
                self.apply(queue::set_state(&self.lib, id, State::Converting));
            }
        }
    }

    fn start_next(&mut self) {
        if self.paused
            || self.stopping
            || self.reload
            || self.running.is_some()
            || !matches!(self.comfy, Comfy::Up { .. })
        {
            return;
        }
        let Some(id) = self.jobs.values().find(|j| j.state == State::Ready).map(|j| j.id) else { return };
        let client_id = format!("rope-queue-{id}-{}-{}", self.jobs[&id].attempts + 1, std::process::id());
        let job = match queue::start(&self.lib, id) {
            Ok(j) => j,
            Err(e) => {
                self.note(format!("#{id}: {e:#}"));
                return;
            }
        };
        self.set(job.clone());
        match runner::spawn(&self.tools, &self.lib, &job, &client_id, self.tx.clone()) {
            Ok(pid) => {
                let now = Instant::now();
                self.running = Some(Running {
                    id,
                    pid,
                    client_id,
                    started: now,
                    last: now,
                    stage: String::from("starting"),
                    progress: None,
                    result: None,
                    error: None,
                    last_line: None,
                    ended: false,
                    seconds: None,
                    outcome: Outcome::Normal,
                    why: None,
                    killed_at: None,
                    killed: false,
                });
                self.idle_since = None;
                self.vram_held = true;
                self.note(format!("#{id} running: {} {}", job.tool, seed_of(&job)));
            }
            Err(e) => {
                let msg = format!("cannot start the tool: {e:#}");
                self.note(format!("#{id} failed: {msg}"));
                self.apply(queue::fail(&self.lib, id, &msg, None));
            }
        }
    }

    /// A child silent for too long is interrupted, then killed.
    fn watch_child(&mut self) {
        let timeout = self.opts.job_timeout;
        let Some(r) = &mut self.running else { return };
        let now = Instant::now();
        match r.killed_at {
            Some(at) if !r.killed && now.duration_since(at) > KILL_AFTER => {
                kill(r.pid, libc::SIGKILL);
                r.killed = true;
                let id = r.id;
                self.note(format!("#{id}: the tool ignored SIGINT for 30 s; killed"));
            }
            None if now.duration_since(r.last) > timeout => {
                r.outcome = Outcome::Fail;
                r.why = Some(format!("no progress for {} minutes", timeout.as_secs() / 60));
                kill(r.pid, libc::SIGINT);
                r.killed_at = Some(now);
                let id = r.id;
                self.note(format!("#{id}: no progress for {} minutes; interrupting", timeout.as_secs() / 60));
            }
            _ => {}
        }
    }

    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Request(req, responder) => self.request(req, responder),
            Msg::Child { job, what } => self.child(job, what),
            Msg::Converted { input, result } => {
                self.converting.remove(&input);
                if let Converted::Failed(e) = &result {
                    self.note(format!("converting {} failed: {e}", short(&input)));
                }
                self.converted.insert(input, result);
            }
            Msg::Key(k) => self.key(k),
        }
    }

    fn child(&mut self, job: u64, what: ChildMsg) {
        if self.running.as_ref().is_none_or(|r| r.id != job) {
            return; // a late word from a child that is gone
        }
        let now = Instant::now();
        let mut event = None;
        {
            let r = self.running.as_mut().expect("checked");
            r.last = now;
            match what {
                ChildMsg::Event(ChildEvent::Stage { text }) => {
                    r.stage = text.clone();
                    event = Some(Event::Stage { job, text });
                }
                ChildMsg::Event(ChildEvent::Progress { value, max }) => {
                    r.progress = Some((value, max));
                    event = Some(Event::Progress { job, value, max });
                }
                ChildMsg::Event(ChildEvent::Comfy { client_id, prompt_id }) => {
                    let comfy = json!({"client_id": client_id, "prompt_id": prompt_id});
                    if let Ok(j) = queue::update(&self.lib, job, |j| {
                        j.comfy = Some(comfy);
                        Ok(j.clone())
                    }) {
                        self.jobs.insert(job, j);
                    }
                }
                ChildMsg::Event(ChildEvent::Result(ev)) => r.result = Some(ev),
                ChildMsg::Event(ChildEvent::Error { message }) => r.error = Some(message),
                ChildMsg::Event(ChildEvent::End { seconds }) => {
                    r.ended = true;
                    r.seconds = Some(seconds);
                }
                ChildMsg::Line(text) => r.last_line = Some(text),
                ChildMsg::Garbage(line) => {
                    if r.outcome == Outcome::Normal {
                        r.outcome = Outcome::Fail;
                        let shown: String = line.chars().take(200).collect();
                        r.why = Some(format!("the tool wrote a line on stdout that is not an event: {shown}"));
                        kill(r.pid, libc::SIGINT);
                        r.killed_at = Some(now);
                        self.note(format!("#{job}: not an event on stdout; interrupting the tool"));
                    }
                }
                ChildMsg::Exited(status) => {
                    self.child_exited(status);
                    return;
                }
            }
        }
        if let Some(ev) = event {
            let group = self.jobs.get(&job).map(|j| j.group.id);
            self.broadcast(ev, group);
        }
    }

    fn child_exited(&mut self, status: ExitStatus) {
        let Some(r) = self.running.take() else { return };
        let id = r.id;
        // a signal is recorded the way Python does, as its negative
        let code = status.code().or_else(|| status.signal().map(|s| -s));
        if !r.ended
            && let Ok(n) = self.http.clear_client(&r.client_id)
            && n > 0
        {
            self.note(format!("#{id}: cleared {n} prompt(s) it left in ComfyUI"));
        }
        if r.outcome == Outcome::Cancel {
            self.apply(queue::cancel(&self.lib, id));
            self.note(format!("#{id} cancelled"));
            return;
        }
        if let Some(ev) = &r.result {
            self.import_result(id, ev, r.seconds, code);
            return;
        }
        if r.outcome == Outcome::Fail {
            let why = r.why.clone().unwrap_or_else(|| "killed".to_owned());
            self.note(format!("#{id} failed: {why}"));
            self.apply(queue::fail(&self.lib, id, &why, code));
            return;
        }
        let comfy_gone = !matches!(self.comfy, Comfy::Up { .. }) || (code != Some(0) && !self.http.answers());
        let attempts = self.jobs.get(&id).map_or(0, |j| j.attempts);
        if code == Some(75) && !comfy_gone && attempts >= 2 {
            // twice "ComfyUI is unreachable" while it answers the daemon: the tool is wrong, not ComfyUI
            let error = "the tool says ComfyUI is unreachable, but it answers rope-queue; see the job's log".to_owned();
            self.note(format!("#{id} failed: {error}"));
            self.apply(queue::fail(&self.lib, id, &error, code));
            return;
        }
        if r.outcome == Outcome::Requeue || self.stopping || code == Some(130) || code == Some(75) || comfy_gone {
            let why = r.why.clone().unwrap_or_else(|| {
                if code == Some(75) || comfy_gone {
                    "ComfyUI was unreachable".to_owned()
                } else {
                    "interrupted".to_owned()
                }
            });
            self.note(format!("#{id} back in line: {why}"));
            self.apply(queue::requeue(&self.lib, id));
            if (code == Some(75) || comfy_gone)
                && let Comfy::Up { misses, .. } = self.comfy
            {
                // make the next tick ask ComfyUI at once
                self.comfy = Comfy::Up { polled: Instant::now() - STATS_EVERY, misses };
            }
            return;
        }
        let error = r.error.clone().or(r.last_line.clone()).unwrap_or_else(|| match (code, status.signal()) {
            (Some(c), _) => format!("the tool exited with code {c} and said nothing"),
            (None, Some(s)) => format!("the tool was killed by signal {s}"),
            _ => "the tool ended without a word".to_owned(),
        });
        self.note(format!("#{id} failed: {error}"));
        self.apply(queue::fail(&self.lib, id, &error, code));
    }

    /// Puts a delivered result into rope. If rope cannot take it, the job fails with its files kept and the
    /// queue pauses: the next job would fail the same way.
    fn import_result(&mut self, id: u64, ev: &ResultEvent, seconds: Option<f64>, exit: Option<i32>) {
        match queue::set_state(&self.lib, id, State::Importing) {
            Ok(j) => self.set(j),
            Err(e) => {
                self.note(format!("#{id}: {e:#}"));
                return;
            }
        }
        let job = self.jobs[&id].clone();
        match import::import(&self.lib, &job, ev, seconds) {
            Ok(result) => {
                let shown = describe_result(Some(&result));
                self.apply(queue::finish(&self.lib, id, result));
                let _ = std::fs::remove_dir_all(queue::out_dir(&self.lib, id));
                self.note(format!("#{id} done -> {shown}"));
            }
            Err(e) => {
                let msg = format!("rope could not take the result: {e:#}");
                self.note(format!("#{id} failed: {msg}"));
                self.apply(queue::fail(&self.lib, id, &msg, exit));
                self.pause(format!(
                    "#{id}: {msg}. Its files are kept in out/; `rope-queue retry {id}` imports them again"
                ));
            }
        }
    }

    // ------------------------------------------------------------ requests

    fn request(&mut self, req: Request, mut responder: Responder) {
        let answer = match req {
            Request::Ping => Ok(json!({
                "pid": std::process::id(), "library": self.lib.root(), "paused": self.paused,
                "comfy": {"version": self.comfy_version, "pid": self.proc.as_ref().map(|p| p.pid),
                          "log": self.proc.as_ref().map(|p| p.log.clone()), "up": matches!(self.comfy, Comfy::Up { .. })},
            })),
            Request::Add(sub) => self.submit(sub),
            Request::Ls => Ok(self.listing()),
            Request::Cancel { jobs, groups } => {
                let (ids, unknown) = self.targets(&jobs, &groups);
                let mut v = self.cancel(&ids);
                v["unknown"] = json!(unknown);
                Ok(v)
            }
            Request::Retry { jobs, groups } => {
                let (ids, unknown) = self.targets(&jobs, &groups);
                let mut v = self.retry(&ids);
                v["unknown"] = json!(unknown);
                Ok(v)
            }
            Request::Pause => {
                if !self.paused {
                    self.pause("by request".to_owned());
                }
                Ok(json!({"paused": true, "finishing": self.running.as_ref().map(|r| r.id)}))
            }
            Request::Resume => {
                self.resume();
                Ok(json!({"paused": false}))
            }
            Request::Stop => {
                let interrupting = self.running.as_ref().map(|r| r.id);
                self.begin_stop();
                Ok(json!({"stopping": true, "interrupting": interrupting}))
            }
            Request::Reload => {
                self.reload = true;
                let after = self.running.as_ref().map(|r| r.id);
                self.note(match after {
                    Some(id) => format!("reload requested; after #{id}"),
                    None => "reload requested".to_owned(),
                });
                Ok(json!({"reload": true, "after": after}))
            }
            Request::Clear => queue::clear(&self.lib).map(|ids| {
                for id in &ids {
                    self.jobs.remove(id);
                }
                json!({"removed": ids})
            }),
            Request::Subscribe { groups } => {
                let jobs: Vec<Job> =
                    self.jobs.values().filter(|j| groups.is_empty() || groups.contains(&j.group.id)).cloned().collect();
                let snapshot =
                    serde_json::to_value(Event::Snapshot { jobs, paused: self.paused }).unwrap_or(Value::Null);
                if responder.send(&protocol::ok(json!({}))).is_ok() && responder.send(&snapshot).is_ok() {
                    self.subs.push((responder, groups));
                }
                return;
            }
        };
        let _ = responder.send(&match answer {
            Ok(v) => protocol::ok(v),
            Err(e) => protocol::error(&format!("{e:#}")),
        });
    }

    fn listing(&self) -> Value {
        let running = self.running.as_ref().map(|r| {
            json!({"job": r.id, "stage": r.stage, "progress": r.progress, "seconds": r.started.elapsed().as_secs()})
        });
        json!({
            "jobs": self.jobs.values().collect::<Vec<_>>(),
            "paused": self.paused, "reason": self.pause_reason, "stopping": self.stopping,
            "running": running,
            "comfy": {"version": self.comfy_version, "up": matches!(self.comfy, Comfy::Up { .. })},
        })
    }

    /// The jobs named, directly or through their groups, and the ids of neither kind.
    fn targets(&self, jobs: &[u64], groups: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let mut ids: Vec<u64> = Vec::new();
        let mut unknown = Vec::new();
        for &j in jobs {
            if self.jobs.contains_key(&j) { ids.push(j) } else { unknown.push(j) }
        }
        for &g in groups {
            let mut members: Vec<u64> = self.jobs.values().filter(|j| j.group.id == g).map(|j| j.id).collect();
            if members.is_empty() {
                unknown.push(g);
            }
            ids.append(&mut members);
        }
        ids.sort_unstable();
        ids.dedup();
        (ids, unknown)
    }

    fn cancel(&mut self, ids: &[u64]) -> Value {
        let (mut cancelled, mut skipped) = (Vec::new(), Vec::new());
        for &id in ids {
            let Some(job) = self.jobs.get(&id) else { continue };
            if job.is_terminal() {
                skipped.push(json!({"job": id, "why": format!("already {}", job.state)}));
                continue;
            }
            if let Some(r) = &mut self.running
                && r.id == id
            {
                r.outcome = Outcome::Cancel;
                kill(r.pid, libc::SIGINT);
                r.killed_at = Some(Instant::now());
                self.note(format!("#{id}: cancelling; interrupting the tool"));
                cancelled.push(id);
                continue;
            }
            match queue::cancel(&self.lib, id) {
                Ok(j) => {
                    self.set(j);
                    self.note(format!("#{id} cancelled"));
                    cancelled.push(id);
                }
                Err(e) => skipped.push(json!({"job": id, "why": format!("{e:#}")})),
            }
        }
        json!({"cancelled": cancelled, "skipped": skipped})
    }

    /// Runs failed or cancelled jobs again as new jobs; a job that failed at import, its result still in `out/`,
    /// is imported again instead.
    fn retry(&mut self, ids: &[u64]) -> Value {
        let (mut retried, mut skipped) = (Vec::new(), Vec::new());
        for &id in ids {
            let Some(job) = self.jobs.get(&id).cloned() else { continue };
            if !matches!(job.state, State::Failed | State::Cancelled) {
                skipped.push(json!({"job": id, "why": format!("is {}", job.state)}));
                continue;
            }
            if job.state == State::Failed
                && let Some(ev) = delivered(&self.lib, id)
            {
                self.note(format!("#{id}: importing its result again"));
                self.import_result(id, &ev, None, None);
                retried.push(json!({"job": id, "imported": self.jobs[&id].state == State::Done}));
                continue;
            }
            for input in &job.inputs {
                if matches!(self.converted.get(&input.id), Some(Converted::Failed(_))) {
                    self.converted.remove(&input.id);
                }
            }
            match queue::retry(&self.lib, id) {
                Ok(new) => {
                    self.note(format!("#{id} runs again as #{}", new.id));
                    retried.push(json!({"job": id, "as": new.id}));
                    self.set(new);
                }
                Err(e) => skipped.push(json!({"job": id, "why": format!("{e:#}")})),
            }
        }
        json!({"retried": retried, "skipped": skipped})
    }

    // ------------------------------------------------------------ submitting

    /// Takes a submission: each request becomes a group of jobs, or is skipped with the reason; the others go on.
    /// The run lock is held shared from importing the inputs to writing the jobs, so purge cannot take an input
    /// in between.
    fn submit(&mut self, sub: Submission) -> Result<Value> {
        self.tools.tool(&sub.tool)?;
        let _hold = self.lib.run_lock_shared()?;
        let mut groups = Vec::new();
        for req in &sub.requests {
            match self.submit_one(&sub, req) {
                Ok(v) => groups.push(v),
                Err(e) => {
                    let why = format!("{e:#}");
                    self.note(format!("skipped a request: {why}"));
                    groups.push(json!({"skipped": why}));
                }
            }
        }
        Ok(json!({"groups": groups}))
    }

    fn submit_one(&mut self, sub: &Submission, req: &SubmitRequest) -> Result<Value> {
        if req.flags.is_empty() {
            bail!("the request expands to no jobs");
        }
        let cwd = Path::new(&sub.submitted.cwd);
        let mut inputs = Vec::with_capacity(req.images.len());
        for img in &req.images {
            inputs.push(self.input(cwd, img, &sub.tool)?);
        }
        let replay = req.replay.as_deref().map(|r| self.resolve(cwd, r)).transpose()?;
        if let Some(into) = &req.into {
            store::ensure_folder(&self.lib, into).with_context(|| format!("--into {into:?}"))?;
        }
        let new = NewGroup {
            tool: sub.tool.clone(),
            prompt: req.prompt.clone(),
            inputs,
            into: req.into.clone(),
            tags: req.tags.clone(),
            note: req.note.clone(),
            fresh: req.fresh,
            replay,
            first_seed: req.first_seed,
            flags: req.flags.clone(),
            submitted: Submitted {
                argv: sub.submitted.argv.clone(),
                cwd: sub.submitted.cwd.clone(),
                at: String::new(),
            },
        };
        let jobs = queue::submit(&self.lib, &new)?;
        let (first, last, group) = (jobs[0].id, jobs[jobs.len() - 1].id, jobs[0].group.id);
        let n = jobs.len();
        for j in jobs {
            self.set(j);
        }
        let ids = if first == last { format!("#{first}") } else { format!("#{first}–#{last}") };
        self.note(format!(
            "queued {ids} (group {group}): {} x{n} → {}",
            sub.tool,
            req.into.as_deref().unwrap_or("(unfiled)")
        ));
        self.idle_since = None;
        Ok(json!({"group": group, "jobs": [first, last]}))
    }

    /// An input as a job records it: `@ID` is looked up; a file inside the library is used as rope has it; any
    /// other file is brought into the library first, so the job uses exactly these bytes however the file fares.
    fn input(&self, cwd: &Path, img: &str, via: &str) -> Result<Input> {
        let path = cwd.join(img);
        let in_library = path.exists() && std::fs::canonicalize(&path).is_ok_and(|c| c.starts_with(self.lib.root()));
        if (img.starts_with('@') && !path.exists()) || in_library {
            let id = store::resolve(&self.lib, if in_library { path.to_str().unwrap_or(img) } else { img })?;
            let meta = store::load(&self.lib, &id)?;
            let name = if meta.name.is_empty() { meta.short().to_owned() } else { meta.name.clone() };
            return Ok(Input { id, name, rendition: None });
        }
        if !path.is_file() {
            bail!("{img}: no such file (looked in {})", cwd.display());
        }
        let added = store::add(&self.lib, &path, &NewImage { via: via.to_owned(), ..Default::default() })?;
        let name =
            path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| short(&added.sha).to_owned());
        Ok(Input { id: added.sha, name, rendition: None })
    }

    fn resolve(&self, cwd: &Path, text: &str) -> Result<String> {
        if text.starts_with('@') && !cwd.join(text).exists() {
            return store::resolve(&self.lib, text);
        }
        let path = cwd.join(text);
        store::resolve(&self.lib, path.to_str().unwrap_or(text))
    }

    // ------------------------------------------------------------ keys

    fn key(&mut self, k: KeyEvent) {
        let selected = self.screen.as_ref().and_then(|s| s.selected());
        match (k.code, k.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) | (KeyCode::Char('q'), _) => self.begin_stop(),
            (KeyCode::Up | KeyCode::Char('k'), _) => {
                if let Some(s) = &mut self.screen {
                    s.move_by(-1);
                }
            }
            (KeyCode::Down | KeyCode::Char('j'), _) => {
                if let Some(s) = &mut self.screen {
                    s.move_by(1);
                }
            }
            (KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right | KeyCode::Left, _) => {
                if let Some(s) = &mut self.screen {
                    s.toggle();
                }
            }
            (KeyCode::Char('p'), _) => {
                if self.paused {
                    self.resume();
                } else {
                    self.pause("from the screen".to_owned());
                }
            }
            (KeyCode::Char('c'), _) => {
                let ids = self.selected_jobs(selected);
                self.cancel(&ids);
            }
            (KeyCode::Char('r'), _) => {
                let ids = self.selected_jobs(selected);
                self.retry(&ids);
            }
            (KeyCode::Char('l'), _) => {
                let id = match selected {
                    Some(RowKind::Job(id)) => Some(id),
                    Some(RowKind::Group(g)) => {
                        // the group's job that says most: the one running, else the last that ran
                        let mut jobs: Vec<&Job> = self.jobs.values().filter(|j| j.group.id == g).collect();
                        jobs.sort_by_key(|j| (j.state != State::Running, j.attempts == 0, std::cmp::Reverse(j.id)));
                        jobs.first().map(|j| j.id)
                    }
                    None => None,
                };
                if let Some(id) = id {
                    let path = queue::log_path(&self.lib, id);
                    if path.exists()
                        && let Some(s) = &mut self.screen
                        && let Err(e) = s.page(&path)
                    {
                        self.note(format!("cannot show the log: {e:#}"));
                    }
                }
            }
            _ => {}
        }
    }

    fn selected_jobs(&self, selected: Option<RowKind>) -> Vec<u64> {
        match selected {
            Some(RowKind::Job(id)) => vec![id],
            Some(RowKind::Group(g)) => self.jobs.values().filter(|j| j.group.id == g).map(|j| j.id).collect(),
            None => Vec::new(),
        }
    }
}

/// The result a child wrote to `out/result.json`, if the job has one.
fn delivered(lib: &Library, id: u64) -> Option<ResultEvent> {
    let text = std::fs::read_to_string(queue::out_dir(lib, id).join("result.json")).ok()?;
    serde_json::from_str(&text).ok()
}

fn running_line(r: &Running) -> String {
    let progress = match r.progress {
        Some((v, m)) if m > 0 => format!(" {v}/{m}"),
        _ => String::new(),
    };
    format!("#{} {}{progress}  {}", r.id, r.stage, hms(r.started.elapsed()))
}

fn describe_result(result: Option<&Value>) -> String {
    let Some(r) = result else { return String::new() };
    let mut s = if let Some(verdict) = r["result"].as_str() {
        // a replay
        match r["id"].as_str() {
            Some(id) => format!("{verdict}; kept as {}", short(id)),
            None => verdict.to_owned(),
        }
    } else {
        format!("{}  {}", r["short"].as_str().unwrap_or("?"), r["folder"].as_str().unwrap_or("/"))
    };
    if let Some(secs) = r["seconds"].as_f64() {
        s += &format!("  {secs:.1}s");
    }
    s
}

fn seed_of(job: &Job) -> String {
    job.flags.iter().find_map(|f| f.strip_prefix("--seed=")).map(|s| format!("seed {s}")).unwrap_or_default()
}

fn group_detail(first: &Job) -> Vec<String> {
    let argv: Vec<String> = first.submitted.argv.iter().map(|a| rope_core::catalog::shell_word(a)).collect();
    vec![
        format!("group #{}  submitted {} in {}", first.group.id, first.submitted.at, first.submitted.cwd),
        argv.join(" "),
    ]
}

fn hms(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}h{:02}m", s / 3600, (s % 3600) / 60) } else { format!("{}:{:02}", s / 60, s % 60) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_and_results_read_well() {
        assert_eq!(hms(Duration::from_secs(6)), "0:06");
        assert_eq!(hms(Duration::from_secs(8000)), "2h13m");
        let r = json!({"short": "3fa9c1e27b0d", "folder": "a/b", "seconds": 6.84});
        assert_eq!(describe_result(Some(&r)), "3fa9c1e27b0d  a/b  6.8s");
        let r = json!({"result": "differs", "id": "f".repeat(64), "compared": "pixels"});
        assert_eq!(describe_result(Some(&r)), "differs; kept as ffffffffffff");
        assert_eq!(describe_result(None), "");
    }
}
