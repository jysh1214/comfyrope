//! The Queue page: what rope-queue is doing, group by group, one row per job. The jobs are the model's, read from
//! the library's `queue/` as the journal reports changes, so the page shows them whether or not a daemon runs;
//! a daemon adds the live part over its socket: the stage and progress of the running job, pause, ComfyUI's state.
//! The controls are the daemon's: cancel, retry, pause, resume, clear. Submitting is the tools' business.

use super::{Action, App};
use crate::thumbs::{self, Key, Source};
use eframe::egui::{self, Color32, Rect, RichText, Sense, Vec2};
use rope_core::hash::short;
use rope_core::queue::protocol::{Event, Request};
use rope_core::queue::{self, Job, State, socket};
use rope_core::repro::shell_join;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// What the Queue page asks for.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Act {
    Select(Option<Sel>),
    Cancel {
        jobs: Vec<u64>,
        groups: Vec<u64>,
    },
    Retry {
        jobs: Vec<u64>,
        groups: Vec<u64>,
    },
    Pause,
    Resume,
    /// Removes the finished jobs: done, failed and cancelled alike.
    Clear,
    Copy(String, &'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Sel {
    Job(u64),
    Group(u64),
}

/// What the thread that follows the daemon sends.
enum Msg {
    /// An `ls` answer, on connecting.
    Listing(Value),
    Event(Event),
    /// No daemon answers.
    Gone,
    /// The daemon on the socket serves this other library; its queue is not ours to touch.
    Other(PathBuf),
}

pub(super) struct Running {
    pub job: u64,
    pub stage: String,
    pub progress: Option<(u32, u32)>,
    pub since: Instant,
}

/// The daemon as the page last heard from it, and what is selected on the page.
pub(super) struct Live {
    rx: Receiver<Msg>,
    /// A daemon for this library answers.
    pub connected: bool,
    /// The daemon on the socket serves another library.
    pub other: Option<PathBuf>,
    pub paused: bool,
    pub reason: Option<String>,
    /// ComfyUI's version and whether it answers.
    pub comfy: Option<(String, bool)>,
    pub stopping: bool,
    pub running: Option<Running>,
    pub selected: Option<Sel>,
    pub show_log: bool,
    /// The selected job's log as last read: job, text, when.
    pub log: Option<(u64, String, Instant)>,
}

impl Live {
    /// Starts following the daemon for the library at `root` on a thread of its own; it reconnects whenever the
    /// daemon comes back.
    pub fn start(ctx: &egui::Context, root: PathBuf) -> Live {
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::Builder::new().name("rope-queue".into()).spawn(move || follow(tx, ctx, root)).expect("thread");
        Live {
            rx,
            connected: false,
            other: None,
            paused: false,
            reason: None,
            comfy: None,
            stopping: false,
            running: None,
            selected: None,
            show_log: false,
            log: None,
        }
    }

    fn running_mut(&mut self, job: u64) -> Option<&mut Running> {
        self.running.as_mut().filter(|r| r.job == job)
    }
}

/// Connects, makes sure the daemon serves our library, asks for the listing, then subscribes and relays every
/// event until the daemon hangs up; tries again two seconds later. Ends when the GUI is gone.
fn follow(tx: Sender<Msg>, ctx: egui::Context, root: PathBuf) {
    loop {
        let msg = match socket::call(&Request::Ping) {
            Ok(ping) => match ping["library"].as_str().map(Path::new) {
                Some(theirs) if !same_dir(theirs, &root) => Msg::Other(theirs.to_owned()),
                _ => match socket::call(&Request::Ls) {
                    Ok(v) => Msg::Listing(v),
                    Err(_) => Msg::Gone,
                },
            },
            Err(_) => Msg::Gone,
        };
        let ours = matches!(msg, Msg::Listing(_));
        if tx.send(msg).is_err() {
            return;
        }
        ctx.request_repaint();
        if !ours {
            std::thread::sleep(Duration::from_secs(2));
            continue;
        }
        let _ = socket::stream(&Request::Subscribe { groups: vec![] }, |v| {
            // the first line answers the request; the events follow
            if v.get("event").is_none() {
                return Ok(true);
            }
            if let Ok(ev) = serde_json::from_value::<Event>(v) {
                tx.send(Msg::Event(ev)).map_err(|_| anyhow::anyhow!("the GUI is gone"))?;
                ctx.request_repaint();
            }
            Ok(true)
        });
        if tx.send(Msg::Gone).is_err() {
            return;
        }
        ctx.request_repaint();
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// Whether two paths name the same directory, through symlinks.
fn same_dir(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

const THUMB: f32 = 40.0;
const ROW: f32 = 44.0;

impl App {
    /// Takes in what the follower thread has sent since the last frame.
    pub(super) fn queue_poll(&mut self) {
        while let Ok(msg) = self.queue.rx.try_recv() {
            match msg {
                Msg::Listing(v) => {
                    let q = &mut self.queue;
                    q.connected = true;
                    q.other = None;
                    q.paused = v["paused"].as_bool().unwrap_or(false);
                    q.reason = v["reason"].as_str().map(str::to_owned);
                    q.stopping = v["stopping"].as_bool().unwrap_or(false);
                    q.comfy = v["comfy"]["version"]
                        .as_str()
                        .map(|ver| (ver.to_owned(), v["comfy"]["up"].as_bool().unwrap_or(false)));
                    q.running = v["running"]["job"].as_u64().map(|job| {
                        let r = &v["running"];
                        let progress = r["progress"][0].as_u64().zip(r["progress"][1].as_u64());
                        Running {
                            job,
                            stage: r["stage"].as_str().unwrap_or("").to_owned(),
                            progress: progress.map(|(a, b)| (a as u32, b as u32)),
                            since: Instant::now() - Duration::from_secs(r["seconds"].as_u64().unwrap_or(0)),
                        }
                    });
                    if let Ok(jobs) = serde_json::from_value::<Vec<Job>>(v["jobs"].clone()) {
                        self.merge_jobs(jobs);
                    }
                }
                Msg::Event(Event::Snapshot { jobs, paused }) => {
                    self.queue.paused = paused;
                    self.merge_jobs(jobs);
                }
                Msg::Event(Event::State { job }) => {
                    let job = *job;
                    let was = self.queue.running.as_ref().is_some_and(|r| r.job == job.id);
                    if job.state == State::Running {
                        if !was {
                            self.queue.running = Some(Running {
                                job: job.id,
                                stage: String::new(),
                                progress: None,
                                since: Instant::now(),
                            });
                        }
                    } else if was {
                        self.queue.running = None;
                    }
                    self.merge_jobs(vec![job]);
                }
                Msg::Event(Event::Stage { job, text }) => {
                    if let Some(r) = self.queue.running_mut(job) {
                        r.stage = text;
                        r.progress = None;
                    }
                }
                Msg::Event(Event::Progress { job, value, max }) => {
                    if let Some(r) = self.queue.running_mut(job) {
                        r.progress = Some((value, max));
                    }
                }
                Msg::Event(Event::Paused { paused, reason }) => {
                    self.queue.paused = paused;
                    self.queue.reason = reason;
                }
                Msg::Event(Event::Stopping) => self.queue.stopping = true,
                Msg::Gone | Msg::Other(_) => {
                    let q = &mut self.queue;
                    q.other = match msg {
                        Msg::Other(p) => Some(p),
                        _ => None,
                    };
                    q.connected = false;
                    q.running = None;
                    q.stopping = false;
                    q.comfy = None;
                }
            }
        }
    }

    /// Jobs as the daemon tells them, over the model's copy; the journal brings the same change a moment later.
    fn merge_jobs(&mut self, jobs: Vec<Job>) {
        let Some(m) = &mut self.model else { return };
        for job in jobs {
            match m.jobs.iter_mut().find(|j| j.id == job.id) {
                Some(j) => *j = job,
                None => m.jobs.push(job),
            }
        }
        m.jobs.sort_by_key(|j| j.id);
    }

    pub(super) fn queue_act(&mut self, act: Act, ctx: Option<&egui::Context>) -> anyhow::Result<()> {
        match act {
            Act::Select(sel) => {
                if sel != self.queue.selected {
                    self.queue.log = None;
                }
                self.queue.selected = sel;
            }
            Act::Cancel { jobs, groups } => {
                let v = self.daemon(&Request::Cancel { jobs, groups })?;
                let n = v["cancelled"].as_array().map_or(0, Vec::len);
                self.say(format!("{n} job(s) cancelled{}", skipped(&v["skipped"])), false);
            }
            Act::Retry { jobs, groups } => {
                let v = self.daemon(&Request::Retry { jobs, groups })?;
                let n = v["retried"].as_array().map_or(0, Vec::len);
                self.say(format!("{n} job(s) queued again{}", skipped(&v["skipped"])), false);
            }
            Act::Pause => {
                let v = self.daemon(&Request::Pause)?;
                let text = match v["finishing"].as_u64() {
                    Some(id) => format!("paused; #{id} finishes first"),
                    None => "paused".to_owned(),
                };
                self.queue.paused = true;
                self.say(text, false);
            }
            Act::Resume => {
                self.daemon(&Request::Resume)?;
                self.queue.paused = false;
                self.say("resumed", false);
            }
            Act::Clear => {
                let removed: Vec<u64> = if self.queue.connected {
                    serde_json::from_value(self.daemon(&Request::Clear)?["removed"].clone())?
                } else {
                    // no daemon for this library: the lock says so, and keeps one from starting halfway through
                    let Some(_lock) = self.lib.try_queue_lock()? else {
                        anyhow::bail!("rope-queue is running for this library but does not answer")
                    };
                    queue::clear(&self.lib)?
                };
                if let Some(m) = &mut self.model {
                    m.jobs.retain(|j| !removed.contains(&j.id));
                }
                if let Some(Sel::Job(id)) = self.queue.selected
                    && removed.contains(&id)
                {
                    self.queue.selected = None;
                }
                self.say(format!("{} finished job(s) cleared", removed.len()), false);
            }
            Act::Copy(text, what) => self.copy(ctx, text, what),
        }
        Ok(())
    }

    /// Asks the daemon for this library; without one, says so.
    fn daemon(&self, req: &Request) -> anyhow::Result<Value> {
        if let Some(other) = &self.queue.other {
            anyhow::bail!("rope-queue is running for another library, {}", other.display());
        }
        if !self.queue.connected {
            anyhow::bail!("rope-queue is not running");
        }
        socket::call(req)
    }

    /// The middle: the status line, then every group newest first, its jobs in order.
    pub(super) fn queue_page(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        self.queue_status(ui, actions);
        ui.separator();
        if self.queue.running.is_some() {
            ui.ctx().request_repaint_after(Duration::from_secs(1));
        }
        // the model is taken out so that thumbnails can be fetched while its jobs are read
        let Some(model) = self.model.take() else { return };
        egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            let mut groups: BTreeMap<u64, Vec<&Job>> = BTreeMap::new();
            for job in &model.jobs {
                groups.entry(job.group.id).or_default().push(job);
            }
            if groups.is_empty() {
                ui.add_space(12.0);
                ui.label(RichText::new("No jobs. img2img.py and img2vid.py put theirs here.").weak());
            }
            for (gid, jobs) in groups.iter().rev() {
                self.group_header(ui, *gid, jobs, actions);
                for job in jobs {
                    self.job_row(ui, &model, job, actions);
                }
                ui.add_space(10.0);
            }
        });
        self.model = Some(model);
    }

    fn queue_status(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let waiting = self.model().jobs.iter().filter(|j| !j.is_terminal()).count();
        ui.horizontal(|ui| {
            if let Some(other) = &self.queue.other {
                ui.label(RichText::new("rope-queue is running for another library").strong());
                ui.label(RichText::new(other.display().to_string()).weak())
                    .on_hover_text("the socket is one per user; stop that daemon to start one for this library");
            } else if !self.queue.connected {
                ui.label(RichText::new("rope-queue is not running").strong());
                ui.label(RichText::new(start_hint(&self.lib)).weak())
                    .on_hover_text("the daemon runs the jobs; start it in a terminal");
            } else {
                ui.label(RichText::new("rope-queue running").strong());
                ui.label(RichText::new("·").weak());
                let comfy = match &self.queue.comfy {
                    Some((ver, true)) => RichText::new(format!("ComfyUI {ver} up")),
                    Some((ver, false)) => {
                        RichText::new(format!("ComfyUI {ver} down")).color(ui.visuals().error_fg_color)
                    }
                    None => RichText::new("ComfyUI starting").weak(),
                };
                ui.label(comfy);
                ui.label(RichText::new("·").weak());
                ui.label(format!("{waiting} to do"));
                if self.queue.stopping {
                    ui.label(RichText::new("· stopping").color(ui.visuals().warn_fg_color));
                } else if self.queue.paused {
                    let reason = self.queue.reason.clone().unwrap_or_default();
                    ui.label(RichText::new(format!("· paused: {reason}")).color(ui.visuals().warn_fg_color));
                    if ui.small_button("Resume").clicked() {
                        actions.push(Action::Queue(Act::Resume));
                    }
                } else if ui.small_button("Pause").on_hover_text("the running job finishes; the next waits").clicked() {
                    actions.push(Action::Queue(Act::Pause));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let finished = self.model().jobs.iter().filter(|j| j.is_terminal()).count();
                let button = egui::Button::new("Clear finished");
                if ui
                    .add_enabled(finished > 0, button)
                    .on_hover_text("forget the done, failed and cancelled jobs")
                    .clicked()
                {
                    actions.push(Action::Queue(Act::Clear));
                }
            });
        });
    }

    fn group_header(&mut self, ui: &mut egui::Ui, gid: u64, jobs: &[&Job], actions: &mut Vec<Action>) {
        let first = jobs[0];
        let selected = self.queue.selected == Some(Sel::Group(gid));
        let fill =
            if selected { ui.visuals().selection.bg_fill.gamma_multiply(0.35) } else { ui.visuals().faint_bg_color };
        let done = jobs.iter().filter(|j| j.state == State::Done).count();
        let open = jobs.iter().filter(|j| !j.is_terminal()).count();
        let failed = jobs.iter().filter(|j| matches!(j.state, State::Failed | State::Cancelled)).count();
        let header = egui::UiBuilder::new().id_salt(("group", gid)).sense(Sense::click());
        let resp = ui
            .scope_builder(header, |ui| {
                egui::Frame::group(ui.style()).fill(fill).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.add(egui::Label::new(RichText::new(format!("g{gid}")).strong()).selectable(false));
                        let command = shell_join(&first.submitted.argv);
                        ui.add(
                            egui::Label::new(RichText::new(command).monospace().small()).truncate().selectable(false),
                        );
                    });
                    ui.horizontal(|ui| {
                        let mut parts = vec![format!("{} job(s)", jobs.len()), format!("{done} done")];
                        if open > 0 {
                            parts.push(format!("{open} to do"));
                        }
                        if failed > 0 {
                            parts.push(format!("{failed} failed or cancelled"));
                        }
                        parts.push(match &first.into {
                            Some(f) => format!("→ {f}"),
                            None => "→ unfiled".to_owned(),
                        });
                        parts.push(clock(&first.created));
                        ui.add(egui::Label::new(RichText::new(parts.join("  ·  ")).weak().small()).selectable(false));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let bar = egui::ProgressBar::new(done as f32 / jobs.len().max(1) as f32)
                                .desired_width(120.0)
                                .desired_height(10.0);
                            ui.add(bar);
                        });
                    });
                });
            })
            .response;
        if resp.clicked() {
            actions.push(Action::Queue(Act::Select(Some(Sel::Group(gid)))));
        }
        resp.context_menu(|ui| {
            if ui.add_enabled(open > 0, egui::Button::new("Cancel group")).clicked() {
                actions.push(Action::Queue(Act::Cancel { jobs: vec![], groups: vec![gid] }));
                ui.close();
            }
            let again: Vec<u64> =
                jobs.iter().filter(|j| matches!(j.state, State::Failed | State::Cancelled)).map(|j| j.id).collect();
            if ui.add_enabled(!again.is_empty(), egui::Button::new("Retry failed and cancelled")).clicked() {
                actions.push(Action::Queue(Act::Retry { jobs: again, groups: vec![] }));
                ui.close();
            }
            if ui.button("Copy command").clicked() {
                actions.push(Action::Queue(Act::Copy(shell_join(&first.submitted.argv), "the command")));
                ui.close();
            }
        });
    }

    fn job_row(&mut self, ui: &mut egui::Ui, model: &crate::model::Model, job: &Job, actions: &mut Vec<Action>) {
        let selected = self.queue.selected == Some(Sel::Job(job.id));
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), ROW), Sense::click());
        if selected {
            ui.painter().rect_filled(rect, 3.0, ui.visuals().selection.bg_fill.gamma_multiply(0.35));
        } else if resp.hovered() {
            ui.painter().rect_filled(rect, 3.0, ui.visuals().widgets.hovered.weak_bg_fill);
        }
        let result_id = job.result.as_ref().and_then(|r| r["id"].as_str()).map(str::to_owned);
        let row = egui::UiBuilder::new().max_rect(rect.shrink2(Vec2::new(6.0, 2.0)));
        ui.scope_builder(row, |ui| {
            ui.horizontal_centered(|ui| {
                // the input, its name, then the job
                let input = job.inputs.first();
                self.thumb_at(ui, model, input.map(|i| i.id.as_str()), THUMB);
                let name = match job.inputs.len() {
                    0 => "–".to_owned(),
                    1 => job.inputs[0].name.clone(),
                    n => format!("{} +{}", job.inputs[0].name, n - 1),
                };
                column(ui, 120.0, |ui| {
                    ui.add(egui::Label::new(RichText::new(name).small()).truncate().selectable(false))
                });
                column(ui, 48.0, |ui| {
                    ui.add(egui::Label::new(RichText::new(format!("#{}", job.id)).weak()).selectable(false))
                });
                column(ui, 84.0, |ui| ui.add(egui::Label::new(state_text(ui, job.state)).selectable(false)));
                column(ui, 64.0, |ui| {
                    ui.add(egui::Label::new(RichText::new(seed_of(job)).weak().small()).selectable(false))
                });
                // what came of it
                if let Some(id) = &result_id {
                    self.thumb_at(ui, model, Some(id), THUMB);
                }
                let outcome = self.outcome(job);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(egui::Label::new(RichText::new(took(job, &self.queue)).weak().small()).selectable(false));
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add(egui::Label::new(outcome).truncate().selectable(false));
                    });
                });
            });
        });
        if resp.clicked() {
            actions.push(Action::Queue(Act::Select(Some(Sel::Job(job.id)))));
        }
        if resp.double_clicked()
            && let Some(id) = &result_id
        {
            actions.push(Action::Open(id.clone()));
        }
        resp.context_menu(|ui| {
            if ui.add_enabled(!job.is_terminal(), egui::Button::new("Cancel")).clicked() {
                actions.push(Action::Queue(Act::Cancel { jobs: vec![job.id], groups: vec![] }));
                ui.close();
            }
            let again = matches!(job.state, State::Failed | State::Cancelled);
            if ui.add_enabled(again, egui::Button::new("Retry")).clicked() {
                actions.push(Action::Queue(Act::Retry { jobs: vec![job.id], groups: vec![] }));
                ui.close();
            }
            if let Some(id) = &result_id
                && ui.button("Open in library").clicked()
            {
                actions.push(Action::Open(id.clone()));
                ui.close();
            }
            if ui.button("Copy command").clicked() {
                actions.push(Action::Queue(Act::Copy(shell_join(&job.submitted.argv), "the command")));
                ui.close();
            }
        });
    }

    /// A thumbnail of `id` in a square of `size`, an outline when the image is not in the library.
    fn thumb_at(&mut self, ui: &mut egui::Ui, model: &crate::model::Model, id: Option<&str>, size: f32) {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
        let Some(item) = id.and_then(|id| model.items.get(id)) else {
            ui.painter().rect_stroke(rect, 3.0, ui.visuals().widgets.inactive.bg_stroke, egui::StrokeKind::Inside);
            return;
        };
        let (file, fallback) = (item.file.clone(), item.fallback.clone());
        let key = Key { id: item.id.clone(), size: thumbs::GRID };
        if let Some(tex) = self.thumbs.get(&key, || Source { file, fallback }) {
            let shown = super::fit(tex.size_vec2(), rect.size());
            egui::Image::new(&tex).paint_at(ui, Rect::from_center_size(rect.center(), shown));
        }
    }

    /// The right-hand words of a row: the result, the stage, the error.
    fn outcome(&self, job: &Job) -> RichText {
        match job.state {
            State::Running => match &self.queue.running {
                Some(r) if r.job == job.id => {
                    let progress = match r.progress {
                        Some((v, m)) if m > 0 => format!(" {v}/{m}"),
                        _ => String::new(),
                    };
                    RichText::new(format!("{}{progress}", r.stage)).small()
                }
                _ => RichText::new("running").weak().small(),
            },
            State::Failed => {
                let why = job.error.as_deref().unwrap_or("failed").lines().next().unwrap_or("").to_owned();
                RichText::new(why).color(Color32::from_rgb(220, 90, 90)).small()
            }
            State::Done => RichText::new(result_words(job)).small(),
            State::Queued | State::Ready => RichText::new("waiting").weak().small(),
            State::Converting => RichText::new("converting the input").weak().small(),
            State::Importing => RichText::new("importing").weak().small(),
            State::Cancelled => RichText::new("").small(),
        }
    }

    /// The right panel: the selected job, the selected group, or nothing.
    pub(super) fn queue_details(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        match self.queue.selected {
            Some(Sel::Job(id)) => match self.model().job(id).cloned() {
                Some(job) => self.job_details(ui, &job, actions),
                None => self.queue.selected = None,
            },
            Some(Sel::Group(gid)) => {
                let jobs: Vec<Job> = self.model().jobs.iter().filter(|j| j.group.id == gid).cloned().collect();
                if jobs.is_empty() {
                    self.queue.selected = None;
                } else {
                    group_details(ui, gid, &jobs, actions);
                }
            }
            None => {
                ui.add_space(6.0);
                ui.label(RichText::new("Select a job or a group.").weak());
                ui.add_space(6.0);
                ui.label(RichText::new("Double-click a done job to open its result in the library.").weak().small());
                ui.label(RichText::new(format!("socket: {}", socket::path().display())).weak().small());
            }
        }
    }

    fn job_details(&mut self, ui: &mut egui::Ui, job: &Job, actions: &mut Vec<Action>) {
        egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(format!("#{}", job.id));
                ui.label(state_text(ui, job.state));
                ui.label(
                    RichText::new(format!("g{}, {} of {}", job.group.id, job.group.index, job.group.count)).weak(),
                );
            });
            if job.state == State::Running
                && let Some(r) = &self.queue.running
                && r.job == job.id
            {
                let progress = match r.progress {
                    Some((v, m)) if m > 0 => format!("{} {v}/{m}", r.stage),
                    _ => r.stage.clone(),
                };
                ui.label(RichText::new(progress).small());
                if let Some((v, m)) = r.progress
                    && m > 0
                {
                    ui.add(egui::ProgressBar::new(v as f32 / m as f32).desired_height(8.0));
                }
            }
            if let Some(err) = &job.error {
                ui.add_space(4.0);
                ui.add(egui::Label::new(RichText::new(err).color(ui.visuals().error_fg_color)).wrap());
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(!job.is_terminal(), egui::Button::new("Cancel")).clicked() {
                    actions.push(Action::Queue(Act::Cancel { jobs: vec![job.id], groups: vec![] }));
                }
                let again = matches!(job.state, State::Failed | State::Cancelled);
                if ui
                    .add_enabled(again, egui::Button::new("Retry"))
                    .on_hover_text("queue it again, at the end")
                    .clicked()
                {
                    actions.push(Action::Queue(Act::Retry { jobs: vec![job.id], groups: vec![] }));
                }
                if ui.selectable_label(self.queue.show_log, "Log").on_hover_text("what the job printed").clicked() {
                    self.queue.show_log = !self.queue.show_log;
                }
            });
            ui.add_space(8.0);

            // the command, its inputs, the prompt and the flags: what the job is
            section(ui, "Command");
            ui.horizontal(|ui| {
                let command = shell_join(&job.submitted.argv);
                if ui.small_button("Copy").clicked() {
                    actions.push(Action::Queue(Act::Copy(command.clone(), "the command")));
                }
                ui.add(egui::Label::new(RichText::new(command).monospace().small()).wrap());
            });
            ui.label(RichText::new(format!("in {}", job.submitted.cwd)).weak().small());
            if !job.inputs.is_empty() || job.replay.is_some() {
                ui.add_space(6.0);
                section(ui, if job.replay.is_some() { "Replays" } else { "Inputs" });
                ui.horizontal_wrapped(|ui| {
                    if let Some(id) = &job.replay {
                        self.small_thumb(ui, id, actions);
                    }
                    for input in &job.inputs {
                        self.small_thumb(ui, &input.id, actions);
                    }
                });
                let names: Vec<&str> = job.inputs.iter().map(|i| i.name.as_str()).collect();
                ui.label(RichText::new(names.join(", ")).weak().small());
            }
            if !job.prompt.is_empty() {
                ui.add_space(6.0);
                section(ui, "Prompt");
                ui.add(egui::Label::new(RichText::new(&job.prompt).small()).wrap());
            }
            if !job.flags.is_empty() {
                ui.add_space(6.0);
                section(ui, "Flags");
                ui.add(egui::Label::new(RichText::new(shell_join(&job.flags)).monospace().small()).wrap());
            }
            ui.add_space(6.0);
            section(ui, "Goes to");
            let mut lines = vec![job.into.clone().unwrap_or_else(|| "unfiled".to_owned())];
            if !job.tags.is_empty() {
                lines.push(format!("tags: {}", job.tags.join(", ")));
            }
            if let Some(n) = &job.note {
                lines.push(format!("note: {n}"));
            }
            if job.fresh {
                lines.push("fresh: the inputs are uploaded again".to_owned());
            }
            if let Some(of) = job.retry_of {
                lines.push(format!("retry of #{of}"));
            }
            ui.add(egui::Label::new(RichText::new(lines.join("\n")).small()).wrap());

            ui.add_space(6.0);
            section(ui, "When");
            let mut when = vec![format!("submitted {}", clock(&job.created))];
            if let Some(s) = &job.started {
                when.push(format!("started {}", clock(s)));
            }
            if let Some(e) = &job.ended {
                when.push(format!("ended {}", clock(e)));
            }
            if job.attempts > 1 {
                when.push(format!("{} attempts", job.attempts));
            }
            ui.label(RichText::new(when.join("  ·  ")).weak().small());

            if let Some(r) = &job.result {
                ui.add_space(6.0);
                section(ui, "Result");
                ui.horizontal(|ui| {
                    if let Some(id) = r["id"].as_str() {
                        self.small_thumb(ui, id, actions);
                        ui.vertical(|ui| {
                            ui.add(egui::Label::new(RichText::new(result_words(job)).small()).wrap());
                            if ui.small_button("Open in library").clicked() {
                                actions.push(Action::Open(id.to_owned()));
                            }
                        });
                    } else {
                        ui.add(egui::Label::new(RichText::new(result_words(job)).small()).wrap());
                    }
                });
            }

            if self.queue.show_log {
                ui.add_space(8.0);
                section(ui, "Log");
                let text = self.log_tail(job);
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    let shown = if text.is_empty() { "nothing yet" } else { text.as_str() };
                    ui.add(egui::Label::new(RichText::new(shown).monospace().small()).wrap());
                });
                if job.state == State::Running {
                    ui.ctx().request_repaint_after(Duration::from_secs(1));
                }
            }
        });
    }

    /// The last lines of the job's log, read again every second while it runs.
    fn log_tail(&mut self, job: &Job) -> String {
        let fresh = match &self.queue.log {
            Some((id, _, at)) => *id != job.id || (!job.is_terminal() && at.elapsed() > Duration::from_secs(1)),
            None => true,
        };
        if fresh {
            let text = std::fs::read_to_string(queue::log_path(&self.lib, job.id)).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            let keep = lines.len().saturating_sub(80);
            self.queue.log = Some((job.id, lines[keep..].join("\n"), Instant::now()));
        }
        self.queue.log.as_ref().map(|(_, t, _)| t.clone()).unwrap_or_default()
    }
}

fn group_details(ui: &mut egui::Ui, gid: u64, jobs: &[Job], actions: &mut Vec<Action>) {
    let first = &jobs[0];
    ui.heading(format!("Group g{gid}"));
    ui.label(RichText::new(format!("{} job(s), #{}–#{}", jobs.len(), first.id, jobs[jobs.len() - 1].id)).weak());
    ui.add_space(6.0);
    let open = jobs.iter().filter(|j| !j.is_terminal()).count();
    let again: Vec<u64> =
        jobs.iter().filter(|j| matches!(j.state, State::Failed | State::Cancelled)).map(|j| j.id).collect();
    ui.horizontal(|ui| {
        if ui.add_enabled(open > 0, egui::Button::new("Cancel group")).clicked() {
            actions.push(Action::Queue(Act::Cancel { jobs: vec![], groups: vec![gid] }));
        }
        if ui.add_enabled(!again.is_empty(), egui::Button::new("Retry failed and cancelled")).clicked() {
            actions.push(Action::Queue(Act::Retry { jobs: again.clone(), groups: vec![] }));
        }
    });
    ui.add_space(8.0);
    section(ui, "Command");
    ui.horizontal(|ui| {
        let command = shell_join(&first.submitted.argv);
        if ui.small_button("Copy").clicked() {
            actions.push(Action::Queue(Act::Copy(command.clone(), "the command")));
        }
        ui.add(egui::Label::new(RichText::new(command).monospace().small()).wrap());
    });
    ui.label(RichText::new(format!("in {}, {}", first.submitted.cwd, clock(&first.created))).weak().small());
    ui.add_space(6.0);
    section(ui, "Jobs");
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for j in jobs {
        *counts.entry(j.state.as_str()).or_default() += 1;
    }
    let words: Vec<String> =
        State::ALL.iter().filter_map(|s| counts.get(s.as_str()).map(|n| format!("{n} {s}"))).collect();
    ui.label(RichText::new(words.join(", ")).small());
    let goes = first.into.clone().unwrap_or_else(|| "unfiled".to_owned());
    ui.label(RichText::new(format!("→ {goes}")).weak().small());
}

/// How to start the daemon for this library, as the status line says it.
fn start_hint(lib: &rope_core::Library) -> String {
    let library = if lib.is_default() { String::new() } else { format!(" --library {}", lib.root().display()) };
    format!("start it in a terminal: rope-queue{library} --repo <comfyrope>")
}

fn skipped(v: &Value) -> String {
    let Some(list) = v.as_array().filter(|l| !l.is_empty()) else { return String::new() };
    let whys: Vec<String> = list
        .iter()
        .map(|s| format!("#{} {}", s["job"].as_u64().unwrap_or(0), s["why"].as_str().unwrap_or("?")))
        .collect();
    format!("; skipped {}", whys.join(", "))
}

/// A column of fixed width in a row.
fn column(ui: &mut egui::Ui, width: f32, add: impl FnOnce(&mut egui::Ui) -> egui::Response) {
    ui.allocate_ui_with_layout(Vec2::new(width, ROW - 4.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.set_min_width(width);
        ui.set_max_width(width);
        add(ui);
    });
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title).strong().small());
}

fn state_text(ui: &egui::Ui, state: State) -> RichText {
    let text = RichText::new(state.as_str());
    match state {
        State::Queued | State::Ready | State::Converting => text.weak(),
        State::Running | State::Importing => text.color(ui.visuals().selection.stroke.color).strong(),
        State::Done => text.color(Color32::from_rgb(90, 180, 90)),
        State::Failed => text.color(ui.visuals().error_fg_color),
        State::Cancelled => text.weak().strikethrough(),
    }
}

/// `s42` from the job's `--seed=42`.
fn seed_of(job: &Job) -> String {
    job.flags
        .iter()
        .find_map(|f| f.strip_prefix("--seed=").or_else(|| f.strip_prefix("-s=")))
        .map(|s| format!("s{s}"))
        .unwrap_or_default()
}

/// `17:56:46` from `2026-10-17T17:56:46+08:00`.
fn clock(iso: &str) -> String {
    iso.get(11..19).unwrap_or(iso).to_owned()
}

/// How long the job took, or has been going.
fn took(job: &Job, live: &Live) -> String {
    if job.state == State::Running
        && let Some(r) = &live.running
        && r.job == job.id
    {
        return hms(r.since.elapsed());
    }
    match job.result.as_ref().and_then(|r| r["seconds"].as_f64()) {
        Some(s) if s > 0.0 => format!("{s:.1}s"),
        _ => String::new(),
    }
}

fn hms(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}h{:02}m", s / 3600, (s % 3600) / 60) } else { format!("{}:{:02}", s / 60, s % 60) }
}

/// The result in words: the name and folder of a new image, the verdict of a replay.
fn result_words(job: &Job) -> String {
    let Some(r) = &job.result else { return String::new() };
    if let Some(verdict) = r["result"].as_str() {
        return match r["id"].as_str() {
            Some(id) => format!("replay {verdict}; kept as {}", short(id)),
            None => format!("replay {verdict}"),
        };
    }
    let name = r["name"].as_str().or_else(|| r["short"].as_str()).unwrap_or("?");
    match r["folder"].as_str() {
        Some(f) if !f.is_empty() => format!("{name}  → {f}"),
        _ => format!("{name}  → unfiled"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_for_rows() {
        assert_eq!(clock("2026-10-17T17:56:46+08:00"), "17:56:46");
        assert_eq!(clock("soon"), "soon");
        assert_eq!(hms(Duration::from_secs(75)), "1:15");
        assert_eq!(hms(Duration::from_secs(3700)), "1h01m");
        assert_eq!(skipped(&serde_json::json!([])), "");
        assert_eq!(skipped(&serde_json::json!([{"job": 3, "why": "already done"}])), "; skipped #3 already done");
    }
}
