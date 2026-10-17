//! ComfyUI: the process rope-queue starts and owns, and the few HTTP calls made to it. The websocket, the
//! uploads and the prompts are the Python child's business.

use anyhow::{Context, Result, bail};
use rope_core::{Library, queue, time};
use serde_json::{Value, json};
use std::fs::{self, File};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use ureq::Agent;

/// How many ComfyUI logs are kept.
const LOGS_KEPT: usize = 10;

pub struct Http {
    base: String,
    agent: Agent,
}

/// A prompt in ComfyUI's queue.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub prompt_id: String,
    pub client_id: Option<String>,
    pub running: bool,
}

impl Http {
    /// `server` is `host:port`, as `$COMFYUI_SERVER` has it.
    pub fn new(server: &str) -> Http {
        // no pooled connections: a server that closes after each answer would otherwise cost a timeout
        let config = Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(5)))
            .http_status_as_error(false)
            .max_idle_connections(0)
            .build();
        Http { base: format!("http://{}", server.trim_end_matches('/')), agent: Agent::new_with_config(config) }
    }

    fn get(&self, path: &str) -> Result<Value> {
        let url = format!("{}{path}", self.base);
        let mut resp = self.agent.get(&url).call().with_context(|| format!("GET {url}"))?;
        let status = resp.status().as_u16();
        let text = resp.body_mut().read_to_string().with_context(|| format!("GET {url}"))?;
        if status >= 400 {
            bail!("GET {url}: HTTP {status}");
        }
        serde_json::from_str(&text).with_context(|| format!("GET {url}: not JSON"))
    }

    fn post(&self, path: &str, body: &Value) -> Result<()> {
        let url = format!("{}{path}", self.base);
        let resp = self
            .agent
            .post(&url)
            .header("Content-Type", "application/json")
            .send(body.to_string().as_str())
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status().as_u16();
        if status >= 400 {
            bail!("POST {url}: HTTP {status}");
        }
        Ok(())
    }

    pub fn stats(&self) -> Result<Value> {
        self.get("/system_stats")
    }

    pub fn answers(&self) -> bool {
        self.stats().is_ok()
    }

    pub fn version(stats: &Value) -> String {
        stats["system"]["comfyui_version"].as_str().unwrap_or("?").to_owned()
    }

    /// Every prompt queued or running, the running ones first.
    pub fn queue(&self) -> Result<Vec<Item>> {
        let v = self.get("/queue")?;
        let mut out = Vec::new();
        for (key, running) in [("queue_running", true), ("queue_pending", false)] {
            for entry in v[key].as_array().into_iter().flatten() {
                // [number, prompt_id, prompt, extra_data, outputs_to_execute]
                let Some(prompt_id) = entry.get(1).and_then(Value::as_str) else { continue };
                let client_id = entry.get(3).and_then(|x| x["client_id"].as_str()).map(str::to_owned);
                out.push(Item { prompt_id: prompt_id.to_owned(), client_id, running });
            }
        }
        Ok(out)
    }

    pub fn delete(&self, prompt_ids: &[String]) -> Result<()> {
        self.post("/queue", &json!({"delete": prompt_ids}))
    }

    pub fn interrupt(&self, prompt_id: Option<&str>) -> Result<()> {
        let body = match prompt_id {
            Some(p) => json!({"prompt_id": p}),
            None => json!({}),
        };
        self.post("/interrupt", &body)
    }

    /// Hands the VRAM back: models out of VRAM (they stay in RAM) and the cache of finished results dropped.
    pub fn free(&self) -> Result<()> {
        self.post("/free", &json!({"unload_models": true, "free_memory": true}))
    }

    /// Removes every prompt of `client_id`, interrupting the one running if it is among them. Answers how many.
    pub fn clear_client(&self, client_id: &str) -> Result<usize> {
        let mine: Vec<Item> = self.queue()?.into_iter().filter(|i| i.client_id.as_deref() == Some(client_id)).collect();
        let pending: Vec<String> = mine.iter().filter(|i| !i.running).map(|i| i.prompt_id.clone()).collect();
        if !pending.is_empty() {
            self.delete(&pending)?;
        }
        for running in mine.iter().filter(|i| i.running) {
            self.interrupt(Some(&running.prompt_id))?;
        }
        Ok(mine.len())
    }
}

/// The ComfyUI process. The daemon holds exactly one, started by it or handed over by the daemon it reloaded.
pub struct Process {
    pub pid: i32,
    pub log: PathBuf,
    pub started: Instant,
    exited: bool,
}

impl Process {
    /// Starts ComfyUI through the repo's `run_comfyui.sh`, in its own session, with its output in `log`, and with
    /// SIGINT delivered to it should this process die. The death signal is tied to the thread that spawns, so
    /// this must be called on the main thread, which lives as long as the daemon.
    pub fn start(repo: &Path, port: u16, reserve_vram: Option<&str>, extra: &[String], log: &Path) -> Result<Process> {
        let out = File::create(log).with_context(|| format!("cannot create {}", log.display()))?;
        let err = out.try_clone()?;
        let script = repo.join("run_comfyui.sh");
        let mut cmd = Command::new(&script);
        cmd.args(["--listen", "127.0.0.1", "--port", &port.to_string()]);
        if let Some(gb) = reserve_vram {
            cmd.args(["--reserve-vram", gb]);
        }
        cmd.args(extra).current_dir(repo).stdin(Stdio::null()).stdout(out).stderr(err);
        // SAFETY: setsid, prctl and signal are async-signal-safe and touch nothing of the parent.
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGINT as libc::c_ulong, 0, 0, 0);
                default_signals();
                Ok(())
            });
        }
        let child = cmd.spawn().with_context(|| format!("cannot start {}", script.display()))?;
        Ok(Process { pid: child.id() as i32, log: log.to_owned(), started: Instant::now(), exited: false })
    }

    /// A ComfyUI started by the daemon this one replaced through `reload`.
    pub fn adopt(pid: i32, log: PathBuf) -> Process {
        Process { pid, log, started: Instant::now(), exited: false }
    }

    /// Whether the process is still there. One that has ended is reaped here.
    pub fn alive(&mut self) -> bool {
        if self.exited {
            return false;
        }
        let mut status = 0;
        // SAFETY: plain libc calls on a pid this process owns.
        let r = unsafe { libc::waitpid(self.pid, &mut status, libc::WNOHANG) };
        if r == self.pid {
            self.exited = true;
            false
        } else if r == -1 {
            // not a child of ours after all: ask whether the pid exists
            unsafe { libc::kill(self.pid, 0) == 0 }
        } else {
            true
        }
    }

    pub fn signal(&self, sig: i32) {
        // SAFETY: a plain libc call.
        unsafe {
            libc::kill(self.pid, sig);
        }
    }

    /// SIGINT, then up to `grace` for it to go, then SIGKILL.
    pub fn stop(&mut self, grace: Duration) {
        if !self.alive() {
            return;
        }
        self.signal(libc::SIGINT);
        let until = Instant::now() + grace;
        while Instant::now() < until {
            if !self.alive() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        self.signal(libc::SIGKILL);
        let until = Instant::now() + Duration::from_secs(5);
        while self.alive() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// Puts the signals a shell may have left ignored back to their defaults, so a child answers SIGINT. A daemon
/// started in the background of a script has SIGINT ignored, and an ignored signal survives exec.
///
/// # Safety
/// Only for `pre_exec`: it calls `signal`, which is async-signal-safe.
pub unsafe fn default_signals() {
    for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGPIPE, libc::SIGQUIT] {
        // SAFETY: see above.
        unsafe {
            libc::signal(sig, libc::SIG_DFL);
        }
    }
}

/// A fresh log file, `queue/comfyui/<time>.log`; older logs beyond the last few are removed.
pub fn new_log(lib: &Library) -> Result<PathBuf> {
    let dir = queue::comfyui_logs_dir(lib);
    fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let stamp: String = time::now().chars().take(19).filter(|c| c.is_ascii_digit() || *c == 'T').collect();
    let path = dir.join(format!("{}.log", stamp.replace('T', "-")));
    let mut logs: Vec<PathBuf> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "log"))
        .collect();
    logs.sort();
    if logs.len() >= LOGS_KEPT {
        for old in &logs[..logs.len() + 1 - LOGS_KEPT] {
            let _ = fs::remove_file(old);
        }
    }
    Ok(path)
}

/// The last `n` lines of a file, or why it cannot be read.
pub fn tail(path: &Path, n: usize) -> String {
    match fs::read_to_string(path) {
        Ok(text) => {
            let lines: Vec<&str> = text.lines().collect();
            lines[lines.len().saturating_sub(n)..].join("\n")
        }
        Err(e) => format!("({} cannot be read: {e})", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_log_keeps_the_last_ten() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(dir.path()).unwrap();
        let logs = queue::comfyui_logs_dir(&lib);
        fs::create_dir_all(&logs).unwrap();
        for i in 0..12 {
            fs::write(logs.join(format!("20260101-{i:06}.log")), "x").unwrap();
        }
        let path = new_log(&lib).unwrap();
        assert!(path.extension().is_some_and(|e| e == "log") && path.parent() == Some(logs.as_path()));
        let left = fs::read_dir(&logs).unwrap().count();
        assert_eq!(left, LOGS_KEPT - 1, "room for the new one");
        assert!(!logs.join("20260101-000000.log").exists() && logs.join("20260101-000011.log").exists());
    }

    #[test]
    fn tail_gives_the_last_lines() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("l");
        fs::write(&p, "a\nb\nc\n").unwrap();
        assert_eq!(tail(&p, 2), "b\nc");
        assert!(tail(&dir.path().join("none"), 2).contains("cannot be read"));
    }

    #[test]
    fn the_queue_listing_is_read_from_comfyuis_arrays() {
        // the shape of GET /queue, parsed the way `queue` does
        let v = json!({"queue_running": [[1, "p1", {}, {"client_id": "c1"}, []]], "queue_pending": [[2, "p2", {}, {}, []]]});
        let mut out = Vec::new();
        for (key, running) in [("queue_running", true), ("queue_pending", false)] {
            for entry in v[key].as_array().into_iter().flatten() {
                let prompt_id = entry[1].as_str().unwrap().to_owned();
                let client_id = entry.get(3).and_then(|x| x["client_id"].as_str()).map(str::to_owned);
                out.push(Item { prompt_id, client_id, running });
            }
        }
        assert_eq!(out[0], Item { prompt_id: "p1".into(), client_id: Some("c1".into()), running: true });
        assert_eq!(out[1], Item { prompt_id: "p2".into(), client_id: None, running: false });
    }
}
