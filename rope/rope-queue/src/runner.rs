//! The child that runs one job: the Python tool from the repo's venv with `--job DIR`, speaking events on stdout.
//! Its environment is built, not inherited, so the shell that started the daemon leaks nothing into a job.

use crate::daemon::{ChildMsg, Msg};
use anyhow::{Context, Result, bail};
use rope_core::queue::protocol::ChildEvent;
use rope_core::queue::{self, Job};
use rope_core::{Library, time};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

/// What every child gets.
#[derive(Debug, Clone)]
pub struct Tools {
    pub repo: PathBuf,
    /// `<repo>/.venv/bin/python`.
    pub python: PathBuf,
    /// `host:port`.
    pub server: String,
}

impl Tools {
    pub fn new(repo: &Path, server: &str) -> Tools {
        Tools { repo: repo.to_owned(), python: repo.join(".venv/bin/python"), server: server.to_owned() }
    }

    /// Checks that the venv is there and can import what the tools need; says to run setup_venv.sh otherwise.
    pub fn check(&self) -> Result<()> {
        if !self.python.exists() {
            bail!("{} does not exist; run ./setup_venv.sh in {}", self.python.display(), self.repo.display());
        }
        let out = Command::new(&self.python)
            .args(["-E", "-c", "import websocket, PIL, pillow_heif, pillow_jxl, rawpy"])
            .env_clear()
            .envs(self.env(None, None))
            .stdin(Stdio::null())
            .output()
            .with_context(|| format!("cannot run {}", self.python.display()))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            bail!(
                "the venv at {} lacks what the tools need; run ./setup_venv.sh\n{}",
                self.python.display(),
                err.lines().last().unwrap_or("")
            );
        }
        Ok(())
    }

    /// The child's environment: the few variables from ours that a program needs, the library, and ComfyUI.
    /// No VIRTUAL_ENV, no PYTHONPATH; `-E` drops any PYTHON* that got through.
    pub fn env(&self, lib: Option<&Library>, client_id: Option<&str>) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = std::env::vars()
            .filter(|(k, _)| {
                matches!(k.as_str(), "PATH" | "HOME" | "LANG" | "TZ" | "TMPDIR" | "TERM")
                    || k.starts_with("LC_")
                    || k.starts_with("XDG_")
            })
            .collect();
        if let Some(lib) = lib {
            env.push(("ROPE_LIBRARY".to_owned(), lib.root().display().to_string()));
        }
        env.push(("COMFYUI_SERVER".to_owned(), self.server.clone()));
        if let Some(id) = client_id {
            env.push(("COMFYUI_CLIENT_ID".to_owned(), id.to_owned()));
        }
        env
    }

    /// The tool `name` as a file in the repo: a plain file name, present.
    pub fn tool(&self, name: &str) -> Result<PathBuf> {
        if name.is_empty() || name.contains('/') || name.starts_with('.') {
            bail!("{name:?} is not a tool: give the name of a script in the repo, such as img2img.py");
        }
        let path = self.repo.join(name);
        if !path.is_file() {
            bail!("there is no {name} in {}", self.repo.display());
        }
        Ok(path)
    }
}

/// Starts the child for `job` and the threads that follow it; what they see arrives as `Msg::Child`, the exit
/// last, after every line. Answers the child's pid.
pub fn spawn(tools: &Tools, lib: &Library, job: &Job, client_id: &str, tx: Sender<Msg>) -> Result<i32> {
    let tool = tools.tool(&job.tool)?;
    let dir = queue::job_dir(lib, job.id);
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(queue::log_path(lib, job.id))
        .with_context(|| format!("cannot open the log of job #{}", job.id))?;
    writeln!(log, "=== attempt {} at {} ===", job.attempts, time::now())?;
    let mut cmd = Command::new(&tools.python);
    cmd.args(["-E", "-u"])
        .arg(&tool)
        .arg("--job")
        .arg(&dir)
        .current_dir(&tools.repo)
        .env_clear()
        .envs(tools.env(Some(lib), Some(client_id)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // SAFETY: only async-signal-safe calls, see default_signals.
    unsafe {
        cmd.pre_exec(|| {
            crate::comfy::default_signals();
            Ok(())
        });
    }
    let mut child = cmd.spawn().with_context(|| format!("cannot start {}", tool.display()))?;
    let pid = child.id() as i32;
    let stdout = child.stdout.take().context("no stdout")?;
    let stderr = child.stderr.take().context("no stderr")?;
    let log = Arc::new(Mutex::new(log));
    let id = job.id;

    let (out_tx, out_log) = (tx.clone(), log.clone());
    let out = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            append(&out_log, &line);
            let what = match serde_json::from_str::<ChildEvent>(&line) {
                Ok(ev) => ChildMsg::Event(ev),
                Err(_) => ChildMsg::Garbage(line),
            };
            if out_tx.send(Msg::Child { job: id, what }).is_err() {
                break;
            }
        }
    });
    let (err_tx, err_log) = (tx.clone(), log);
    let err = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            append(&err_log, &line);
            if err_tx.send(Msg::Child { job: id, what: ChildMsg::Line(line) }).is_err() {
                break;
            }
        }
    });
    std::thread::spawn(move || {
        let status = child.wait();
        let _ = out.join();
        let _ = err.join();
        if let Ok(status) = status {
            let _ = tx.send(Msg::Child { job: id, what: ChildMsg::Exited(status) });
        }
    });
    Ok(pid)
}

fn append(log: &Mutex<File>, line: &str) {
    if let Ok(mut f) = log.lock() {
        let _ = writeln!(f, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_environment_is_built_not_inherited() {
        let tools = Tools::new(Path::new("/r"), "127.0.0.1:8188");
        let env = tools.env(None, Some("c1"));
        let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"COMFYUI_SERVER") && keys.contains(&"COMFYUI_CLIENT_ID") && keys.contains(&"PATH"));
        assert!(!keys.iter().any(|k| k.starts_with("PYTHON") || *k == "VIRTUAL_ENV" || *k == "ROPE_LIBRARY"));
        assert_eq!(tools.python, PathBuf::from("/r/.venv/bin/python"));
    }

    #[test]
    fn a_tool_is_a_plain_file_name_in_the_repo() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("img2img.py"), "").unwrap();
        let tools = Tools::new(dir.path(), "127.0.0.1:8188");
        assert_eq!(tools.tool("img2img.py").unwrap(), dir.path().join("img2img.py"));
        for bad in ["", "../x.py", "a/b.py", ".hidden", "img2vid.py"] {
            assert!(tools.tool(bad).is_err(), "{bad}");
        }
    }
}
