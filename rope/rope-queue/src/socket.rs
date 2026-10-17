//! The Unix socket the daemon listens on and the clients call: one JSON line in, one line out, more lines for a
//! subscription. `$XDG_RUNTIME_DIR/rope-queue.sock`, so a reboot leaves no stale socket behind.

use crate::daemon::Msg;
use crate::protocol::{self, Request};
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

pub fn path() -> PathBuf {
    match std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty()) {
        Some(dir) => PathBuf::from(dir).join("rope-queue.sock"),
        None => std::env::temp_dir().join(format!("rope-queue-{}.sock", unsafe { libc::getuid() })),
    }
}

/// One client's connection, written JSON lines. Kept by the daemon for a subscriber.
#[derive(Debug)]
pub struct Responder {
    stream: UnixStream,
}

impl Responder {
    pub fn send(&mut self, v: &Value) -> std::io::Result<()> {
        let mut line = v.to_string();
        line.push('\n');
        self.stream.write_all(line.as_bytes())
    }
}

/// Binds the socket. One left behind by a daemon that is gone (nothing answers on it) is replaced.
pub fn listen() -> Result<UnixListener> {
    listen_at(&path())
}

pub fn listen_at(path: &Path) -> Result<UnixListener> {
    if path.exists() {
        if UnixStream::connect(path).is_ok() {
            bail!("a rope-queue is listening on {} already", path.display());
        }
        std::fs::remove_file(path).with_context(|| format!("cannot remove the stale {}", path.display()))?;
    }
    let listener = UnixListener::bind(path).with_context(|| format!("cannot listen on {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

/// Accepts connections for good, handing each request to the daemon's loop. Runs on a thread of its own; a
/// request that is not one is answered here.
pub fn serve(listener: UnixListener, tx: Sender<Msg>) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let tx = tx.clone();
        std::thread::spawn(move || {
            let Ok(reading) = stream.try_clone() else { return };
            let mut reader = BufReader::new(reading);
            let mut line = String::new();
            let mut responder = Responder { stream };
            if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                return;
            }
            match serde_json::from_str::<Request>(&line) {
                Ok(req) => {
                    let _ = tx.send(Msg::Request(req, responder));
                }
                Err(e) => {
                    let _ = responder.send(&protocol::error(&format!("not a request: {e}")));
                }
            }
        });
    }
}

/// Sends one request to the daemon and reads its answer. An answer with `ok: false` is an error.
pub fn call(req: &Request) -> Result<Value> {
    call_at(&path(), req)
}

pub fn call_at(path: &Path, req: &Request) -> Result<Value> {
    let mut stream = connect(path)?;
    send(&mut stream, req)?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).context("rope-queue hung up without answering")?;
    if line.is_empty() {
        bail!("rope-queue hung up without answering");
    }
    answer(&line)
}

/// Sends one request and hands every line the daemon then writes to `each`, the answer first, until the daemon
/// hangs up or `each` answers false.
pub fn stream(req: &Request, mut each: impl FnMut(Value) -> Result<bool>) -> Result<()> {
    let mut stream = connect(&path())?;
    send(&mut stream, req)?;
    for line in BufReader::new(stream).lines() {
        if !each(answer(&line?)?)? {
            break;
        }
    }
    Ok(())
}

fn connect(path: &Path) -> Result<UnixStream> {
    UnixStream::connect(path)
        .map_err(|_| anyhow::anyhow!("rope-queue is not running; start it first (it listens on {})", path.display()))
}

fn send(stream: &mut UnixStream, req: &Request) -> Result<()> {
    let mut line = serde_json::to_string(req)?;
    line.push('\n');
    stream.write_all(line.as_bytes()).context("cannot write to rope-queue")
}

fn answer(line: &str) -> Result<Value> {
    let v: Value = serde_json::from_str(line).context("rope-queue answered something that is not JSON")?;
    if v.get("ok") == Some(&Value::Bool(false)) {
        bail!("{}", v["error"].as_str().unwrap_or("rope-queue refused"));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn a_request_reaches_the_loop_and_its_answer_the_client() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("s.sock");
        let (tx, rx) = channel();
        let listener = listen_at(&sock).unwrap();
        std::thread::spawn(move || serve(listener, tx));
        let sock2 = sock.clone();
        let client = std::thread::spawn(move || call_at(&sock2, &Request::Pause));
        let Msg::Request(req, mut responder) = rx.recv().unwrap() else { panic!("not a request") };
        assert_eq!(req, Request::Pause);
        responder.send(&protocol::ok(serde_json::json!({"paused": true}))).unwrap();
        assert_eq!(client.join().unwrap().unwrap()["paused"], true);

        let sock3 = sock.clone();
        let client = std::thread::spawn(move || call_at(&sock3, &Request::Stop));
        let Msg::Request(_, mut responder) = rx.recv().unwrap() else { panic!("not a request") };
        responder.send(&protocol::error("not now")).unwrap();
        assert_eq!(client.join().unwrap().unwrap_err().to_string(), "not now");

        let err = call_at(&dir.path().join("none.sock"), &Request::Ping).unwrap_err().to_string();
        assert!(err.contains("not running"), "{err}");
        assert!(listen_at(&sock).unwrap_err().to_string().contains("already"));
    }
}
