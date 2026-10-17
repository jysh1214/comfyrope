//! The client side of rope-queue's Unix socket: one JSON line in, one line out, more lines for a subscription.
//! `$XDG_RUNTIME_DIR/rope-queue.sock`, so a reboot leaves no stale socket behind. The daemon's listening side is in
//! the rope-queue crate; this half is here so that rope-gui can talk to the daemon.

use super::protocol::Request;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

pub fn path() -> PathBuf {
    match std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty()) {
        Some(dir) => PathBuf::from(dir).join("rope-queue.sock"),
        None => std::env::temp_dir().join(format!("rope-queue-{}.sock", uid())),
    }
}

/// This process's user id, from /proc, for a socket name of its own when there is no runtime directory.
fn uid() -> u32 {
    use std::os::unix::fs::MetadataExt;

    std::fs::metadata("/proc/self").map(|m| m.uid()).unwrap_or(0)
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

/// Whether a daemon answers on the socket.
pub fn running() -> bool {
    UnixStream::connect(path()).is_ok()
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
