//! rope-queue: the work queue for ComfyUI jobs. Started alone it is the daemon: it starts ComfyUI, takes jobs
//! from img2img.py and img2vid.py, runs them in order and puts every result into rope. With a subcommand it is a
//! client of the running daemon.

mod comfy;
mod convert;
mod daemon;
mod import;
mod runner;
mod screen;
mod socket;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rope_core::Library;
use rope_core::hash::short;
use rope_core::queue::protocol::{Request, Submission};
use rope_core::queue::{self, Job, State};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "rope-queue",
    version,
    about = "A work queue for ComfyUI jobs: starts ComfyUI, runs jobs in order, keeps results in rope"
)]
struct Cli {
    /// Answer in JSON
    #[arg(long, global = true)]
    json: bool,
    /// The rope library (default: $ROPE_LIBRARY, else ~/.local/share/rope)
    #[arg(long, global = true, value_name = "DIR")]
    library: Option<PathBuf>,
    /// The comfyrope repo (default: where img2img.py last recorded it)
    #[arg(long, value_name = "DIR")]
    repo: Option<PathBuf>,
    /// Where ComfyUI listens
    #[arg(long, default_value = "127.0.0.1:8188", value_name = "HOST:PORT")]
    server: String,
    /// GB of VRAM left to the desktop; 0 for ComfyUI's default
    #[arg(long, default_value = "2", value_name = "GB")]
    reserve_vram: String,
    /// Minutes a job may stay silent before it is killed
    #[arg(long, default_value = "30", value_name = "MIN")]
    job_timeout: u64,
    /// The input converter (default: comfyconvert.py in the repo)
    #[arg(long, value_name = "FILE")]
    converter: Option<PathBuf>,
    /// Print one line per change instead of drawing the screen
    #[arg(long)]
    no_screen: bool,
    /// Passed on to ComfyUI, after --
    #[arg(last = true, value_name = "COMFY_ARGS")]
    comfy_args: Vec<String>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Submits the requests read from stdin (the JSON img2img.py and img2vid.py write)
    Add {
        /// Stay until every job of the submission is finished, and report each
        #[arg(long)]
        wait: bool,
    },
    /// Says whether a daemon runs, with its library and its ComfyUI
    Status,
    /// Lists the jobs (from the files when no daemon runs)
    Ls,
    /// Prints a job's log
    Log {
        id: u64,
    },
    /// Cancels jobs (123) or groups (g40); the running one is interrupted
    Cancel {
        targets: Vec<String>,
    },
    /// Runs failed or cancelled jobs (123) or groups (g40) again as new jobs
    Retry {
        targets: Vec<String>,
    },
    /// Finishes the running job and starts none
    Pause,
    Resume,
    /// Interrupts the running job (it goes back in line) and stops the daemon with ComfyUI
    Stop,
    /// Restarts the daemon binary after the running job, keeping ComfyUI
    Reload,
    /// Removes finished jobs and their files (also when no daemon runs)
    Clear,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("rope-queue: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    // the handover from the daemon this one replaces, read before anything else runs
    let adopt = std::env::var("ROPE_QUEUE_ADOPT").ok().and_then(|s| {
        let (pid, log) = s.split_once(':')?;
        Some((pid.parse().ok()?, PathBuf::from(log)))
    });
    // SAFETY: the process is single-threaded here, at the very start.
    unsafe { std::env::remove_var("ROPE_QUEUE_ADOPT") };
    let cli = Cli::parse();
    let lib = Library::locate(cli.library.as_deref())?;
    let json = cli.json;
    match cli.cmd {
        None => start_daemon(cli, lib, adopt),
        Some(Cmd::Add { wait }) => add(json, wait),
        Some(Cmd::Status) => {
            let v = socket::call(&Request::Ping)?;
            emit(json, &v, || {
                let comfy = &v["comfy"];
                let version = comfy["version"].as_str().map(|s| format!(" {s}")).unwrap_or_default();
                println!(
                    "rope-queue is running (pid {}), library {}; ComfyUI{version} {}{}",
                    v["pid"],
                    v["library"].as_str().unwrap_or("?"),
                    if comfy["up"] == true { "up" } else { "starting" },
                    if v["paused"] == true { "; paused" } else { "" }
                );
            })
        }
        Some(Cmd::Ls) => ls(&lib, json),
        Some(Cmd::Log { id }) => {
            let path = queue::log_path(&lib, id);
            let text = std::fs::read_to_string(&path).with_context(|| format!("no log for job #{id}"))?;
            print!("{text}");
            Ok(ExitCode::SUCCESS)
        }
        Some(Cmd::Cancel { targets }) => {
            let (jobs, groups) = parse_targets(&targets)?;
            let v = socket::call(&Request::Cancel { jobs, groups })?;
            emit(json, &v, || {
                let n = v["cancelled"].as_array().map_or(0, Vec::len);
                println!("cancelled {n} job(s){}", skipped(&v));
            })
        }
        Some(Cmd::Retry { targets }) => {
            let (jobs, groups) = parse_targets(&targets)?;
            let v = socket::call(&Request::Retry { jobs, groups })?;
            emit(json, &v, || {
                for r in v["retried"].as_array().into_iter().flatten() {
                    match r["as"].as_u64() {
                        Some(new) => println!("#{} runs again as #{new}", r["job"]),
                        None => println!(
                            "#{} imported again: {}",
                            r["job"],
                            if r["imported"] == true { "done" } else { "failed again" }
                        ),
                    }
                }
                let s = skipped(&v);
                if !s.is_empty() {
                    println!("{}", s.trim_start_matches("; "));
                }
            })
        }
        Some(Cmd::Pause) => {
            let v = socket::call(&Request::Pause)?;
            emit(json, &v, || match v["finishing"].as_u64() {
                Some(id) => println!("paused; #{id} finishes first"),
                None => println!("paused"),
            })
        }
        Some(Cmd::Resume) => {
            let v = socket::call(&Request::Resume)?;
            emit(json, &v, || println!("resumed"))
        }
        Some(Cmd::Stop) => {
            let v = socket::call(&Request::Stop)?;
            emit(json, &v, || match v["interrupting"].as_u64() {
                Some(id) => println!("stopping; #{id} is interrupted and goes back in line"),
                None => println!("stopping"),
            })
        }
        Some(Cmd::Reload) => {
            let v = socket::call(&Request::Reload)?;
            emit(json, &v, || match v["after"].as_u64() {
                Some(id) => println!("reloading after #{id}"),
                None => println!("reloading"),
            })
        }
        Some(Cmd::Clear) => clear(&lib, json),
    }
}

fn start_daemon(cli: Cli, lib: Library, adopt: Option<(i32, PathBuf)>) -> Result<ExitCode> {
    let repo = match cli.repo {
        Some(r) => std::path::absolute(r)?,
        None => lib
            .repo()
            .context("the library does not say where the comfyrope repo is; pass --repo DIR (a tool run records it)")?,
    };
    let port: u16 = cli
        .server
        .rsplit(':')
        .next()
        .and_then(|p| p.parse().ok())
        .with_context(|| format!("--server {}: give HOST:PORT", cli.server))?;
    let reserve_vram = (cli.reserve_vram != "0").then_some(cli.reserve_vram);
    daemon::run(daemon::Options {
        lib,
        converter: cli.converter.unwrap_or_else(|| repo.join("comfyconvert.py")),
        repo,
        server: cli.server,
        port,
        reserve_vram,
        comfy_args: cli.comfy_args,
        job_timeout: Duration::from_secs(cli.job_timeout.max(1) * 60),
        screen: !cli.no_screen,
        adopt,
    })?;
    Ok(ExitCode::SUCCESS)
}

/// `rope-queue add`: the submission on stdin, the groups and job ids out. With `--wait`, stays subscribed until
/// every job of the submission has finished. Exits 1 when a request was skipped or a job failed.
fn add(json: bool, wait: bool) -> Result<ExitCode> {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text)?;
    let sub: Submission = serde_json::from_str(&text).context("stdin is not a submission")?;
    let mut answer = socket::call(&Request::Add(sub))?;
    let mut bad = false;
    let mut groups = Vec::new();
    for g in answer["groups"].as_array().into_iter().flatten() {
        match (g["skipped"].as_str(), g["group"].as_u64()) {
            (Some(why), _) => {
                bad = true;
                if !json {
                    eprintln!("skipped: {why}");
                }
            }
            (None, Some(id)) => {
                groups.push(id);
                if !json {
                    let (first, last) = (g["jobs"][0].as_u64().unwrap_or(0), g["jobs"][1].as_u64().unwrap_or(0));
                    if first == last {
                        println!("queued #{first} (group {id})");
                    } else {
                        println!("queued #{first}–#{last} (group {id})");
                    }
                }
            }
            _ => {}
        }
    }
    if wait && !groups.is_empty() {
        let jobs = wait_for(&groups)?;
        for j in jobs.values() {
            match j.state {
                State::Done => {}
                State::Failed | State::Cancelled => bad = true,
                _ => {
                    bad = true;
                    if !json {
                        eprintln!("#{}: rope-queue stopped while it was {}", j.id, j.state);
                    }
                }
            }
            if !json {
                let path = j.result.as_ref().and_then(|r| r["path"].as_str()).unwrap_or("");
                println!("#{} {}  {}  {path}", j.id, j.state.as_str(), outcome(j));
            }
        }
        answer["jobs"] = json!(jobs.values().collect::<Vec<_>>());
    }
    if json {
        println!("{answer}");
    }
    Ok(if bad { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

/// Follows the jobs of `groups` until every one has finished, or the daemon goes away.
fn wait_for(groups: &[u64]) -> Result<BTreeMap<u64, Job>> {
    let mut jobs: BTreeMap<u64, Job> = BTreeMap::new();
    socket::stream(&Request::Subscribe { groups: groups.to_vec() }, |v| {
        match v["event"].as_str() {
            Some("snapshot") => {
                for j in v["jobs"].as_array().into_iter().flatten() {
                    if let Ok(j) = serde_json::from_value::<Job>(j.clone()) {
                        jobs.insert(j.id, j);
                    }
                }
            }
            Some("state") => {
                if let Ok(j) = serde_json::from_value::<Job>(v["job"].clone()) {
                    jobs.insert(j.id, j);
                }
            }
            Some("stopping") => return Ok(false),
            _ => {}
        }
        Ok(!(v["event"].is_string() && !jobs.is_empty() && jobs.values().all(Job::is_terminal)))
    })?;
    Ok(jobs)
}

fn outcome(j: &Job) -> String {
    match j.state {
        State::Done => {
            let r = j.result.as_ref();
            let secs = r.and_then(|r| r["seconds"].as_f64()).map(|s| format!("  {s:.1}s")).unwrap_or_default();
            match r.and_then(|r| r["result"].as_str()) {
                Some(verdict) => format!("{verdict}{secs}"),
                None => format!(
                    "{}  {}{secs}",
                    r.and_then(|r| r["short"].as_str()).unwrap_or("?"),
                    r.and_then(|r| r["folder"].as_str()).unwrap_or("/")
                ),
            }
        }
        State::Failed => j.error.clone().unwrap_or_default(),
        _ => String::new(),
    }
}

fn ls(lib: &Library, json: bool) -> Result<ExitCode> {
    let (jobs, live): (Vec<Job>, Option<Value>) = match socket::call(&Request::Ls) {
        Ok(v) => (serde_json::from_value(v["jobs"].clone())?, Some(v)),
        Err(_) => (queue::list(lib)?, None),
    };
    if json {
        println!("{}", json!({"jobs": jobs, "daemon": live}));
        return Ok(ExitCode::SUCCESS);
    }
    if live.is_none() {
        println!("(rope-queue is not running; this is what its files say)");
    } else if let Some(v) = &live
        && v["paused"] == true
    {
        println!("paused: {}", v["reason"].as_str().unwrap_or("by request"));
    }
    let running = live.as_ref().and_then(|v| v["running"].as_object().cloned());
    for j in &jobs {
        let what = match j.state {
            State::Running => match &running {
                Some(r) if r["job"].as_u64() == Some(j.id) => format!(
                    "{} {}  {}s",
                    r["stage"].as_str().unwrap_or(""),
                    match (r["progress"][0].as_u64(), r["progress"][1].as_u64()) {
                        (Some(v), Some(m)) => format!("{v}/{m}"),
                        _ => String::new(),
                    },
                    r["seconds"].as_u64().unwrap_or(0)
                ),
                _ => String::new(),
            },
            State::Done | State::Failed => outcome(j),
            _ => j.inputs.iter().map(|i| format!("{} ({})", i.name, short(&i.id))).collect::<Vec<_>>().join(", "),
        };
        println!(
            "#{:<5} g{:<4} {:<10} {:<10} {:<24} {}",
            j.id,
            j.group.id,
            j.state.as_str(),
            j.tool.trim_end_matches(".py"),
            j.into.as_deref().unwrap_or(""),
            what
        );
    }
    if jobs.is_empty() {
        println!("no jobs");
    }
    Ok(ExitCode::SUCCESS)
}

fn clear(lib: &Library, json: bool) -> Result<ExitCode> {
    let removed: Vec<u64> = match socket::call(&Request::Clear) {
        Ok(v) => serde_json::from_value(v["removed"].clone())?,
        Err(_) => {
            // no daemon: the lock says so, and keeps one from starting halfway through
            let Some(_lock) = lib.try_queue_lock()? else { bail!("rope-queue is running but does not answer") };
            queue::clear(lib)?
        }
    };
    emit(json, &json!({"removed": removed}), || println!("removed {} finished job(s)", removed.len()))
}

/// `123` or `#123` is a job; `g40` is group 40.
fn parse_targets(targets: &[String]) -> Result<(Vec<u64>, Vec<u64>)> {
    if targets.is_empty() {
        bail!("name the jobs (123) or groups (g40)");
    }
    let (mut jobs, mut groups) = (Vec::new(), Vec::new());
    for t in targets {
        let t = t.trim();
        if let Some(g) = t.strip_prefix("group").or_else(|| t.strip_prefix('g')) {
            groups.push(g.trim_start_matches([':', ' ']).parse().with_context(|| format!("{t}: not a group"))?);
        } else {
            jobs.push(t.trim_start_matches('#').parse().with_context(|| format!("{t}: not a job id"))?);
        }
    }
    Ok((jobs, groups))
}

fn skipped(v: &Value) -> String {
    let list: Vec<String> = v["skipped"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| format!("#{} {}", s["job"], s["why"].as_str().unwrap_or("")))
        .collect();
    let unknown: Vec<String> = v["unknown"].as_array().into_iter().flatten().map(|u| format!("#{u}")).collect();
    let mut out = String::new();
    if !list.is_empty() {
        out += &format!("; skipped {}", list.join(", "));
    }
    if !unknown.is_empty() {
        out += &format!("; unknown {}", unknown.join(", "));
    }
    out
}

fn emit(json: bool, v: &Value, text: impl FnOnce()) -> Result<ExitCode> {
    if json {
        println!("{v}");
    } else {
        text();
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_are_jobs_or_groups() {
        let (j, g) = parse_targets(&["12".into(), "#13".into(), "g40".into(), "group:41".into()]).unwrap();
        assert_eq!((j, g), (vec![12, 13], vec![40, 41]));
        assert!(parse_targets(&[]).is_err() && parse_targets(&["x".into()]).is_err());
    }
}
