//! queue/: the jobs rope-queue runs, one directory each, and the counters that give jobs and groups their ids.
//!
//! A job is one image or one clip to make: a tool, its flags pinned, a prompt, and inputs that are in the
//! library already. The jobs of one command form a group. Ids count up forever and are never reused, so FIFO is
//! a sort by id and "#123" names the same job for good.
//!
//! rope-core holds the model, the files, the state machine and the readers that the daemon, the CLI, the GUI
//! and purge share. Only rope-queue's daemon, which holds `queue/lock` for its life, changes a job. Purge leaves
//! the images of unfinished jobs alone through `active_inputs`, as it leaves those of img2img.py runs through
//! `runs`; the daemon holds `run.lock` shared from importing a submission's inputs to writing its jobs, so purge
//! cannot slip in between.

use crate::{Library, atomic, journal, time};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

pub mod protocol;
pub mod socket;

pub const FORMAT: u32 = 1;

/// Where a job is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    /// Accepted; its inputs are in the library.
    Queued,
    /// An input is being converted for ComfyUI.
    Converting,
    /// Can go to ComfyUI.
    Ready,
    /// A child is working on it.
    Running,
    /// The child has delivered the result; the daemon is putting it into rope.
    Importing,
    /// In rope; `result` says where.
    Done,
    /// `error` says why.
    Failed,
    /// By the user.
    Cancelled,
}

impl State {
    pub const ALL: [State; 8] = [
        State::Queued,
        State::Converting,
        State::Ready,
        State::Running,
        State::Importing,
        State::Done,
        State::Failed,
        State::Cancelled,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            State::Queued => "queued",
            State::Converting => "converting",
            State::Ready => "ready",
            State::Running => "running",
            State::Importing => "importing",
            State::Done => "done",
            State::Failed => "failed",
            State::Cancelled => "cancelled",
        }
    }

    /// Done, failed or cancelled: nothing more happens to the job, and its images are not needed any more.
    pub fn is_terminal(self) -> bool {
        matches!(self, State::Done | State::Failed | State::Cancelled)
    }

    /// Whether a job may go from `self` to `to`. Staying put is always allowed. Running goes back to ready when
    /// the daemon interrupts the job, so it is first in line next time; failed goes to importing when a result
    /// that rope could not take is imported again.
    pub fn can_go(self, to: State) -> bool {
        use State::*;
        self == to
            || matches!(
                (self, to),
                (Queued, Converting | Ready | Failed | Cancelled)
                    | (Converting, Ready | Failed | Cancelled)
                    | (Ready, Running | Failed | Cancelled)
                    | (Running, Importing | Ready | Failed | Cancelled)
                    | (Importing, Done | Failed)
                    | (Failed, Importing)
            )
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for State {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<State> {
        State::ALL.into_iter().find(|st| st.as_str() == s).with_context(|| {
            let names: Vec<&str> = State::ALL.iter().map(|st| st.as_str()).collect();
            format!("no job state {s:?}; they are {}", names.join(", "))
        })
    }
}

/// The command a job came from, and the job's place in it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    pub id: u64,
    /// Counted from 1, as recipes say "3 of 50".
    pub index: u32,
    pub count: u32,
    /// The seed of the first job, so repro.sh can say "seeds 42-91".
    pub first_seed: Option<u64>,
}

/// An image a job works on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    /// The image in the library.
    pub id: String,
    /// What the result is named after: the file's stem, or the library name of an `@ID`.
    pub name: String,
    /// The converted PNG in the library, when the input needed converting for ComfyUI.
    pub rendition: Option<String>,
}

/// What the user typed.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Submitted {
    pub argv: Vec<String>,
    pub cwd: String,
    pub at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub format: u32,
    pub id: u64,
    pub group: Group,
    pub state: State,
    /// When the group was submitted. The result's name takes it, so one command's images sort together.
    pub created: String,
    pub started: Option<String>,
    pub ended: Option<String>,
    /// `img2img.py` or `img2vid.py`: a file in the repo.
    pub tool: String,
    /// Pinned: every default spelled out and the seed this job's own, as the tool's `--replay` parses them.
    pub flags: Vec<String>,
    pub prompt: String,
    pub inputs: Vec<Input>,
    /// The folder the result goes into, as the tool's `--into` takes it.
    pub into: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub note: Option<String>,
    #[serde(default)]
    pub fresh: bool,
    /// For a replay, the image replayed.
    pub replay: Option<String>,
    /// The job this one runs again.
    pub retry_of: Option<u64>,
    pub submitted: Submitted,
    /// How many times a child was started for it.
    #[serde(default)]
    pub attempts: u32,
    /// The last child's exit code.
    pub exit: Option<i32>,
    pub error: Option<String>,
    /// `{"client_id", "prompt_id"}` while a child runs, so a dead child's prompt can be cleared from ComfyUI.
    pub comfy: Option<Value>,
    /// Where the result went, once imported.
    pub result: Option<Value>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Job {
    pub fn dir(&self, lib: &Library) -> PathBuf {
        job_dir(lib, self.id)
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// The images the job needs in the library: its inputs, their renditions, and the image it replays.
    pub fn uses(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for input in &self.inputs {
            ids.push(input.id.clone());
            ids.extend(input.rendition.clone());
        }
        ids.extend(self.replay.clone());
        ids
    }

    /// Writes job.json. Callers that change a job go through `update`, which takes the library lock.
    pub fn save(&self, lib: &Library) -> Result<()> {
        let dir = self.dir(lib);
        fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        atomic::write(&dir.join("job.json"), &bytes)
    }
}

// ---------------------------------------------------------------- paths

pub fn jobs_dir(lib: &Library) -> PathBuf {
    lib.queue_dir().join("jobs")
}

/// `queue/jobs/000123`.
pub fn job_dir(lib: &Library, id: u64) -> PathBuf {
    jobs_dir(lib).join(format!("{id:06}"))
}

pub fn job_path(lib: &Library, id: u64) -> PathBuf {
    job_dir(lib, id).join("job.json")
}

/// The child's output, stdout and stderr.
pub fn log_path(lib: &Library, id: u64) -> PathBuf {
    job_dir(lib, id).join("log")
}

/// What the child made, until it is imported.
pub fn out_dir(lib: &Library, id: u64) -> PathBuf {
    job_dir(lib, id).join("out")
}

/// One ComfyUI log per start.
pub fn comfyui_logs_dir(lib: &Library) -> PathBuf {
    lib.queue_dir().join("comfyui")
}

fn counters_path(lib: &Library) -> PathBuf {
    lib.queue_dir().join("queue.json")
}

// ---------------------------------------------------------------- counters

/// queue.json: the next ids to give out. Both only grow, until `clear` leaves the queue empty: then they start from
/// one again, since nothing is left that an id could refer to.
#[derive(Debug, Serialize, Deserialize)]
struct Counters {
    format: u32,
    next_job: u64,
    next_group: u64,
    #[serde(flatten)]
    other: Map<String, Value>,
}

impl Counters {
    fn load(lib: &Library) -> Result<Counters> {
        let path = counters_path(lib);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Counters { format: FORMAT, next_job: 1, next_group: 1, other: Map::new() });
            }
            Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
        };
        // a file that does not parse is an error, never a fresh start: that would give ids out again
        let c: Counters =
            serde_json::from_slice(&bytes).with_context(|| format!("{} does not parse", path.display()))?;
        if c.format > FORMAT {
            bail!("{} has format {}; this rope reads up to {FORMAT}", path.display(), c.format);
        }
        Ok(c)
    }

    fn save(&self, lib: &Library) -> Result<()> {
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        atomic::write(&counters_path(lib), &bytes)
    }
}

// ---------------------------------------------------------------- reading

fn parse(bytes: &[u8]) -> Result<Job> {
    let v: Value = serde_json::from_slice(bytes)?;
    let format = v.get("format").and_then(Value::as_u64).unwrap_or(0);
    if format > FORMAT as u64 {
        bail!("format {format}; this rope reads up to {FORMAT}");
    }
    Ok(serde_json::from_value(v)?)
}

pub fn load(lib: &Library, id: u64) -> Result<Job> {
    let path = job_path(lib, id);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!("no job #{id}"),
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    parse(&bytes).with_context(|| format!("{} does not parse", path.display()))
}

/// Every job, by id, and the job directories whose job.json cannot be read.
pub fn scan(lib: &Library) -> Result<(Vec<Job>, Vec<String>)> {
    let (mut jobs, mut problems) = (Vec::new(), Vec::new());
    let entries = match fs::read_dir(jobs_dir(lib)) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((jobs, problems)),
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", jobs_dir(lib).display())),
    };
    for entry in entries {
        let dir = entry?.path();
        if !dir.is_dir() {
            continue;
        }
        let path = dir.join("job.json");
        match fs::read(&path).map_err(anyhow::Error::from).and_then(|b| parse(&b)) {
            Ok(job) => jobs.push(job),
            Err(e) => problems.push(format!("{}: {e:#}", path.display())),
        }
    }
    jobs.sort_by_key(|j| j.id);
    Ok((jobs, problems))
}

/// Every job that can be read, by id.
pub fn list(lib: &Library) -> Result<Vec<Job>> {
    scan(lib).map(|(jobs, _)| jobs)
}

/// Each image an unfinished job needs, with the id of the first such job: purge leaves these alone. A job.json
/// that cannot be read is an error, since it may name an image.
pub fn active_inputs(lib: &Library) -> Result<HashMap<String, u64>> {
    let (jobs, problems) = scan(lib)?;
    if !problems.is_empty() {
        bail!(
            "these jobs in the queue cannot be read, so the images they use are unknown:\n  {}",
            problems.join("\n  ")
        );
    }
    let mut out = HashMap::new();
    for job in jobs.iter().filter(|j| !j.is_terminal()) {
        for id in job.uses() {
            out.entry(id).or_insert(job.id);
        }
    }
    Ok(out)
}

/// Whether a rope-queue daemon holds `queue/lock` now.
pub fn daemon_running(lib: &Library) -> Result<bool> {
    Ok(lib.try_queue_lock()?.is_none())
}

// ---------------------------------------------------------------- writing

/// One command's worth of jobs: what every job shares, and the pinned flags of each.
#[derive(Debug, Clone, Default)]
pub struct NewGroup {
    pub tool: String,
    pub prompt: String,
    pub inputs: Vec<Input>,
    pub into: Option<String>,
    pub tags: Vec<String>,
    pub note: Option<String>,
    pub fresh: bool,
    pub replay: Option<String>,
    pub first_seed: Option<u64>,
    /// One job per entry, in this order.
    pub flags: Vec<Vec<String>>,
    /// `at` is filled in when empty.
    pub submitted: Submitted,
}

/// Gives the group an id and each of its jobs the next consecutive ids, writes them as `queued`, and journals
/// the group. The counters are written first, so ids of jobs that then fail to be written are skipped, never
/// given out again.
pub fn submit(lib: &Library, new: &NewGroup) -> Result<Vec<Job>> {
    if new.tool.trim().is_empty() {
        bail!("a job needs a tool");
    }
    if new.flags.is_empty() {
        bail!("a group needs at least one job");
    }
    let count = new.flags.len() as u64;
    let _lock = lib.lock()?;
    fs::create_dir_all(jobs_dir(lib)).with_context(|| format!("cannot create {}", jobs_dir(lib).display()))?;
    let mut counters = Counters::load(lib)?;
    let (group, first) = (counters.next_group, counters.next_job);
    if let Some(taken) = (first..first + count).find(|id| job_dir(lib, *id).exists()) {
        bail!(
            "{} exists, but {} says the next job is #{first}",
            job_dir(lib, taken).display(),
            counters_path(lib).display()
        );
    }
    counters.next_group += 1;
    counters.next_job += count;
    counters.save(lib)?;
    let now = time::now();
    let mut submitted = new.submitted.clone();
    if submitted.at.is_empty() {
        submitted.at = now.clone();
    }
    let mut jobs = Vec::with_capacity(new.flags.len());
    for (i, flags) in new.flags.iter().enumerate() {
        let job = Job {
            format: FORMAT,
            id: first + i as u64,
            group: Group { id: group, index: i as u32 + 1, count: count as u32, first_seed: new.first_seed },
            state: State::Queued,
            created: now.clone(),
            started: None,
            ended: None,
            tool: new.tool.clone(),
            flags: flags.clone(),
            prompt: new.prompt.clone(),
            inputs: new.inputs.clone(),
            into: new.into.clone(),
            tags: new.tags.clone(),
            note: new.note.clone(),
            fresh: new.fresh,
            replay: new.replay.clone(),
            retry_of: None,
            submitted: submitted.clone(),
            attempts: 0,
            exit: None,
            error: None,
            comfy: None,
            result: None,
            other: Map::new(),
        };
        job.save(lib)?;
        jobs.push(job);
    }
    let last = first + count - 1;
    journal::append_with(lib, "queue", &[], json!({"group": group, "jobs": [first, last], "state": "queued"}))?;
    Ok(jobs)
}

/// Changes the job `id` under the library lock and saves it. A change of state must be one `State::can_go`
/// allows. `started` is set on becoming running, `ended` on reaching a terminal state and cleared on leaving
/// one. Reaching a terminal state is journaled, so the GUI sees it; nothing else is, since the GUI rereads the
/// catalog on every journal line.
pub fn update<T>(lib: &Library, id: u64, change: impl FnOnce(&mut Job) -> Result<T>) -> Result<T> {
    apply(lib, id, change).map(|(_, out)| out)
}

/// `update`, answering the job as saved too.
fn apply<T>(lib: &Library, id: u64, change: impl FnOnce(&mut Job) -> Result<T>) -> Result<(Job, T)> {
    let _lock = lib.lock()?;
    let mut job = load(lib, id)?;
    let before = job.state;
    let out = change(&mut job)?;
    let after = job.state;
    if after != before {
        if !before.can_go(after) {
            bail!("job #{id} is {before} and cannot become {after}");
        }
        if after == State::Running {
            job.started = Some(time::now());
        }
        if after.is_terminal() {
            job.ended = Some(time::now());
        } else if before.is_terminal() {
            job.ended = None;
        }
    }
    job.save(lib)?;
    if after != before && after.is_terminal() {
        journal::append_with(lib, "queue", &[], json!({"job": id, "group": job.group.id, "state": after}))?;
    }
    Ok((job, out))
}

fn edit(lib: &Library, id: u64, change: impl FnOnce(&mut Job)) -> Result<Job> {
    apply(lib, id, |j| {
        change(j);
        Ok(())
    })
    .map(|(job, ())| job)
}

pub fn set_state(lib: &Library, id: u64, state: State) -> Result<Job> {
    edit(lib, id, |j| j.state = state)
}

/// A child is starting on the job.
pub fn start(lib: &Library, id: u64) -> Result<Job> {
    edit(lib, id, |j| {
        j.state = State::Running;
        j.attempts += 1;
        j.exit = None;
        j.error = None;
        j.comfy = None;
    })
}

/// The result is in rope; `result` says where.
pub fn finish(lib: &Library, id: u64, result: Value) -> Result<Job> {
    edit(lib, id, |j| {
        j.state = State::Done;
        j.result = Some(result);
    })
}

pub fn fail(lib: &Library, id: u64, error: &str, exit: Option<i32>) -> Result<Job> {
    edit(lib, id, |j| {
        j.state = State::Failed;
        j.error = Some(error.to_owned());
        j.exit = exit;
    })
}

pub fn cancel(lib: &Library, id: u64) -> Result<Job> {
    edit(lib, id, |j| j.state = State::Cancelled)
}

/// Puts an interrupted job back in line with its id, so it is first next time.
pub fn requeue(lib: &Library, id: u64) -> Result<Job> {
    edit(lib, id, |j| {
        j.state = State::Ready;
        j.started = None;
        j.comfy = None;
    })
}

/// A new job with the group, flags, inputs and prompt of the failed or cancelled job `id`. It gets the next id,
/// so it joins the line at the end, and keeps the group's `created`, so its result sorts with the group's.
pub fn retry(lib: &Library, id: u64) -> Result<Job> {
    let _lock = lib.lock()?;
    let old = load(lib, id)?;
    if !matches!(old.state, State::Failed | State::Cancelled) {
        bail!("job #{id} is {}; only a failed or cancelled job is run again", old.state);
    }
    let mut counters = Counters::load(lib)?;
    let new_id = counters.next_job;
    if job_dir(lib, new_id).exists() {
        bail!(
            "{} exists, but {} says the next job is #{new_id}",
            job_dir(lib, new_id).display(),
            counters_path(lib).display()
        );
    }
    counters.next_job += 1;
    counters.save(lib)?;
    let job = Job {
        id: new_id,
        state: State::Queued,
        started: None,
        ended: None,
        retry_of: Some(id),
        attempts: 0,
        exit: None,
        error: None,
        comfy: None,
        result: None,
        ..old
    };
    job.save(lib)?;
    let line = json!({"group": job.group.id, "jobs": [new_id, new_id], "state": "queued", "retry_of": id});
    journal::append_with(lib, "queue", &[], line)?;
    Ok(job)
}

/// Removes the finished jobs `ids` with their directories. An unfinished one is refused before anything is
/// removed.
pub fn remove(lib: &Library, ids: &[u64]) -> Result<()> {
    let _lock = lib.lock()?;
    let mut jobs = Vec::with_capacity(ids.len());
    for &id in ids {
        let job = load(lib, id)?;
        if !job.is_terminal() {
            bail!("job #{id} is {}; cancel it first", job.state);
        }
        jobs.push(job);
    }
    for job in &jobs {
        let dir = job.dir(lib);
        fs::remove_dir_all(&dir).with_context(|| format!("cannot remove {}", dir.display()))?;
    }
    journal::append_with(lib, "queue", &[], json!({"removed": ids}))?;
    Ok(())
}

/// Removes every finished job and answers their ids. A queue left empty counts from one again (user's decision,
/// 2026-10-17): the ids tell jobs apart on the screen, and once every job is gone there is nothing they could clash
/// with.
pub fn clear(lib: &Library) -> Result<Vec<u64>> {
    let ids: Vec<u64> = list(lib)?.iter().filter(|j| j.is_terminal()).map(|j| j.id).collect();
    if !ids.is_empty() {
        remove(lib, &ids)?;
    }
    let _lock = lib.lock()?;
    if list(lib)?.is_empty() {
        Counters { format: FORMAT, next_job: 1, next_group: 1, other: Map::new() }.save(lib)?;
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library() -> (tempfile::TempDir, Library) {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(dir.path()).unwrap();
        (dir, lib)
    }

    fn group(tool: &str, n: usize) -> NewGroup {
        NewGroup {
            tool: tool.to_owned(),
            prompt: "make it so".to_owned(),
            inputs: vec![Input { id: "a".repeat(64), name: "photo".to_owned(), rendition: None }],
            flags: (0..n).map(|i| vec!["--model=2511".to_owned(), format!("--seed={}", 42 + i)]).collect(),
            first_seed: Some(42),
            submitted: Submitted { argv: vec!["./img2img.py".to_owned()], cwd: "/home".to_owned(), at: String::new() },
            ..NewGroup::default()
        }
    }

    fn ids(jobs: &[Job]) -> Vec<u64> {
        jobs.iter().map(|j| j.id).collect()
    }

    fn journal_lines(lib: &Library) -> Vec<Value> {
        journal::read_from(lib, 0).unwrap().0
    }

    #[test]
    fn ids_count_up_across_groups_and_are_never_given_out_again() {
        let (_dir, lib) = library();
        let first = submit(&lib, &group("img2img.py", 3)).unwrap();
        let second = submit(&lib, &group("img2vid.py", 2)).unwrap();
        assert_eq!(ids(&first), [1, 2, 3]);
        assert_eq!(ids(&second), [4, 5]);
        assert!(first.iter().all(|j| j.group.id == 1 && j.group.count == 3 && j.created == first[0].created));
        assert_eq!((first[2].group.index, second[0].group.id, second[1].group.index), (3, 2, 2));
        assert_eq!(first[1].flags, ["--model=2511", "--seed=43"]);
        assert!(!first[0].submitted.at.is_empty(), "the time of submission is filled in");
        assert!(job_path(&lib, 5).exists() && job_dir(&lib, 5).ends_with("jobs/000005"));
        assert_eq!(load(&lib, 3).unwrap(), first[2]);
        assert!(load(&lib, 6).unwrap_err().to_string().contains("no job #6"));
        assert_eq!(ids(&list(&lib).unwrap()), [1, 2, 3, 4, 5]);

        assert!(clear(&lib).unwrap().is_empty(), "nothing is finished yet");
        for id in 1..=5 {
            cancel(&lib, id).unwrap();
        }
        assert_eq!(clear(&lib).unwrap(), [1, 2, 3, 4, 5]);
        assert!(list(&lib).unwrap().is_empty() && !job_dir(&lib, 1).exists());
        let third = submit(&lib, &group("img2img.py", 1)).unwrap();
        assert_eq!((third[0].id, third[0].group.id), (1, 1), "an emptied queue counts from one again");
        let counters = fs::read_to_string(counters_path(&lib)).unwrap();
        assert!(counters.contains(r#""next_job": 2"#) && counters.contains(r#""next_group": 2"#), "{counters}");

        // a clear that leaves a job behind keeps counting
        let fourth = submit(&lib, &group("img2img.py", 1)).unwrap();
        assert_eq!((fourth[0].id, fourth[0].group.id), (2, 2));
        cancel(&lib, 1).unwrap();
        assert_eq!(clear(&lib).unwrap(), [1]);
        assert_eq!(submit(&lib, &group("img2img.py", 1)).unwrap()[0].id, 3, "job #2 is still there");

        let lines = journal_lines(&lib);
        let queue: Vec<&Value> = lines.iter().filter(|l| l["op"] == "queue").collect();
        assert_eq!(queue.len(), 5 + 6 + 2, "five submissions, six cancellations, two clears");
        assert_eq!(queue[0]["jobs"], json!([1, 3]));
        assert!(queue.iter().all(|l| l["ids"] == json!([])), "the GUI loads every id as an image");
        assert_eq!(queue[7]["removed"], json!([1, 2, 3, 4, 5]));
        assert_eq!(queue[8]["jobs"], json!([1, 1]));
    }

    #[test]
    fn submit_refuses_an_empty_group_and_a_counter_file_that_is_behind() {
        let (_dir, lib) = library();
        assert!(submit(&lib, &group("img2img.py", 0)).is_err());
        assert!(submit(&lib, &group(" ", 1)).is_err());
        submit(&lib, &group("img2img.py", 1)).unwrap();
        fs::write(counters_path(&lib), r#"{"format": 1, "next_job": 1, "next_group": 1}"#).unwrap();
        let err = submit(&lib, &group("img2img.py", 1)).unwrap_err().to_string();
        assert!(err.contains("000001") && err.contains("next job is #1"), "{err}");
        fs::write(counters_path(&lib), "not json").unwrap();
        assert!(submit(&lib, &group("img2img.py", 1)).unwrap_err().to_string().contains("does not parse"));
        fs::write(counters_path(&lib), r#"{"format": 9, "next_job": 5, "next_group": 2}"#).unwrap();
        assert!(submit(&lib, &group("img2img.py", 1)).unwrap_err().to_string().contains("format 9"));
    }

    #[test]
    fn a_job_walks_through_its_states_and_wrong_steps_are_refused() {
        let (_dir, lib) = library();
        let jobs = submit(&lib, &group("img2img.py", 2)).unwrap();
        let (a, b) = (jobs[0].id, jobs[1].id);

        let err = start(&lib, a).unwrap_err().to_string();
        assert!(err.contains("is queued and cannot become running"), "{err}");
        set_state(&lib, a, State::Converting).unwrap();
        set_state(&lib, a, State::Ready).unwrap();
        let running = start(&lib, a).unwrap();
        assert!(running.started.is_some() && running.attempts == 1 && running.state == State::Running);
        set_state(&lib, a, State::Importing).unwrap();
        assert!(set_state(&lib, a, State::Ready).is_err(), "a delivered result is imported, not run again");
        let done = finish(&lib, a, json!({"id": "f".repeat(64)})).unwrap();
        assert!(done.ended.is_some() && done.result.is_some() && done.is_terminal());
        assert!(start(&lib, a).is_err() && cancel(&lib, a).is_err() && fail(&lib, a, "x", None).is_err());
        assert_eq!(set_state(&lib, a, State::Done).unwrap(), done, "staying put changes nothing");

        set_state(&lib, b, State::Ready).unwrap();
        start(&lib, b).unwrap();
        let back = requeue(&lib, b).unwrap();
        assert!(back.state == State::Ready && back.started.is_none());
        assert_eq!(start(&lib, b).unwrap().attempts, 2);
        let failed = fail(&lib, b, "ComfyUI said no", Some(1)).unwrap();
        assert_eq!((failed.error.as_deref(), failed.exit), (Some("ComfyUI said no"), Some(1)));
        assert!(failed.ended.is_some());
        let again = set_state(&lib, b, State::Importing).unwrap();
        assert!(again.ended.is_none(), "leaving a terminal state clears ended");
        finish(&lib, b, json!({})).unwrap();

        let lines = journal_lines(&lib);
        let states: Vec<&str> = lines.iter().filter_map(|l| l["state"].as_str()).collect();
        assert_eq!(states, ["queued", "done", "failed", "done"], "only submission and the ends are journaled");
        assert_eq!(lines.last().unwrap()["job"], json!(b));
    }

    #[test]
    fn active_inputs_names_the_images_unfinished_jobs_need() {
        let (_dir, lib) = library();
        let (a, r, p) = ("a".repeat(64), "b".repeat(64), "c".repeat(64));
        let mut g = group("img2img.py", 2);
        g.inputs[0].rendition = Some(r.clone());
        g.replay = Some(p.clone());
        submit(&lib, &g).unwrap();
        let busy = active_inputs(&lib).unwrap();
        assert_eq!((busy.get(&a), busy.get(&r), busy.get(&p)), (Some(&1), Some(&1), Some(&1)));
        cancel(&lib, 1).unwrap();
        assert_eq!(active_inputs(&lib).unwrap().get(&a), Some(&2), "the first unfinished job");
        set_state(&lib, 2, State::Ready).unwrap();
        start(&lib, 2).unwrap();
        assert_eq!(active_inputs(&lib).unwrap().len(), 3, "a running job still needs them");
        fail(&lib, 2, "x", None).unwrap();
        assert!(active_inputs(&lib).unwrap().is_empty());
    }

    #[test]
    fn scan_reports_a_job_that_does_not_parse_and_active_inputs_refuses_to_guess() {
        let (_dir, lib) = library();
        submit(&lib, &group("img2img.py", 1)).unwrap();
        fs::create_dir_all(job_dir(&lib, 9)).unwrap();
        fs::write(job_path(&lib, 9), "{").unwrap();
        fs::write(jobs_dir(&lib).join("notes"), "ignored").unwrap();
        let (jobs, problems) = scan(&lib).unwrap();
        assert_eq!((ids(&jobs), problems.len()), (vec![1], 1));
        assert!(problems[0].contains("000009/job.json"), "{problems:?}");
        assert_eq!(ids(&list(&lib).unwrap()), [1]);
        assert!(active_inputs(&lib).unwrap_err().to_string().contains("cannot be read"));
        fs::write(job_path(&lib, 9), r#"{"format": 2, "id": 9}"#).unwrap();
        assert!(scan(&lib).unwrap().1[0].contains("format 2"));
    }

    #[test]
    fn retry_makes_a_new_job_at_the_end_of_the_line() {
        let (_dir, lib) = library();
        let jobs = submit(&lib, &group("img2img.py", 2)).unwrap();
        assert!(retry(&lib, 1).unwrap_err().to_string().contains("is queued"));
        fail(&lib, 1, "x", Some(1)).unwrap();
        let new = retry(&lib, 1).unwrap();
        assert_eq!((new.id, new.retry_of, new.state, new.attempts), (3, Some(1), State::Queued, 0));
        assert!(new.error.is_none() && new.exit.is_none() && new.ended.is_none());
        assert_eq!((&new.group, &new.flags, &new.created), (&jobs[0].group, &jobs[0].flags, &jobs[0].created));
        assert_eq!(load(&lib, 1).unwrap().state, State::Failed, "the old job stays as it was");
        assert_eq!(ids(&list(&lib).unwrap()), [1, 2, 3]);
        assert_eq!(journal_lines(&lib).last().unwrap()["retry_of"], json!(1));
    }

    #[test]
    fn remove_refuses_an_unfinished_job_and_removes_nothing_then() {
        let (_dir, lib) = library();
        submit(&lib, &group("img2img.py", 2)).unwrap();
        cancel(&lib, 1).unwrap();
        let err = remove(&lib, &[1, 2]).unwrap_err().to_string();
        assert!(err.contains("job #2 is queued"), "{err}");
        assert!(job_dir(&lib, 1).exists(), "nothing was removed");
        remove(&lib, &[1]).unwrap();
        assert!(!job_dir(&lib, 1).exists() && job_dir(&lib, 2).exists());
        assert!(remove(&lib, &[1]).unwrap_err().to_string().contains("no job #1"));
    }

    #[test]
    fn the_queue_lock_says_whether_a_daemon_runs() {
        let (_dir, lib) = library();
        assert!(!daemon_running(&lib).unwrap());
        let held = lib.try_queue_lock().unwrap().expect("free");
        assert!(daemon_running(&lib).unwrap());
        assert!(lib.try_queue_lock().unwrap().is_none());
        drop(held);
        assert!(!daemon_running(&lib).unwrap());
    }

    #[test]
    fn states_are_named_in_lowercase_both_ways() {
        for s in State::ALL {
            assert_eq!(serde_json::to_value(s).unwrap(), json!(s.as_str()));
            assert_eq!(s.as_str().parse::<State>().unwrap(), s);
            assert!(s.can_go(s));
        }
        assert!("Done".parse::<State>().is_err());
        assert!(State::Running.can_go(State::Ready) && !State::Done.can_go(State::Ready));
        assert!(State::Failed.can_go(State::Importing) && !State::Cancelled.can_go(State::Importing));
    }
}
