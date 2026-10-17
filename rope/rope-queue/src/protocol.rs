//! What goes over the socket, one JSON object per line: a request, its answer, and for a subscription the events
//! that follow. The Python tools and rope-gui speak this; the client subcommands wrap it. Also the events the
//! child that runs a job writes on its stdout.

use rope_core::queue::Job;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    /// Answers the daemon's pid, library and ComfyUI.
    Ping,
    Add(Submission),
    Ls,
    Cancel {
        #[serde(default)]
        jobs: Vec<u64>,
        #[serde(default)]
        groups: Vec<u64>,
    },
    Retry {
        #[serde(default)]
        jobs: Vec<u64>,
        #[serde(default)]
        groups: Vec<u64>,
    },
    Pause,
    Resume,
    Stop,
    Reload,
    Clear,
    /// Streams the events of these groups (of every job with none) until the client hangs up. The answer is
    /// followed by a `snapshot` event, then by `state`, `stage` and `progress` events as they happen.
    Subscribe {
        #[serde(default)]
        groups: Vec<u64>,
    },
}

/// One command's worth of requests, as the Python tool sends it on `rope-queue add`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Submission {
    /// `img2img.py` or `img2vid.py`: a file in the repo.
    pub tool: String,
    pub requests: Vec<SubmitRequest>,
    pub submitted: Origin,
}

/// What the user typed, and where.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Origin {
    #[serde(default)]
    pub argv: Vec<String>,
    #[serde(default)]
    pub cwd: String,
}

/// One request: the jobs it expands to and what they share. Everything a job needs that the tool knows is
/// pinned here; the daemon interprets none of it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SubmitRequest {
    /// One job per entry: its pinned flags, the seed among them.
    pub flags: Vec<Vec<String>>,
    #[serde(default)]
    pub prompt: String,
    /// Files, relative to `cwd` unless absolute, or `@ID`s of images in rope.
    #[serde(default)]
    pub images: Vec<String>,
    pub into: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub note: Option<String>,
    #[serde(default)]
    pub fresh: bool,
    /// For a replay: the image replayed, `@ID` or a file.
    pub replay: Option<String>,
    pub first_seed: Option<u64>,
}

/// What a subscriber receives after its answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The jobs of the subscribed groups as the subscription starts.
    Snapshot {
        jobs: Vec<Job>,
        paused: bool,
    },
    /// A job changed; the whole job.
    State {
        job: Box<Job>,
    },
    Stage {
        job: u64,
        text: String,
    },
    Progress {
        job: u64,
        value: u32,
        max: u32,
    },
    Paused {
        paused: bool,
        reason: Option<String>,
    },
    Stopping,
}

/// One line of the child's stdout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ChildEvent {
    Stage {
        text: String,
    },
    Progress {
        value: u32,
        max: u32,
    },
    /// What the child queued, so the daemon can clear it if the child dies.
    Comfy {
        client_id: String,
        prompt_id: String,
    },
    Result(ResultEvent),
    Error {
        message: String,
    },
    End {
        seconds: f64,
    },
}

/// What the child made: paths relative to the job's directory unless absolute. The child also writes this object
/// to `out/result.json`, so a result delivered to a daemon that then crashed is imported on the next start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultEvent {
    pub file: PathBuf,
    /// The recipe, made.json.
    pub made: PathBuf,
    /// The API workflow, graph.json.
    pub graph: Option<PathBuf>,
    /// The result's name in rope.
    pub name: String,
    #[serde(default)]
    pub kind: Kind,
    /// For a replay, the image replayed; the job's `replay` when missing.
    pub replay_of: Option<String>,
    /// The enhanced prompt, when the workflow has one.
    pub text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A new image or clip, `rope add`.
    #[default]
    Add,
    /// The result of a replay, compared with the original, `rope replayed`.
    Replayed,
}

pub fn ok(mut v: Value) -> Value {
    if let Value::Object(m) = &mut v {
        m.insert("ok".to_owned(), Value::Bool(true));
        return v;
    }
    json!({"ok": true, "value": v})
}

pub fn error(text: &str) -> Value {
    json!({"ok": false, "error": text})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_submission_reads_as_the_design_writes_it() {
        let text = r#"{"cmd": "add", "tool": "img2img.py",
            "requests": [{"flags": [["--model=2511", "--seed=42"], ["--model=2511", "--seed=43"]], "prompt": "p",
                          "images": ["photo.jpg", "@9f4ad8"], "into": "a/b", "first_seed": 42}],
            "submitted": {"argv": ["./img2img.py", "photo.jpg"], "cwd": "/home/alex"}}"#;
        let Request::Add(sub) = serde_json::from_str::<Request>(text).unwrap() else { panic!("not add") };
        assert_eq!(
            (sub.tool.as_str(), sub.requests.len(), sub.submitted.cwd.as_str()),
            ("img2img.py", 1, "/home/alex")
        );
        let r = &sub.requests[0];
        assert_eq!((r.flags.len(), r.images.len(), r.into.as_deref(), r.first_seed), (2, 2, Some("a/b"), Some(42)));
        assert!(r.tags.is_empty() && r.note.is_none() && !r.fresh && r.replay.is_none());
        let back: Request = serde_json::from_str(&serde_json::to_string(&Request::Add(sub.clone())).unwrap()).unwrap();
        assert_eq!(back, Request::Add(sub));
        assert_eq!(serde_json::to_value(Request::Cancel { jobs: vec![3], groups: vec![] }).unwrap()["cmd"], "cancel");
        assert_eq!(
            serde_json::from_str::<Request>(r#"{"cmd": "subscribe"}"#).unwrap(),
            Request::Subscribe { groups: vec![] }
        );
    }

    #[test]
    fn child_events_parse_from_their_lines() {
        let ev: ChildEvent = serde_json::from_str(r#"{"event": "progress", "value": 3, "max": 4}"#).unwrap();
        assert_eq!(ev, ChildEvent::Progress { value: 3, max: 4 });
        let ev: ChildEvent =
            serde_json::from_str(r#"{"event": "result", "file": "out/x.png", "made": "out/made.json", "name": "n"}"#)
                .unwrap();
        let ChildEvent::Result(r) = ev else { panic!("not a result") };
        assert_eq!((r.kind, r.graph, r.replay_of), (Kind::Add, None, None));
        assert!(serde_json::from_str::<ChildEvent>(r#"{"event": "dance"}"#).is_err());
        assert!(serde_json::from_str::<ChildEvent>("hello").is_err());
    }

    #[test]
    fn answers_carry_ok() {
        assert_eq!(ok(json!({"a": 1})), json!({"a": 1, "ok": true}));
        assert_eq!(error("no")["ok"], false);
    }
}
