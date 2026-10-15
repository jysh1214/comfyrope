//! The img2img.py runs in progress, so that purge leaves their inputs alone and nothing else.
//!
//! A run writes `runs/<pid>.json`, naming the images it uses, and holds an exclusive flock on that file until the
//! process ends. The lock, not the file, says that the run is alive: a file whose lock can be taken was left by a
//! run that is gone (a crash, a kill), and is removed on sight.

use crate::Library;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{self, File};

#[derive(Debug, Clone, Deserialize)]
pub struct Run {
    pub pid: u32,
    #[serde(default)]
    pub started: String,
    /// The images the run uses: its inputs, and for a replay, the image replayed.
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub command: String,
}

/// The runs alive now. Files of runs that have ended are removed on the way.
pub fn running(lib: &Library) -> Result<Vec<Run>> {
    scan(lib).map(|(runs, _)| runs)
}

/// The runs alive now, and the names of the files removed because the run that wrote each has ended.
pub fn scan(lib: &Library) -> Result<(Vec<Run>, Vec<String>)> {
    let (mut out, mut removed) = (Vec::new(), Vec::new());
    let Ok(entries) = fs::read_dir(lib.runs_dir()) else { return Ok((out, removed)) };
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let Ok(file) = File::open(&path) else { continue }; // gone since the listing
        if crate::library::try_lock(&file).with_context(|| format!("cannot lock {}", path.display()))? {
            if fs::remove_file(&path).is_ok() {
                removed.push(path.file_name().unwrap_or_default().to_string_lossy().into_owned());
            }
        } else {
            let text = fs::read_to_string(&path).with_context(|| format!("cannot read {}", path.display()))?;
            // a run that has just taken its lock may not have written the file yet; it holds run.lock shared
            // meanwhile, which purge excludes, so this is only for a reader outside purge
            if let Ok(run) = serde_json::from_str::<Run>(&text) {
                out.push(run);
            }
        }
    }
    out.sort_by_key(|r| r.pid);
    Ok((out, removed))
}

/// Each image some live run uses, with the pid of that run.
pub fn in_use(lib: &Library) -> Result<HashMap<String, u32>> {
    let mut out = HashMap::new();
    for run in running(lib)? {
        for id in run.inputs {
            out.entry(id).or_insert(run.pid);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_locked_file_is_a_live_run_and_an_unlocked_one_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(dir.path()).unwrap();
        let path = lib.runs_dir().join("4242.json");
        fs::write(&path, r#"{"pid": 4242, "inputs": ["a", "b"]}"#).unwrap();
        fs::write(lib.runs_dir().join("notes"), "ignored").unwrap();
        let held = File::open(&path).unwrap();
        held.lock().unwrap();
        let runs = running(&lib).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!((runs[0].pid, runs[0].inputs.as_slice()), (4242, &["a".to_owned(), "b".to_owned()][..]));
        assert_eq!(in_use(&lib).unwrap().get("b"), Some(&4242));
        drop(held);
        let (live, removed) = scan(&lib).unwrap();
        assert!(live.is_empty() && removed == ["4242.json"], "{live:?} {removed:?}");
        assert!(!path.exists(), "the file of a run that ended is removed");
        assert!(lib.runs_dir().join("notes").exists());
    }
}
