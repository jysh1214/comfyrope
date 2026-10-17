//! journal.jsonl: one line per change, only ever appended. The GUI follows it to see changes made by the CLI and
//! by img2img.py, and `rope log` prints it. It is a record, not a source of truth.

use crate::{Library, time};
use anyhow::{Context, Result};
use serde_json::{Map, Value, json};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

/// Appends one line. The file is opened and closed for each line, so a file watcher sees every change.
pub fn append(lib: &Library, op: &str, ids: &[String]) -> Result<()> {
    append_with(lib, op, ids, Value::Null)
}

/// Appends one line with more than `at`, `op` and `ids`: the entries of `fields`, a JSON object, follow them. For
/// changes that are not about images, such as the queue's; `ids` must hold only image hashes, since the GUI
/// loads every one of them.
pub fn append_with(lib: &Library, op: &str, ids: &[String], fields: Value) -> Result<()> {
    let mut line = Map::new();
    line.insert("at".to_owned(), json!(time::now()));
    line.insert("op".to_owned(), json!(op));
    line.insert("ids".to_owned(), json!(ids));
    if let Value::Object(fields) = fields {
        for (k, v) in fields {
            line.entry(k).or_insert(v);
        }
    }
    let mut line = Value::Object(line).to_string();
    line.push('\n');
    let path = lib.journal_path();
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(line.as_bytes()))
        .with_context(|| format!("cannot append to {}", path.display()))
}

/// The complete lines after byte `offset`, and the offset after the last of them. A line still being written is
/// left for the next call.
pub fn read_from(lib: &Library, offset: u64) -> Result<(Vec<Value>, u64)> {
    let mut f = match std::fs::File::open(lib.journal_path()) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(e.into()),
    };
    let len = f.metadata()?.len();
    let start = if offset > len { 0 } else { offset }; // the journal was replaced: start over
    f.seek(SeekFrom::Start(start))?;
    let mut text = String::new();
    f.read_to_string(&mut text)?;
    let complete = text.rfind('\n').map_or(0, |i| i + 1);
    let events = text[..complete].lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    Ok((events, start + complete as u64))
}
