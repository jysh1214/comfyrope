//! The conversion stage: an input that is not a single-frame PNG, JPEG or WebP becomes a PNG before ComfyUI sees
//! it. The pixels are the Python converter's business (`comfyconvert.py`, run from the venv); this module runs it
//! once per distinct input, on a thread, and keeps the PNG in the library as that image's rendition.

use crate::daemon::Msg;
use crate::runner::Tools;
use anyhow::{Context, Result, bail};
use rope_core::{Library, store};
use serde_json::{Value, json};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;

#[derive(Debug, Clone, PartialEq)]
pub enum Converted {
    /// ComfyUI takes the file as it is.
    AsIs,
    /// The id of the PNG kept in the library in its place.
    Rendition(String),
    Failed(String),
}

/// Converts input `id` on a thread of its own; the outcome arrives as `Msg::Converted`.
pub fn spawn(tools: &Tools, lib: &Library, converter: &Path, id: String, tx: Sender<Msg>) {
    let (tools, lib, converter) = (tools.clone(), lib.clone(), converter.to_owned());
    std::thread::spawn(move || {
        let result = match convert(&tools, &lib, &converter, &id) {
            Ok(c) => c,
            Err(e) => Converted::Failed(format!("{e:#}")),
        };
        let _ = tx.send(Msg::Converted { input: id, result });
    });
}

fn convert(tools: &Tools, lib: &Library, converter: &Path, id: &str) -> Result<Converted> {
    let meta = store::load(lib, id)?;
    // a PNG made for it earlier serves again: the bytes have not changed
    if let Some(r) = kept_rendition(lib, id, &meta) {
        return Ok(Converted::Rendition(r));
    }
    let file = store::image_path(lib, &meta);
    let out = lib.queue_dir().join("convert").join(&id[..12]);
    std::fs::create_dir_all(&out).with_context(|| format!("cannot create {}", out.display()))?;
    let result = run(tools, lib, converter, id, &file, &out);
    let _ = std::fs::remove_dir_all(&out);
    result
}

fn run(tools: &Tools, lib: &Library, converter: &Path, id: &str, file: &Path, out: &Path) -> Result<Converted> {
    let mut cmd = Command::new(&tools.python);
    cmd.args(["-E", "-u"])
        .arg(converter)
        .arg(file)
        .arg(out)
        .arg("--json")
        .current_dir(&tools.repo)
        .env_clear()
        .envs(tools.env(Some(lib), None))
        .stdin(Stdio::null());
    // SAFETY: only async-signal-safe calls, see default_signals.
    unsafe {
        cmd.pre_exec(|| {
            crate::comfy::default_signals();
            Ok(())
        });
    }
    let output = cmd.output().with_context(|| format!("cannot run {}", converter.display()))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!("{}", err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("the converter failed"));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().rev().find(|l| !l.trim().is_empty()).context("the converter said nothing")?;
    let v: Value = serde_json::from_str(line).context("the converter's answer is not JSON")?;
    if v["as_is"] == true {
        return Ok(Converted::AsIs);
    }
    let name = v["file"].as_str().context("the converter named no file")?;
    let png = if Path::new(name).is_absolute() { PathBuf::from(name) } else { out.join(name) };
    let info = json!({"steps": v["steps"], "libs": v["libs"]});
    let (rsha, _) = store::add_rendition(lib, id, &png, info)?;
    Ok(Converted::Rendition(rsha))
}

/// A rendition kept with the image from an earlier conversion, if its file is still there.
fn kept_rendition(lib: &Library, id: &str, meta: &rope_core::Meta) -> Option<String> {
    let up = lib.item_dir(id).join("uploaded");
    meta.uploaded
        .iter()
        .find(|(_, info)| info["file"].as_str().is_some_and(|f| up.join(f).exists()))
        .map(|(rsha, _)| rsha.clone())
}
