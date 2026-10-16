//! Making an image again: each item's repro.sh, and the commands the CLI and the GUI copy.
//!
//! Nothing here stores an absolute path that can go stale. repro.sh finds the library from its own location and
//! the comfyrope repo from the library's REPO file; the copied commands are built when asked for.

use crate::catalog::shell_word;
use crate::{Library, Meta};
use anyhow::{Context, Result, bail};
use serde_json::Value;

/// The repro.sh of an image or clip with a recipe: it runs the tool that made it (`img2img.py`, `img2vid.py`)
/// with `--replay` on this item, passing on its arguments. The comments say what the recipe was.
pub fn script(meta: &Meta) -> String {
    let made = meta.made.as_ref();
    let get = |key: &str| made.and_then(|m| m.get(key));
    let (tool, kind) = (meta.tool(), if meta.is_clip() { "clip" } else { "image" });
    let mut s = String::from("#!/bin/sh\n");
    s += &format!("# Makes rope {kind} {} again: {}\n", meta.short(), clean(&meta.name));
    let at = get("at").and_then(Value::as_str).unwrap_or(&meta.added);
    let mut line = format!("# Made {at}");
    if let Some(b) = get("batch") {
        let (i, n, first) = (b["index"].as_u64(), b["count"].as_u64(), b["first_seed"].as_u64());
        if let (Some(i), Some(n), Some(first)) = (i, n, first)
            && n > 1
        {
            line += &format!(" as {i} of {n} (seeds {first}-{})", first + n - 1);
        }
    }
    let blob_key = format!("{}_blob", tool.trim_end_matches(".py"));
    if let Some(code) = get("code")
        && let Some(blob) = code[blob_key.as_str()].as_str()
    {
        line += &format!(" with {tool} {}", &blob[..blob.len().min(8)]);
        if let Some(commit) = code["commit"].as_str() {
            let dirty = if code["dirty"].as_bool() == Some(true) { ", modified" } else { "" };
            line += &format!(" ({commit}{dirty})");
        }
    }
    s += &line;
    s.push('\n');
    if let Some(server) = get("server").and_then(Value::as_str).filter(|v| !v.is_empty()) {
        s += &format!("# by {}\n", clean(server));
    }
    let flags = meta.flags();
    if !flags.is_empty() {
        s += &format!("# Recipe: {}\n", clean(&shell_join(&flags)));
    }
    for (i, input) in meta.inputs().iter().enumerate() {
        let id = input.id.as_deref().map_or("missing", crate::hash::short);
        let upload = input.upload.as_deref().map(|u| format!(" (uploaded as {})", clean(u))).unwrap_or_default();
        s += &format!("#   image {}: {id}{upload}\n", i + 1);
    }
    for (key, label) in [("prompt", "prompt"), ("enhanced_prompt", "enhanced prompt")] {
        if let Some(text) = get(key).and_then(Value::as_str).filter(|t| !t.is_empty()) {
            let pad = " ".repeat(label.len() + 2);
            for (i, l) in text.lines().enumerate() {
                let lead = if i == 0 { format!("{label}: ") } else { pad.clone() };
                s += &format!("#   {lead}{}\n", clean(l));
            }
        }
    }
    s += &format!(
        "# It replays the recorded workflow on the recorded input bytes, then compares the result with this {kind}.\n"
    );
    s += &format!(
        "# Arguments are passed on: --dry-run prints the workflow, -s N tries seed N, --rebuild uses today's {tool}.\n"
    );
    s += &format!(
        r#"lib=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
for repo in "${{IMG2IMG_REPO:-}}" "$(cat "$lib/REPO" 2>/dev/null)"; do
    [ -n "$repo" ] && [ -f "$repo/{tool}" ] && break
    repo=
done
if [ -z "$repo" ]; then
    echo "repro.sh: cannot find {tool}; set IMG2IMG_REPO to the comfyrope repo" >&2
    exit 1
fi
py="$repo/.venv/bin/python3"
[ -x "$py" ] || py=""
"#
    );
    s += &format!("exec ${{py:+\"$py\"}} \"$repo/{tool}\" --library \"$lib\" --replay {} \"$@\"\n", meta.short());
    s
}

/// The command that replays an image: what "copy repro" copies.
pub fn replay_argv(lib: &Library, meta: &Meta) -> Result<Vec<String>> {
    if meta.made.is_none() {
        bail!("{} has no recipe: rope did not see it being made", meta.short());
    }
    let mut argv = tool(lib, meta.tool())?;
    if !lib.is_default() {
        argv.push(format!("--library={}", lib.root().display()));
    }
    argv.push(format!("--replay={}", meta.short()));
    Ok(argv)
}

/// The full command for an image or clip, with its flags and its inputs as `@id`: a starting point for a
/// variation, made with today's tool (`img2img.py`, `img2vid.py`: the one that made it).
pub fn command_argv(lib: &Library, meta: &Meta, folder: Option<&str>) -> Result<Vec<String>> {
    if meta.made.is_none() {
        bail!("{} has no recipe: rope did not see it being made", meta.short());
    }
    let mut argv = tool(lib, meta.tool())?;
    if !lib.is_default() {
        argv.push(format!("--library={}", lib.root().display()));
    }
    argv.extend(meta.flags());
    if let Some(f) = folder.filter(|f| !f.is_empty()) {
        argv.push(format!("--into={f}"));
    }
    argv.push("--".into());
    for (i, input) in meta.inputs().iter().enumerate() {
        let id = input.id.as_deref().with_context(|| format!("image {} of the recipe is missing from rope", i + 1))?;
        argv.push(format!("@{}", crate::hash::short(id)));
    }
    argv.push(meta.made_str("prompt").unwrap_or_default().to_owned());
    Ok(argv)
}

/// How to run `script`, a tool in the comfyrope repo: with the repo's .venv Python if there is one, else the
/// script itself (uv supplies its packages).
fn tool(lib: &Library, script: &str) -> Result<Vec<String>> {
    let repo = std::env::var_os("IMG2IMG_REPO")
        .filter(|r| !r.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| lib.repo())
        .context("rope does not know where the comfyrope repo is yet: run img2img.py once, or set IMG2IMG_REPO")?;
    let script = repo.join(script);
    let py = repo.join(".venv/bin/python3");
    let mut argv = Vec::new();
    if py.exists() {
        argv.push(py.display().to_string());
    }
    argv.push(script.display().to_string());
    Ok(argv)
}

pub fn shell_join(argv: &[String]) -> String {
    argv.iter().map(|a| shell_word(a)).collect::<Vec<_>>().join(" ")
}

/// One line of a comment: control characters become spaces.
fn clean(s: &str) -> String {
    s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn made_meta() -> Meta {
        serde_json::from_value(json!({
            "format": 1, "sha256": "3fa9c1e27b0d".repeat(5) + "abcd", "file": "3fa9c1e27b0d.png",
            "name": "20261014-101919_photo_2511_s7", "added": "2026-10-14T10:19:31+08:00", "image": {"bytes": 1},
            "made": {"at": "2026-10-14T10:19:19+08:00", "flags": ["--model=2511", "--seed=7", "--negative=a b"],
                     "prompt": "第一行\n$(rm -rf ~)", "batch": {"index": 2, "count": 4, "first_seed": 6},
                     "inputs": [{"id": "9b2e0f1a2b3c".repeat(5) + "abcd", "upload": "img2img/9b2e0f1a2b3c_photo.jpg"}]}
        }))
        .unwrap()
    }

    #[test]
    fn the_script_keeps_the_recipe_in_comments_and_runs_only_the_replay() {
        let s = script(&made_meta());
        assert!(s.contains("# Made 2026-10-14T10:19:19+08:00 as 2 of 4 (seeds 6-9)\n"), "{s}");
        assert!(s.contains("#   prompt: 第一行\n#           $(rm -rf ~)\n"), "{s}");
        assert!(s.contains("#   image 1: 9b2e0f1a2b3c (uploaded as img2img/9b2e0f1a2b3c_photo.jpg)\n"), "{s}");
        let code: Vec<&str> = s.lines().filter(|l| !l.starts_with('#')).collect();
        assert!(code.last().unwrap().ends_with("--replay 3fa9c1e27b0d \"$@\""), "{s}");
        assert!(!code.iter().any(|l| l.contains("rm -rf")), "the prompt only ever appears in comments");
    }

    #[test]
    fn a_clips_script_runs_the_tool_that_made_it_and_only_a_tool_in_the_repo() {
        let mut meta = made_meta();
        let made = meta.made.as_mut().unwrap();
        made["by"] = json!("img2vid.py");
        made["code"] = json!({"img2vid_blob": "abcdef0123456789", "commit": "cd9b57f"});
        meta.video = Some(crate::meta::VideoInfo::default());
        let s = script(&meta);
        assert!(s.contains("# Makes rope clip 3fa9c1e27b0d again"), "{s}");
        assert!(s.contains(" with img2vid.py abcdef01 (cd9b57f)\n"), "{s}");
        assert!(s.contains("[ -f \"$repo/img2vid.py\" ]") && s.contains("\"$repo/img2vid.py\" --library"), "{s}");
        assert!(!s.contains("img2img.py"), "{s}");
        for bad in ["../evil.py", "/tmp/x.py", "evil", ".hidden.py"] {
            meta.made.as_mut().unwrap()["by"] = json!(bad);
            assert_eq!(meta.tool(), "img2img.py", "{bad}");
        }
    }

    #[test]
    fn the_script_is_valid_sh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("repro.sh");
        std::fs::write(&path, script(&made_meta())).unwrap();
        let status = std::process::Command::new("sh").arg("-n").arg(&path).status().unwrap();
        assert!(status.success());
    }
}
