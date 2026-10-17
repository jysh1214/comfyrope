//! Putting a job's result into rope, in process through rope-core: what `rope add` and `rope replayed` do for
//! the direct path.

use anyhow::{Context, Result};
use rope_core::Library;
use rope_core::queue::protocol::{Kind, ResultEvent};
use rope_core::queue::{self, Job};
use rope_core::store::{self, NewImage};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Imports what the child delivered and answers what `result` records: for an image or clip, what `rope add`
/// answers (`id`, `short`, `path`, `name`, `folder`, ...); for a replay, the comparison record. The result's
/// files are left where they are; the caller removes `out/` once the job is done.
pub fn import(lib: &Library, job: &Job, ev: &ResultEvent, seconds: Option<f64>) -> Result<Value> {
    let dir = queue::job_dir(lib, job.id);
    let abs = |p: &Path| -> PathBuf { if p.is_absolute() { p.to_owned() } else { dir.join(p) } };
    let file = abs(&ev.file);
    let made = read_json(&abs(&ev.made))?;
    let graph = ev.graph.as_deref().map(|g| read_json(&abs(g))).transpose()?;
    let mut result = match ev.kind {
        Kind::Add => {
            let made = store::with_env_stored(lib, made)?;
            let folder = job.into.as_deref().map(|f| store::ensure_folder(lib, f).map(|(id, _)| id)).transpose()?;
            let new = NewImage {
                name: Some(ev.name.clone()),
                made: Some(made),
                graph,
                via: job.tool.clone(),
                folder,
                tags: job.tags.clone(),
                note: job.note.clone(),
                file_inputs: job.into.is_some(),
            };
            let added = store::add(lib, &file, &new)?;
            let meta = store::load(lib, &added.sha)?;
            store::added_json(lib, &meta, added.created, added.restored)?
        }
        Kind::Replayed => {
            let sha = ev
                .replay_of
                .clone()
                .or_else(|| job.replay.clone())
                .context("a replayed result names no image it replays")?;
            let mut record = store::replayed(lib, &sha, &file, made, graph)?;
            record["replay_of"] = json!(sha);
            record
        }
    };
    if let Some(s) = seconds {
        result["seconds"] = json!(s);
    }
    if let Some(t) = &ev.text {
        result["text"] = json!(t);
    }
    Ok(result)
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rope_core::queue::{Input, NewGroup, Submitted};

    fn png(path: &Path, seed: u8) {
        image_bytes(seed, path);
    }

    /// A small PNG written without the image crate: rope-core has it, this crate does not.
    fn image_bytes(seed: u8, path: &Path) {
        // 1x1 RGB PNG, built by hand: signature, IHDR, IDAT (stored deflate), IEND
        fn chunk(kind: &[u8], data: &[u8]) -> Vec<u8> {
            let mut out = (data.len() as u32).to_be_bytes().to_vec();
            out.extend(kind);
            out.extend(data);
            let mut crc = crc32(kind);
            crc = crc32_continue(crc, data);
            out.extend(crc.to_be_bytes());
            out
        }
        fn crc32(data: &[u8]) -> u32 {
            crc32_continue(0xFFFF_FFFF, data)
        }
        fn crc32_continue(mut crc: u32, data: &[u8]) -> u32 {
            for &b in data {
                crc ^= b as u32;
                for _ in 0..8 {
                    crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
                }
            }
            crc
        }
        fn adler(data: &[u8]) -> u32 {
            let (mut a, mut b) = (1u32, 0u32);
            for &d in data {
                a = (a + d as u32) % 65521;
                b = (b + a) % 65521;
            }
            (b << 16) | a
        }
        let raw = [0u8, seed, 0, 0]; // filter byte + one RGB pixel
        let mut zlib = vec![0x78, 0x01, 0x01, 4, 0, 0xFB, 0xFF];
        zlib.extend(raw);
        zlib.extend(adler(&raw).to_be_bytes());
        let mut ihdr = Vec::new();
        ihdr.extend(1u32.to_be_bytes());
        ihdr.extend(1u32.to_be_bytes());
        ihdr.extend([8, 2, 0, 0, 0]);
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend(chunk(b"IHDR", &ihdr));
        png.extend(chunk(b"IDAT", &zlib));
        png.extend(chunk(b"IEND", &[]));
        // the adler and crc above are real, but a wrong one would still be a file rope hashes and keeps
        std::fs::write(path, png).unwrap();
    }

    #[test]
    fn a_result_goes_into_the_jobs_folder_with_its_recipe() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(&dir.path().join("lib")).unwrap();
        let input = dir.path().join("in.png");
        png(&input, 1);
        let added = store::add(&lib, &input, &NewImage { via: "img2img.py".into(), ..Default::default() }).unwrap();
        let group = NewGroup {
            tool: "img2img.py".into(),
            prompt: "p".into(),
            inputs: vec![Input { id: added.sha.clone(), name: "in".into(), rendition: None }],
            into: Some("a/b".into()),
            tags: vec!["t".into()],
            note: Some("n".into()),
            flags: vec![vec!["--seed=1".into()]],
            submitted: Submitted::default(),
            ..Default::default()
        };
        let job = queue::submit(&lib, &group).unwrap().remove(0);
        let out = queue::out_dir(&lib, job.id);
        std::fs::create_dir_all(&out).unwrap();
        png(&out.join("r.png"), 2);
        let made = json!({"flags": ["--seed=1"], "prompt": "p", "inputs": [{"id": added.sha}], "env": {"gpu": "x"}});
        std::fs::write(out.join("made.json"), made.to_string()).unwrap();
        std::fs::write(out.join("graph.json"), r#"{"1": {"class_type": "LoadImage", "inputs": {}}}"#).unwrap();
        let ev = ResultEvent {
            file: "out/r.png".into(),
            made: "out/made.json".into(),
            graph: Some("out/graph.json".into()),
            name: "in_2511_s1".into(),
            kind: Kind::Add,
            replay_of: None,
            text: Some("enhanced".into()),
        };
        let result = import(&lib, &job, &ev, Some(6.5)).unwrap();
        assert_eq!((result["name"].as_str(), result["folder"].as_str()), (Some("in_2511_s1"), Some("a/b")));
        assert_eq!(
            (result["seconds"].as_f64(), result["text"].as_str(), result["created"].as_bool()),
            (Some(6.5), Some("enhanced"), Some(true))
        );
        let meta = store::load(&lib, result["id"].as_str().unwrap()).unwrap();
        assert!(meta.made.as_ref().unwrap()["env"].is_string(), "the env is stored by id");
        assert!(std::fs::read_to_string(lib.item_dir(&meta.sha256).join("note.txt")).unwrap().contains('n'));
        let catalog = rope_core::Catalog::load(&lib).unwrap();
        assert!(catalog.entry(&added.sha).folder.is_some(), "the unfiled input follows the result into the folder");
        assert!(catalog.entry(&meta.sha256).tags.contains(&"t".to_owned()));
        assert!(import(&lib, &job, &ResultEvent { file: "out/none.png".into(), ..ev.clone() }, None).is_err());
    }
}
