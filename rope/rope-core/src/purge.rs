//! Deleting images for good.
//!
//! Deleting an image that others were made from would leave them impossible to replay. So by default such an
//! image keeps its bytes (and the converted files uploaded in its place) and loses everything else: name,
//! sources, recipe, note, folder, tags. It goes for good once the last image made from it is purged.

use crate::catalog::Catalog;
use crate::hash::short;
use crate::{Library, Meta, atomic, journal, store, time};
use anyhow::{Result, bail};
use serde_json::json;
use std::collections::{BTreeMap, HashSet};
use std::fs;

#[derive(Debug, Default)]
pub struct Purged {
    pub deleted: Vec<String>,
    /// Images whose bytes stay, with the images made from them.
    pub kept: Vec<(String, Vec<String>)>,
}

/// Purges `ids`. With `break_lineage`, images others were made from go too, and those can no longer be replayed.
pub fn purge(lib: &Library, ids: &[String], break_lineage: bool) -> Result<Purged> {
    let Some(_run) = lib.try_run_lock()? else {
        bail!("img2img.py is running; purge again when it is done, since it could delete an input that run uses");
    };
    let _lock = lib.lock()?;
    let (metas, problems) = store::scan(lib)?;
    if !problems.is_empty() {
        bail!(
            "purge reads every meta.json to learn which images are still needed, and these cannot be read:\n  {}",
            problems.join("\n  ")
        );
    }
    let known: HashSet<&str> = metas.iter().map(|m| m.sha256.as_str()).collect();
    let targets: HashSet<String> = ids.iter().cloned().collect();
    for id in &targets {
        if !known.contains(id.as_str()) {
            bail!("no image {} in rope", short(id));
        }
    }
    let staying: Vec<Meta> = metas.iter().filter(|m| !targets.contains(&m.sha256)).cloned().collect();
    let users = users_of(&staying);

    let mut catalog = Catalog::load(lib)?;
    let mut out = Purged::default();
    let mut order: Vec<&String> = targets.iter().collect();
    order.sort();
    for id in order {
        catalog.forget(id);
        match users.get(id.as_str()) {
            Some(children) if !break_lineage => {
                keep(lib, id, children)?;
                out.kept.push((id.clone(), children.clone()));
            }
            _ => {
                delete_item(lib, id)?;
                out.deleted.push(id.clone());
            }
        }
    }
    // An image kept earlier goes once nothing made from it is left.
    loop {
        let (metas, _) = store::scan(lib)?;
        let users = users_of(&metas);
        let orphans: Vec<String> = metas
            .iter()
            .filter(|m| m.kept.is_some() && !users.contains_key(m.sha256.as_str()))
            .map(|m| m.sha256.clone())
            .collect();
        if orphans.is_empty() {
            break;
        }
        for id in orphans {
            catalog.forget(&id);
            delete_item(lib, &id)?;
            out.kept.retain(|(k, _)| *k != id);
            out.deleted.push(id);
        }
    }
    catalog.save(lib)?;
    let mut changed: Vec<String> = out.deleted.clone();
    changed.extend(out.kept.iter().map(|(k, _)| k.clone()));
    journal::append(lib, "purge", &changed)?;
    Ok(out)
}

/// For each image used as an input, the images (not themselves kept) made from it.
fn users_of(metas: &[Meta]) -> BTreeMap<String, Vec<String>> {
    let mut users: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for m in metas.iter().filter(|m| m.kept.is_none()) {
        for p in m.upstream() {
            let list = users.entry(p).or_default();
            if !list.contains(&m.sha256) {
                list.push(m.sha256.clone());
            }
        }
    }
    users
}

/// Strips an image down to its bytes, its converted uploads and a minimal meta.json.
fn keep(lib: &Library, id: &str, children: &[String]) -> Result<()> {
    let dir = lib.item_dir(id);
    let mut meta = store::load(lib, id)?;
    for f in ["note.txt", "graph.json", "repro.sh"] {
        let p = dir.join(f);
        if p.exists() {
            fs::remove_file(p)?;
        }
    }
    meta.name = String::new();
    meta.sources.clear();
    meta.made = None;
    meta.replays.clear();
    meta.kept = Some(json!({"since": time::now(), "for": children}));
    atomic::write(&dir.join("meta.json"), &meta.to_json())?;
    remove_thumbnails(lib, id);
    Ok(())
}

fn delete_item(lib: &Library, id: &str) -> Result<()> {
    let dir = lib.item_dir(id);
    // out of items/ in one rename first, so a crash midway never leaves half an item there
    let doomed = atomic::temp_path(&lib.tmp_dir().join(format!("purge-{id}")));
    fs::rename(&dir, &doomed)?;
    fs::remove_dir_all(&doomed)?;
    remove_thumbnails(lib, id);
    Ok(())
}

fn remove_thumbnails(lib: &Library, id: &str) {
    let shard = lib.cache_dir().join("thumbs").join(&id[..2]);
    if let Ok(entries) = fs::read_dir(&shard) {
        for e in entries.flatten() {
            if e.file_name().to_string_lossy().starts_with(id) {
                let _ = fs::remove_file(e.path());
            }
        }
    }
}
