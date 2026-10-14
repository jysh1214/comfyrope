//! Finding, and where it is safe, fixing what is inconsistent in a library.

use crate::catalog::Catalog;
use crate::hash::{self, short};
use crate::{Library, journal, store};
use anyhow::Result;
use std::collections::HashSet;
use std::fs;
use std::time::{Duration, SystemTime};

#[derive(Debug, Default)]
pub struct Report {
    /// What is wrong and stays wrong (or was fixed, then it is also in `fixed`).
    pub problems: Vec<String>,
    pub fixed: Vec<String>,
}

/// Checks every item and the catalog. With `hash`, every stored file is hashed again. With `fix`, catalog
/// entries for missing images are dropped, references to missing folders are cleared, stale scratch space is
/// removed and outdated repro.sh files are written again; nothing is fixed by deleting an image.
pub fn check(lib: &Library, hash: bool, fix: bool) -> Result<Report> {
    let mut r = Report::default();
    let (metas, unreadable) = store::scan(lib)?;
    r.problems.extend(unreadable.into_iter().map(|p| format!("unreadable meta.json: {p}")));
    let ids: HashSet<&str> = metas.iter().map(|m| m.sha256.as_str()).collect();

    for m in &metas {
        let dir = lib.item_dir(&m.sha256);
        if dir.file_name().and_then(|n| n.to_str()) != Some(m.sha256.as_str()) {
            r.problems.push(format!("{}: meta.json names another sha256", short(&m.sha256)));
        }
        let file = dir.join(&m.file);
        if !file.exists() {
            r.problems.push(format!("{}: the image file {} is missing", m.short(), m.file));
        } else if hash && hash::sha256_file(&file)? != m.sha256 {
            r.problems.push(format!("{}: the image file no longer matches its hash (changed on disk)", m.short()));
        }
        for (rsha, info) in &m.uploaded {
            let name = info.get("file").and_then(|f| f.as_str()).unwrap_or_default();
            let path = dir.join("uploaded").join(name);
            if name.is_empty() || !path.exists() {
                r.problems.push(format!("{}: the uploaded file {} is missing", m.short(), short(rsha)));
            } else if hash && hash::sha256_file(&path)? != *rsha {
                r.problems.push(format!("{}: the uploaded file {} no longer matches its hash", m.short(), short(rsha)));
            }
        }
        for (i, input) in m.inputs().iter().enumerate() {
            match &input.id {
                Some(id) if !ids.contains(id.as_str()) => r.problems.push(format!(
                    "{}: input {} ({}) is not in the library, so it cannot be replayed",
                    m.short(),
                    i + 1,
                    short(id)
                )),
                None => r.problems.push(format!("{}: input {} was never found", m.short(), i + 1)),
                _ => {}
            }
        }
    }

    let _lock = lib.lock()?;
    let mut catalog = Catalog::load(lib)?;
    let mut changed = false;
    let gone: Vec<String> = catalog.items.keys().filter(|k| !ids.contains(k.as_str())).cloned().collect();
    for id in gone {
        r.problems.push(format!("catalog lists {}, which is not in the library", short(&id)));
        if fix {
            catalog.forget(&id);
            r.fixed.push(format!("dropped {} from the catalog", short(&id)));
            changed = true;
        }
    }
    let folders: HashSet<String> = catalog.folders.keys().cloned().collect();
    let orphaned: Vec<String> = catalog
        .items
        .iter()
        .filter(|(_, e)| e.folder.as_ref().is_some_and(|f| !folders.contains(f)))
        .map(|(id, _)| id.clone())
        .collect();
    for id in orphaned {
        r.problems.push(format!("{} is in a folder that does not exist", short(&id)));
        if fix {
            catalog.set_folder(&id, None);
            r.fixed.push(format!("{} is unfiled now", short(&id)));
            changed = true;
        }
    }
    let lost: Vec<String> = catalog
        .folders
        .iter()
        .filter(|(_, f)| f.parent.as_ref().is_some_and(|p| !folders.contains(p)))
        .map(|(id, _)| id.clone())
        .collect();
    for id in lost {
        r.problems.push(format!("folder {:?} has a parent that does not exist", catalog.folders[&id].name));
        if fix {
            catalog.folders.get_mut(&id).expect("listed").parent = None;
            r.fixed.push(format!("folder {:?} is at the top now", catalog.folders[&id].name));
            changed = true;
        }
    }
    if changed {
        catalog.save(lib)?;
        journal::append(lib, "check", &[])?;
    }

    if let Ok(entries) = fs::read_dir(lib.root().join("rescue")) {
        for e in entries.flatten().filter(|e| e.path().is_dir()) {
            r.problems.push(format!(
                "rescue/{0} holds a result img2img.py could not hand to rope; run rescue/{0}/bring-in.sh, then delete \
                 the folder",
                e.file_name().to_string_lossy()
            ));
        }
    }

    let hour_ago = SystemTime::now() - Duration::from_secs(3600);
    if let Ok(entries) = fs::read_dir(lib.tmp_dir()) {
        for e in entries.flatten() {
            let old = e.metadata().and_then(|m| m.modified()).is_ok_and(|t| t < hour_ago);
            if old {
                r.problems.push(format!("leftover scratch space tmp/{}", e.file_name().to_string_lossy()));
                if fix {
                    let p = e.path();
                    let _ = if p.is_dir() { fs::remove_dir_all(&p) } else { fs::remove_file(&p) };
                    r.fixed.push(format!("removed tmp/{}", e.file_name().to_string_lossy()));
                }
            }
        }
    }
    drop(_lock);
    if fix {
        let n = store::rewrite_repro_scripts(lib)?;
        if n > 0 {
            r.fixed.push(format!("wrote {n} repro.sh again"));
        }
    }
    Ok(r)
}
