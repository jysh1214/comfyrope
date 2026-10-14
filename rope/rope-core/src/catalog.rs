//! catalog.json: the folder tree, which folder each image is in, its tags, and the trash.
//!
//! Folders exist only here, not on disk. Each has an id that never changes, so a rename or a move is one write,
//! and a job that resolved a folder before it was renamed still files its results in the right place. An image
//! is in at most one folder; one in none is unfiled. Tags are names; an image can have any number.

use crate::{Library, atomic};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub format: u32,
    #[serde(default)]
    pub folders: BTreeMap<String, Folder>,
    /// Images with a folder, tags, or a place in the trash, by SHA-256. An image with none of these has no entry.
    #[serde(default)]
    pub items: BTreeMap<String, Entry>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Folder {
    pub name: String,
    #[serde(default)]
    pub parent: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folder: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// When it went into the trash. A trashed image keeps its folder and tags, for a restore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trashed: Option<String>,
}

impl Default for Catalog {
    fn default() -> Self {
        Catalog { format: FORMAT, folders: BTreeMap::new(), items: BTreeMap::new(), other: Map::new() }
    }
}

impl Catalog {
    /// Reads catalog.json; a library without one has an empty catalog. A catalog that does not parse is an
    /// error, never an empty catalog: saving that would lose every folder.
    pub fn load(lib: &Library) -> Result<Catalog> {
        let path = lib.catalog_path();
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Catalog::default()),
            Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
        };
        let catalog: Catalog = serde_json::from_slice(&bytes).with_context(|| {
            format!("{} does not parse; the previous version is in catalog.json.bak", path.display())
        })?;
        if catalog.format > FORMAT {
            bail!("{} has format {}; this rope reads up to {FORMAT}", path.display(), catalog.format);
        }
        Ok(catalog)
    }

    /// Writes catalog.json, keeping the previous version as catalog.json.bak. The caller holds the library lock
    /// and loaded this catalog under it. There is no moment without a complete catalog.json.
    pub fn save(&self, lib: &Library) -> Result<()> {
        let path = lib.catalog_path();
        if path.exists() {
            let bak = lib.root().join("catalog.json.bak");
            let tmp = atomic::temp_path(&bak);
            fs::hard_link(&path, &tmp).and_then(|()| fs::rename(&tmp, &bak)).ok();
        }
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        atomic::write(&path, &bytes)
    }

    // ---------------------------------------------------------------- folders

    /// A folder's path from the top, like `trips/kyoto`.
    pub fn folder_path(&self, id: &str) -> String {
        let mut names = Vec::new();
        let mut cur = Some(id.to_owned());
        while let Some(id) = cur {
            let Some(f) = self.folders.get(&id) else { break };
            names.push(f.name.clone());
            cur = f.parent.clone();
            if names.len() > self.folders.len() {
                break; // a cycle in a hand-edited file
            }
        }
        names.reverse();
        names.join("/")
    }

    /// The subfolders of `parent` (`None`: the top), sorted by name.
    pub fn children(&self, parent: Option<&str>) -> Vec<(&str, &Folder)> {
        let mut v: Vec<_> = self
            .folders
            .iter()
            .filter(|(_, f)| f.parent.as_deref() == parent)
            .map(|(id, f)| (id.as_str(), f))
            .collect();
        v.sort_by(|a, b| a.1.name.cmp(&b.1.name));
        v
    }

    fn child(&self, parent: Option<&str>, name: &str) -> Option<String> {
        self.folders.iter().find(|(_, f)| f.parent.as_deref() == parent && f.name == name).map(|(id, _)| id.clone())
    }

    /// The folder at `path`: `Some(None)` for the top (`/` or empty), `None` when there is no such folder.
    pub fn find(&self, path: &str) -> Option<Option<String>> {
        let mut cur: Option<String> = None;
        for name in split(path) {
            cur = Some(self.child(cur.as_deref(), name)?);
        }
        Some(cur)
    }

    /// Like `find`, but a missing folder is an error that names similar folders.
    pub fn resolve(&self, path: &str) -> Result<Option<String>> {
        if let Some(id) = self.find(path) {
            return Ok(id);
        }
        let last = split(path).last().unwrap_or("").to_lowercase();
        let mut similar: Vec<String> = self
            .folders
            .iter()
            .filter(|(_, f)| !last.is_empty() && f.name.to_lowercase().contains(&last))
            .map(|(id, _)| self.folder_path(id))
            .collect();
        similar.sort();
        let hint = if similar.is_empty() { String::new() } else { format!("; similar: {}", similar.join(", ")) };
        bail!("no folder {path:?} in rope (make it with: rope mkdir -p {}){hint}", shell_word(path))
    }

    /// Makes the folder at `path` and returns its id; with `parents`, missing ancestors too, and an existing
    /// folder is fine.
    pub fn mkdir(&mut self, path: &str, parents: bool) -> Result<String> {
        let names: Vec<&str> = split(path).collect();
        if names.is_empty() {
            bail!("a folder needs a name");
        }
        let mut cur: Option<String> = None;
        for (i, name) in names.iter().enumerate() {
            check_name(name)?;
            let last = i + 1 == names.len();
            match self.child(cur.as_deref(), name) {
                Some(_) if last && !parents => bail!("folder {path:?} exists"),
                Some(id) => cur = Some(id),
                None if !last && !parents => {
                    bail!("no folder {:?}; use -p to make missing parents", names[..=i].join("/"))
                }
                None => {
                    let id = self.new_id();
                    self.folders.insert(id.clone(), Folder { name: (*name).to_owned(), parent: cur.clone() });
                    cur = Some(id);
                }
            }
        }
        Ok(cur.expect("at least one name"))
    }

    /// Renames or moves the folder at `from` to `to`, a full path whose parent must exist: `mv a/b c/d`.
    pub fn move_folder(&mut self, from: &str, to: &str) -> Result<()> {
        let id = self.resolve(from)?.context("the top cannot be moved")?;
        let mut names: Vec<&str> = split(to).collect();
        let name = names.pop().context("the new path needs a name")?;
        check_name(name)?;
        let parent = self.resolve(&names.join("/"))?;
        if let Some(p) = &parent
            && (*p == id || self.ancestors(p).contains(&id))
        {
            bail!("cannot move {from:?} into itself");
        }
        if let Some(other) = self.child(parent.as_deref(), name)
            && other != id
        {
            bail!("folder {to:?} exists");
        }
        let f = self.folders.get_mut(&id).expect("resolved");
        f.name = name.to_owned();
        f.parent = parent;
        Ok(())
    }

    /// Deletes the folder at `path`, which must hold no folders and no images, trashed ones included.
    pub fn rmdir(&mut self, path: &str) -> Result<()> {
        let id = self.resolve(path)?.context("the top cannot be removed")?;
        if !self.children(Some(&id)).is_empty() {
            bail!("folder {path:?} has subfolders");
        }
        let n = self.items.values().filter(|e| e.folder.as_deref() == Some(&id)).count();
        if n > 0 {
            bail!("folder {path:?} holds {n} image(s); move them out or trash them, and empty the trash, first");
        }
        self.folders.remove(&id);
        Ok(())
    }

    /// Removes the folder at `path` with everything below it: its subfolders go, and its images go to the trash
    /// (`at` is the time). They lose their folder, since it no longer exists, so a restore leaves them unfiled.
    /// Returns how many folders went, and the images newly trashed.
    pub fn remove_tree(&mut self, path: &str, at: &str) -> Result<(usize, Vec<String>)> {
        let id = self.resolve(path)?.context("the top cannot be removed")?;
        let doomed = self.subtree(&id);
        let mut trashed = Vec::new();
        for (sha, e) in self.items.iter_mut() {
            if e.folder.as_ref().is_some_and(|f| doomed.contains(f)) {
                if e.trashed.is_none() {
                    e.trashed = Some(at.to_owned());
                    trashed.push(sha.clone());
                }
                e.folder = None;
            }
        }
        for f in &doomed {
            self.folders.remove(f);
        }
        Ok((doomed.len(), trashed))
    }

    /// The images in a folder and the folders below it, outside the trash, and how many folders those are.
    pub fn tree_size(&self, id: &str) -> (usize, usize) {
        let folders = self.subtree(id);
        let images = self
            .items
            .values()
            .filter(|e| e.trashed.is_none() && e.folder.as_ref().is_some_and(|f| folders.contains(f)))
            .count();
        (folders.len(), images)
    }

    /// The ids of a folder's ancestors, nearest first.
    fn ancestors(&self, id: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = self.folders.get(id).and_then(|f| f.parent.clone());
        while let Some(p) = cur {
            if out.contains(&p) {
                break;
            }
            cur = self.folders.get(&p).and_then(|f| f.parent.clone());
            out.push(p);
        }
        out
    }

    /// `id` and every folder below it.
    pub fn subtree(&self, id: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::from([id.to_owned()]);
        loop {
            let more: Vec<String> = self
                .folders
                .iter()
                .filter(|(c, f)| !out.contains(*c) && f.parent.as_ref().is_some_and(|p| out.contains(p)))
                .map(|(c, _)| c.clone())
                .collect();
            if more.is_empty() {
                return out;
            }
            out.extend(more);
        }
    }

    fn new_id(&self) -> String {
        loop {
            let id = random_id();
            if !self.folders.contains_key(&id) {
                return id;
            }
        }
    }

    // ---------------------------------------------------------------- images

    pub fn entry(&self, sha: &str) -> Entry {
        self.items.get(sha).cloned().unwrap_or_default()
    }

    fn update(&mut self, sha: &str, change: impl FnOnce(&mut Entry)) {
        let mut e = self.entry(sha);
        change(&mut e);
        if e == Entry::default() {
            self.items.remove(sha);
        } else {
            self.items.insert(sha.to_owned(), e);
        }
    }

    pub fn set_folder(&mut self, sha: &str, folder: Option<String>) {
        self.update(sha, |e| e.folder = folder);
    }

    /// Adds `tag`; returns whether the image lacked it.
    pub fn add_tag(&mut self, sha: &str, tag: &str) -> Result<bool> {
        let tag = check_tag(tag)?;
        let mut added = false;
        self.update(sha, |e| {
            if !e.tags.iter().any(|t| t == tag) {
                e.tags.push(tag.to_owned());
                added = true;
            }
        });
        Ok(added)
    }

    /// Removes `tag`; returns whether the image had it.
    pub fn remove_tag(&mut self, sha: &str, tag: &str) -> bool {
        let mut removed = false;
        self.update(sha, |e| {
            let before = e.tags.len();
            e.tags.retain(|t| t != tag);
            removed = e.tags.len() != before;
        });
        removed
    }

    /// Renames a tag on every image, merging into `new` where both are present; returns how many images had it.
    pub fn rename_tag(&mut self, old: &str, new: &str) -> Result<usize> {
        let new = check_tag(new)?;
        let mut n = 0;
        for e in self.items.values_mut() {
            if let Some(i) = e.tags.iter().position(|t| t == old) {
                n += 1;
                if e.tags.iter().any(|t| t == new) {
                    e.tags.remove(i);
                } else {
                    e.tags[i] = new.to_owned();
                }
            }
        }
        Ok(n)
    }

    /// Every tag on an image outside the trash, with how many images carry it.
    pub fn tags(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for e in self.items.values().filter(|e| e.trashed.is_none()) {
            for t in &e.tags {
                *out.entry(t.clone()).or_insert(0) += 1;
            }
        }
        out
    }

    /// Puts an image in the trash; returns whether it was not there already.
    pub fn trash(&mut self, sha: &str, at: &str) -> bool {
        let mut changed = false;
        self.update(sha, |e| {
            if e.trashed.is_none() {
                e.trashed = Some(at.to_owned());
                changed = true;
            }
        });
        changed
    }

    /// Takes an image out of the trash; returns whether it was there.
    pub fn restore(&mut self, sha: &str) -> bool {
        let mut changed = false;
        self.update(sha, |e| changed = e.trashed.take().is_some());
        changed
    }

    pub fn is_trashed(&self, sha: &str) -> bool {
        self.items.get(sha).is_some_and(|e| e.trashed.is_some())
    }

    /// Drops an image's entry: folder, tags and trash.
    pub fn forget(&mut self, sha: &str) {
        self.items.remove(sha);
    }
}

/// The names in a folder path; `/`, empty parts and surrounding spaces are ignored.
pub fn split(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').map(str::trim).filter(|s| !s.is_empty())
}

fn check_name(name: &str) -> Result<()> {
    if name == "." || name == ".." || name.chars().any(char::is_control) {
        bail!("{name:?} cannot be a folder name");
    }
    Ok(())
}

fn check_tag(tag: &str) -> Result<&str> {
    let tag = tag.trim();
    if tag.is_empty() || tag.chars().any(char::is_control) {
        bail!("{tag:?} cannot be a tag");
    }
    Ok(tag)
}

/// 8 lowercase base32 characters from the kernel's random source.
fn random_id() -> String {
    const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut bytes = [0u8; 8];
    if let Ok(mut f) = fs::File::open("/dev/urandom") {
        use std::io::Read;
        let _ = f.read_exact(&mut bytes);
    }
    if bytes == [0; 8] {
        // no /dev/urandom: the clock and the process id are unique enough for a folder id
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        bytes = ((nanos as u64) ^ ((std::process::id() as u64) << 40)).to_le_bytes();
    }
    bytes.iter().map(|b| ALPHABET[(b & 31) as usize] as char).collect()
}

/// `s`, quoted for a shell if it needs to be.
pub fn shell_word(s: &str) -> String {
    let safe = !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || "_-./:=@%+,".contains(c));
    if safe { s.to_owned() } else { format!("'{}'", s.replace('\'', r"'\''")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mkdir_p_makes_the_missing_parents_and_find_walks_the_path() {
        let mut c = Catalog::default();
        let kyoto = c.mkdir("trips/京都", true).unwrap();
        assert_eq!(c.folder_path(&kyoto), "trips/京都");
        assert_eq!(c.find("/trips/京都/"), Some(Some(kyoto)));
        assert_eq!(c.find("/"), Some(None));
        assert_eq!(c.find("trips/osaka"), None);
    }

    #[test]
    fn mkdir_without_p_needs_the_parent_and_refuses_an_existing_folder() {
        let mut c = Catalog::default();
        assert!(c.mkdir("a/b", false).is_err());
        c.mkdir("a", false).unwrap();
        assert!(c.mkdir("a", false).is_err());
        c.mkdir("a", true).unwrap();
    }

    #[test]
    fn a_folder_keeps_its_id_and_its_images_when_renamed_or_moved() {
        let mut c = Catalog::default();
        let b = c.mkdir("a/b", true).unwrap();
        c.mkdir("x", false).unwrap();
        c.set_folder("img", Some(b.clone()));
        c.move_folder("a/b", "x/renamed").unwrap();
        assert_eq!(c.folder_path(&b), "x/renamed");
        assert_eq!(c.entry("img").folder.as_deref(), Some(b.as_str()));
    }

    #[test]
    fn a_folder_cannot_move_into_itself_or_onto_a_taken_name() {
        let mut c = Catalog::default();
        c.mkdir("a/b", true).unwrap();
        c.mkdir("c", false).unwrap();
        assert!(c.move_folder("a", "a/b/a").is_err());
        assert!(c.move_folder("a", "c").is_err());
    }

    #[test]
    fn rmdir_refuses_a_folder_with_images_even_trashed_ones() {
        let mut c = Catalog::default();
        let a = c.mkdir("a", false).unwrap();
        c.set_folder("img", Some(a));
        c.trash("img", "t");
        assert!(c.rmdir("a").is_err());
        c.forget("img");
        c.rmdir("a").unwrap();
        assert!(c.folders.is_empty());
    }

    #[test]
    fn remove_tree_trashes_the_images_and_drops_every_folder_below() {
        let mut c = Catalog::default();
        let a = c.mkdir("a", false).unwrap();
        let b = c.mkdir("a/b", false).unwrap();
        let keep = c.mkdir("keep", false).unwrap();
        c.set_folder("1", Some(a.clone()));
        c.set_folder("2", Some(b.clone()));
        c.add_tag("2", "t").unwrap();
        c.set_folder("3", Some(b));
        c.trash("3", "earlier");
        c.set_folder("4", Some(keep.clone()));
        assert_eq!(c.tree_size(&a), (2, 2), "the trashed one is not counted");
        let (folders, trashed) = c.remove_tree("a", "now").unwrap();
        assert_eq!(folders, 2);
        assert_eq!(trashed, ["1", "2"]);
        assert_eq!(c.folders.keys().collect::<Vec<_>>(), [&keep]);
        assert_eq!(c.entry("2"), Entry { folder: None, tags: vec!["t".into()], trashed: Some("now".into()) });
        assert_eq!(c.entry("3").trashed.as_deref(), Some("earlier"), "already in the trash: left as it was");
        assert!(c.entry("3").folder.is_none());
        assert_eq!(c.entry("4").folder, Some(keep));
        assert!(c.remove_tree("/", "now").is_err());
    }

    #[test]
    fn tags_rename_merges_and_counts_skip_the_trash() {
        let mut c = Catalog::default();
        c.add_tag("1", "海邊").unwrap();
        c.add_tag("2", "海邊").unwrap();
        c.add_tag("2", "beach").unwrap();
        c.add_tag("3", "beach").unwrap();
        c.trash("3", "t");
        assert_eq!(c.rename_tag("海邊", "beach").unwrap(), 2);
        assert_eq!(c.entry("2").tags, ["beach"]);
        assert_eq!(c.tags(), BTreeMap::from([("beach".to_owned(), 2)]));
    }

    #[test]
    fn an_entry_with_nothing_left_is_dropped() {
        let mut c = Catalog::default();
        c.add_tag("img", "x").unwrap();
        c.remove_tag("img", "x");
        assert!(c.items.is_empty());
    }

    #[test]
    fn resolve_names_similar_folders() {
        let mut c = Catalog::default();
        c.mkdir("trips/kyoto", true).unwrap();
        let err = c.resolve("kyoto").unwrap_err().to_string();
        assert!(err.contains("trips/kyoto"), "{err}");
    }
}
