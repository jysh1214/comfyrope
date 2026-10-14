//! What the GUI shows: a summary of every image, the catalog, and who was made from whom. Kept in step with the
//! library by reading the journal, which every writer appends to.

use anyhow::Result;
use rope_core::store::RAW_SUFFIXES;
use rope_core::{Catalog, Library, Meta, journal, store};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub enum View {
    All,
    Unfiled,
    Folder(String),
    Tag(String),
    Trash,
}

/// The part of an image's meta.json the grid and the lineage need; the details panel reads the whole file.
pub struct Item {
    pub id: String,
    pub name: String,
    pub added: String,
    pub file: PathBuf,
    /// A converted copy to show when `file` cannot be decoded here (camera RAW).
    pub fallback: Option<PathBuf>,
    pub made: bool,
    pub upstream: Vec<String>,
    /// Purged, but kept because images were made from it: shown only in lineages.
    pub kept: bool,
    /// Lower-case name and prompt, for the search box.
    haystack: String,
}

impl Item {
    fn new(lib: &Library, m: &Meta) -> Item {
        let dir = lib.item_dir(&m.sha256);
        let raw = m.file.rsplit('.').next().is_some_and(|e| RAW_SUFFIXES.contains(&e));
        let fallback = raw
            .then(|| m.uploaded.values().find_map(|u| u.get("file").and_then(Value::as_str)))
            .flatten()
            .map(|f| dir.join("uploaded").join(f));
        let prompt = m.made_str("prompt").unwrap_or_default();
        Item {
            id: m.sha256.clone(),
            name: if m.name.is_empty() { m.short().to_owned() } else { m.name.clone() },
            added: m.added.clone(),
            file: dir.join(&m.file),
            fallback,
            made: m.made.is_some(),
            upstream: m.upstream(),
            kept: m.kept.is_some(),
            haystack: format!("{} {} {}", m.name, prompt, m.short()).to_lowercase(),
        }
    }
}

pub struct Model {
    pub items: HashMap<String, Item>,
    pub catalog: Catalog,
    pub children: HashMap<String, Vec<String>>,
    /// Goes up with every change, so views know to recompute.
    pub version: u64,
    offset: u64,
    /// Counted once per change rather than once per frame: there may be 100k images.
    pub counts: HashMap<Option<String>, usize>,
    pub shown: usize,
    pub trashed: usize,
    pub tags: BTreeMap<String, usize>,
    /// Every folder's path and id, sorted by path.
    pub paths: Vec<(String, String)>,
}

impl Model {
    pub fn load(lib: &Library) -> Result<Model> {
        // read the journal's length first: a change made during the scan is applied again, never missed
        let offset = std::fs::metadata(lib.journal_path()).map_or(0, |m| m.len());
        let (metas, _) = store::scan(lib)?;
        let items = metas.iter().map(|m| (m.sha256.clone(), Item::new(lib, m))).collect();
        let mut model = Model {
            items,
            catalog: Catalog::load(lib)?,
            children: HashMap::new(),
            version: 1,
            offset,
            counts: HashMap::new(),
            shown: 0,
            trashed: 0,
            tags: BTreeMap::new(),
            paths: Vec::new(),
        };
        model.link();
        Ok(model)
    }

    /// Applies what the journal recorded since the last call; returns the ids of the images that changed.
    pub fn refresh(&mut self, lib: &Library) -> Result<Vec<String>> {
        let (events, offset) = journal::read_from(lib, self.offset)?;
        self.offset = offset;
        if events.is_empty() {
            return Ok(Vec::new());
        }
        let mut changed = Vec::new();
        for e in &events {
            for id in e["ids"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                match store::load(lib, id) {
                    Ok(m) => {
                        self.items.insert(id.to_owned(), Item::new(lib, &m));
                    }
                    Err(_) if !lib.item_dir(id).exists() => {
                        self.items.remove(id);
                    }
                    Err(_) => {} // half written; the next event brings it in
                }
                changed.push(id.to_owned());
            }
        }
        self.catalog = Catalog::load(lib)?;
        self.link();
        self.version += 1;
        Ok(changed)
    }

    fn link(&mut self) {
        self.count();
        self.children.clear();
        let mut made: Vec<&Item> = self.items.values().filter(|i| !i.upstream.is_empty()).collect();
        made.sort_by(|a, b| (&a.added, &a.id).cmp(&(&b.added, &b.id)));
        for item in made {
            let mut seen = Vec::new();
            for p in &item.upstream {
                if !seen.contains(&p) {
                    seen.push(p);
                    self.children.entry(p.clone()).or_default().push(item.id.clone());
                }
            }
        }
    }

    pub fn children(&self, id: &str) -> &[String] {
        self.children.get(id).map_or(&[], Vec::as_slice)
    }

    fn shown(&self, item: &Item) -> bool {
        !item.kept && !self.catalog.is_trashed(&item.id)
    }

    /// The images of a view, newest first, filtered by `search` (in name, prompt and id).
    pub fn view(&self, view: &View, search: &str) -> Vec<String> {
        let search = search.trim().to_lowercase();
        let folders = match view {
            View::Folder(id) => Some(self.catalog.subtree(id)),
            _ => None,
        };
        let mut items: Vec<&Item> = self
            .items
            .values()
            .filter(|i| {
                let e = self.catalog.items.get(&i.id);
                let in_view = match view {
                    View::All => self.shown(i),
                    View::Unfiled => self.shown(i) && e.is_none_or(|e| e.folder.is_none()),
                    View::Folder(_) => {
                        self.shown(i)
                            && e.and_then(|e| e.folder.as_ref()).is_some_and(|f| folders.as_ref().unwrap().contains(f))
                    }
                    View::Tag(t) => self.shown(i) && e.is_some_and(|e| e.tags.contains(t)),
                    View::Trash => !i.kept && self.catalog.is_trashed(&i.id),
                };
                in_view && (search.is_empty() || i.haystack.contains(&search))
            })
            .collect();
        items.sort_by(|a, b| (&b.added, &b.id).cmp(&(&a.added, &a.id)));
        items.into_iter().map(|i| i.id.clone()).collect()
    }

    /// How many images each folder holds directly (`None`: unfiled) outside the trash, how many are shown and
    /// trashed, the tags, and the folder paths.
    fn count(&mut self) {
        let mut counts = HashMap::new();
        let (mut shown, mut trashed) = (0, 0);
        for i in self.items.values().filter(|i| !i.kept) {
            match self.catalog.items.get(&i.id) {
                Some(e) if e.trashed.is_some() => trashed += 1,
                e => {
                    shown += 1;
                    *counts.entry(e.and_then(|e| e.folder.clone())).or_insert(0) += 1;
                }
            }
        }
        let mut paths: Vec<(String, String)> =
            self.catalog.folders.keys().map(|id| (self.catalog.folder_path(id), id.clone())).collect();
        paths.sort();
        (self.counts, self.shown, self.trashed, self.paths) = (counts, shown, trashed, paths);
        self.tags = self.catalog.tags();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// cargo test --release -p rope-gui -- --ignored, with ROPE_BENCH_LIBRARY naming a large library.
    #[test]
    #[ignore]
    fn bench_a_large_library() {
        let Ok(root) = std::env::var("ROPE_BENCH_LIBRARY") else { return };
        let lib = Library::open(std::path::Path::new(&root)).unwrap();
        let t = std::time::Instant::now();
        let model = Model::load(&lib).unwrap();
        eprintln!("load {} items: {:?}", model.items.len(), t.elapsed());
        let folder = model.catalog.folders.keys().next().unwrap().clone();
        for (name, view) in [
            ("all", View::All),
            ("unfiled", View::Unfiled),
            ("folder", View::Folder(folder)),
            ("tag", View::Tag("海邊".into())),
            ("trash", View::Trash),
        ] {
            let t = std::time::Instant::now();
            let n = model.view(&view, "").len();
            eprintln!("view {name}: {n} images in {:?}", t.elapsed());
        }
        let t = std::time::Instant::now();
        let n = model.view(&View::All, "海邊").len();
        eprintln!("search: {n} images in {:?}", t.elapsed());
    }
}
