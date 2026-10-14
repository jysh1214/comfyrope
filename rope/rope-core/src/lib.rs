//! rope keeps images by content hash. Each image carries its own metadata: where it came from, the recipe that
//! made it, and a note. One catalog holds the folders, the tags and the trash.
//!
//! On disk, under the library root (by default `~/.local/share/rope`):
//!
//! ```text
//! rope.json            marks the root: {"format": 1, "created": ...}
//! catalog.json         folders, the folder of each image, tags, the trash
//! journal.jsonl        one line per change, only ever appended; the GUI follows it
//! REPO                 where the img2img repo was last seen, for repro.sh
//! .lock                flock held around every metadata change
//! run.lock             img2img.py holds it shared for a whole run; purge takes it exclusive
//! envs/<id>.json       environment snapshots, one per distinct environment
//! items/ab/<sha256>/   one image: <sha12>.<ext> (read-only), meta.json, graph.json, repro.sh, note.txt, uploaded/
//! tmp/                 items being assembled; a complete one is renamed into items/
//! .cache/              thumbnails; safe to delete
//! ```

pub mod atomic;
pub mod catalog;
pub mod check;
pub mod hash;
pub mod journal;
pub mod library;
pub mod lineage;
pub mod meta;
pub mod purge;
pub mod repro;
pub mod store;
pub mod time;

pub use catalog::Catalog;
pub use library::Library;
pub use meta::Meta;
