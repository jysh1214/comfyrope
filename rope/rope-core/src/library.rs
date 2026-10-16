//! Where a library lives, its paths, and its locks.

use crate::{atomic, time};
use anyhow::{Context, Result, bail};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

/// The on-disk format this build reads and writes. A library made by a newer rope is refused, not guessed at.
pub const FORMAT: u32 = 1;
const MARKER: &str = "rope.json";

#[derive(Debug, Clone)]
pub struct Library {
    root: PathBuf,
}

/// An flock, released when this is dropped.
#[derive(Debug)]
pub struct Lock {
    _file: File,
}

impl Library {
    /// `$XDG_DATA_HOME/rope`, or `~/.local/share/rope`.
    pub fn default_root() -> Result<PathBuf> {
        if let Some(dir) = std::env::var_os("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
            return Ok(PathBuf::from(dir).join("rope"));
        }
        let home = std::env::var_os("HOME").context("HOME is not set")?;
        Ok(PathBuf::from(home).join(".local/share/rope"))
    }

    /// The library to use: `explicit` (a --library flag), else `$ROPE_LIBRARY`, else the default location.
    ///
    /// A library named explicitly must exist already; `rope init DIR` makes one. Making it on the spot would
    /// split the images between two libraries, say when a data disk is not mounted. The default one is made on
    /// first use.
    pub fn locate(explicit: Option<&Path>) -> Result<Library> {
        let named = explicit
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("ROPE_LIBRARY").filter(|v| !v.is_empty()).map(PathBuf::from));
        match named {
            Some(root) => Library::open(&root),
            None => {
                let root = Self::default_root()?;
                if root.join(MARKER).exists() { Library::open(&root) } else { Library::init(&root) }
            }
        }
    }

    pub fn open(root: &Path) -> Result<Library> {
        let marker = root.join(MARKER);
        let text = fs::read_to_string(&marker).with_context(|| {
            format!(
                "{} is not a rope library (it has no {MARKER}); make one with: rope init {}",
                root.display(),
                root.display()
            )
        })?;
        let v: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("{} is not valid JSON", marker.display()))?;
        let format = v.get("format").and_then(|f| f.as_u64()).unwrap_or(0);
        if format > FORMAT as u64 {
            bail!("{} has format {format}; this rope reads up to {FORMAT}, so update rope", root.display());
        }
        Ok(Library { root: std::path::absolute(root)? })
    }

    /// Makes a new, empty library at `root`, which must be missing or an empty directory.
    pub fn init(root: &Path) -> Result<Library> {
        if root.join(MARKER).exists() {
            return Library::open(root);
        }
        if root.exists() && fs::read_dir(root)?.next().is_some() {
            bail!("{} exists and is not empty; rope makes a library only in an empty or new directory", root.display());
        }
        for dir in ["items", "tmp", "envs", "runs", ".cache"] {
            fs::create_dir_all(root.join(dir)).with_context(|| format!("cannot create {}", root.display()))?;
        }
        let marker = serde_json::json!({"format": FORMAT, "created": time::now()});
        atomic::write(&root.join(MARKER), format!("{marker:#}\n").as_bytes())?;
        Library::open(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether this is the default library, which commands then need not name.
    pub fn is_default(&self) -> bool {
        Self::default_root().is_ok_and(|d| std::path::absolute(d).is_ok_and(|d| d == self.root))
    }

    pub fn items_dir(&self) -> PathBuf {
        self.root.join("items")
    }

    /// `items/<first 2 hex digits>/<sha256>`.
    pub fn item_dir(&self, sha: &str) -> PathBuf {
        self.items_dir().join(&sha[..2]).join(sha)
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }

    pub fn runs_dir(&self) -> PathBuf {
        self.root.join("runs")
    }

    pub fn envs_dir(&self) -> PathBuf {
        self.root.join("envs")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.root.join(".cache")
    }

    pub fn catalog_path(&self) -> PathBuf {
        self.root.join("catalog.json")
    }

    pub fn prompts_path(&self) -> PathBuf {
        self.root.join("prompts.json")
    }

    pub fn journal_path(&self) -> PathBuf {
        self.root.join("journal.jsonl")
    }

    /// Takes the lock every metadata change holds. Each call opens the lock file anew: flock belongs to an open
    /// file, so threads sharing one descriptor would not exclude each other.
    pub fn lock(&self) -> Result<Lock> {
        let file = self.lock_file(".lock")?;
        file.lock().context("cannot lock the library")?;
        Ok(Lock { _file: file })
    }

    /// The run lock, exclusive and without waiting: `None` while an img2img.py run holds it shared, which it does
    /// from bringing its inputs in to recording them in `runs/` (see `runs`). Purge holds it, so that no run can
    /// slip its inputs in between purge reading `runs/` and deleting.
    pub fn try_run_lock(&self) -> Result<Option<Lock>> {
        let file = self.lock_file("run.lock")?;
        Ok(try_lock(&file).context("cannot lock run.lock")?.then_some(Lock { _file: file }))
    }

    pub(crate) fn lock_file(&self, name: &str) -> Result<File> {
        let path = self.root.join(name);
        OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .with_context(|| format!("cannot open {}", path.display()))
    }

    /// Where the comfyrope repo was last seen, as recorded by img2img.py through `set_repo`.
    pub fn repo(&self) -> Option<PathBuf> {
        let text = fs::read_to_string(self.root.join("REPO")).ok()?;
        let line = text.trim();
        (!line.is_empty()).then(|| PathBuf::from(line))
    }

    pub fn set_repo(&self, repo: &Path) -> Result<()> {
        let repo = std::path::absolute(repo)?;
        if self.repo().as_deref() != Some(repo.as_path()) {
            atomic::write(&self.root.join("REPO"), format!("{}\n", repo.display()).as_bytes())?;
        }
        Ok(())
    }
}

/// Takes an exclusive flock on `file` without waiting for a holder: `false` when someone holds it. A lock just
/// let go can look held for an instant: a process of this user between fork and exec carries a copy of every
/// open file, locks included, until the exec closes them. So a few tries over about a tenth of a second are
/// made before the lock counts as held.
pub(crate) fn try_lock(file: &File) -> std::io::Result<bool> {
    for attempt in 0..6 {
        match file.try_lock() {
            Ok(()) => return Ok(true),
            Err(TryLockError::WouldBlock) if attempt < 5 => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Error(e)) => return Err(e),
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_makes_a_library_that_open_accepts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("lib");
        Library::init(&root).unwrap();
        assert!(root.join("items").is_dir());
        Library::open(&root).unwrap();
    }

    #[test]
    fn open_refuses_a_directory_that_is_not_a_library() {
        let dir = tempfile::tempdir().unwrap();
        let err = Library::open(dir.path()).unwrap_err().to_string();
        assert!(err.contains("rope init"), "{err}");
    }

    #[test]
    fn init_refuses_a_directory_with_other_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("x"), b"").unwrap();
        assert!(Library::init(dir.path()).is_err());
    }

    #[test]
    fn the_run_lock_is_refused_while_someone_holds_it_shared() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(dir.path()).unwrap();
        let shared = File::open(dir.path().join("run.lock"))
            .or_else(|_| {
                fs::write(dir.path().join("run.lock"), b"")?;
                File::open(dir.path().join("run.lock"))
            })
            .unwrap();
        shared.lock_shared().unwrap();
        assert!(lib.try_run_lock().unwrap().is_none());
        drop(shared);
        assert!(lib.try_run_lock().unwrap().is_some());
    }
}
