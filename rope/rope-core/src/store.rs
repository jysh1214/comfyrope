//! Bringing images into the library, and changing what is attached to them.
//!
//! A new item is assembled in `tmp/` (image, meta.json, graph.json, repro.sh, note.txt) and renamed into
//! `items/` in one step, under the library lock, so a reader never sees half an item. Bytes already in the
//! library are not stored again: the item gains a source path instead.

use crate::catalog::Catalog;
use crate::hash::{self, short};
use crate::meta::{self, ImageInfo, Meta, Source};
use crate::{Library, atomic, journal, repro, time};
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Suffixes img2img.py treats as camera RAW (its RAW_SUFFIXES). They stay as they are: img2img.py picks the
/// RAW decoder by suffix.
pub const RAW_SUFFIXES: &[&str] = &[
    "dng", "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "raf", "orf", "rw2", "pef", "srw", "x3f", "3fr",
    "iiq", "rwl", "mrw", "erf", "kdc", "dcr", "mos",
];
const IMAGE_SUFFIXES: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "webp", "gif", "bmp", "tif", "tiff", "heic", "heif", "avif", "jxl", "psd", "tga",
    "qoi", "ico", "jp2",
];

/// Whether rope takes a file with this name as an image when walking a folder.
pub fn looks_like_image(path: &Path) -> bool {
    let ext = suffix(path);
    IMAGE_SUFFIXES.contains(&ext.as_str()) || RAW_SUFFIXES.contains(&ext.as_str())
}

/// What to attach to a file being added.
#[derive(Debug, Default, Clone)]
pub struct NewImage {
    /// The display name; the file's stem when not given.
    pub name: Option<String>,
    /// The recipe of an image a tool made.
    pub made: Option<Value>,
    /// The API workflow that made it.
    pub graph: Option<Value>,
    /// What brought it in: "import", "img2img.py", "gui".
    pub via: String,
    /// Put it in this folder (`Some(None)`: unfiled). `None` leaves an existing image where it is.
    pub folder: Option<Option<String>>,
    pub tags: Vec<String>,
    /// Appended to the note.
    pub note: Option<String>,
    /// Also file this image's inputs in `folder`, those that are unfiled.
    pub file_inputs: bool,
}

#[derive(Debug, Clone)]
pub struct Added {
    pub sha: String,
    /// False when the bytes were in the library already.
    pub created: bool,
    /// True when the image was in the trash, or purged and kept for its descendants, and came back.
    pub restored: bool,
}

/// Adds the file at `src`. Bytes the library has already only gain `src` as a source (and whatever else `new`
/// attaches); a trashed image comes out of the trash.
pub fn add(lib: &Library, src: &Path, new: &NewImage) -> Result<Added> {
    let staging = staging_dir(lib)?;
    let result = assemble(lib, src, new, &staging).and_then(|meta| publish(lib, &staging, meta, new));
    if staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn assemble(lib: &Library, src: &Path, new: &NewImage, staging: &Path) -> Result<Meta> {
    let src_abs = std::path::absolute(src)?;
    let tmp = staging.join("image.tmp");
    let (sha, len) = hash::copy_hashed(src, &tmp)?;
    let mut image = image_info(&tmp, len);
    if new.made.is_some() {
        image.pixels = pixel_hash(&tmp).ok();
    }
    let file = format!("{}.{}", short(&sha), extension(src, image.format.as_deref()));
    fs::rename(&tmp, staging.join(&file))?;
    set_readonly(&staging.join(&file))?;
    let now = time::now();
    let sources = if inside(lib, &src_abs) {
        Vec::new()
    } else {
        vec![Source { path: src_abs.display().to_string(), at: now.clone(), via: new.via.clone() }]
    };
    let meta = Meta {
        format: meta::FORMAT,
        sha256: sha,
        file,
        name: new.name.clone().unwrap_or_else(|| stem(src)),
        added: now,
        image,
        sources,
        made: new.made.clone(),
        uploaded: Map::new(),
        replays: Vec::new(),
        kept: None,
        other: Map::new(),
    };
    write_recipe_files(staging, &meta, new.graph.as_ref())?;
    if let Some(note) = new.note.as_deref().filter(|n| !n.trim().is_empty()) {
        fs::write(staging.join("note.txt"), with_newline(note))?;
    }
    fs::write(staging.join("meta.json"), meta.to_json())?;
    Ok(meta)
}

/// graph.json and repro.sh, for an image with a recipe.
fn write_recipe_files(dir: &Path, meta: &Meta, graph: Option<&Value>) -> Result<()> {
    if let Some(graph) = graph {
        let mut bytes = serde_json::to_vec_pretty(graph)?;
        bytes.push(b'\n');
        atomic::write(&dir.join("graph.json"), &bytes)?;
    }
    if meta.made.is_some() {
        let path = dir.join("repro.sh");
        atomic::write(&path, repro::script(meta).as_bytes())?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}

fn publish(lib: &Library, staging: &Path, meta: Meta, new: &NewImage) -> Result<Added> {
    let _lock = lib.lock()?;
    let sha = meta.sha256.clone();
    let dest = lib.item_dir(&sha);
    let mut restored = false;
    let created = if dest.join("meta.json").exists() {
        let mut old = Meta::load(&dest.join("meta.json"))?;
        let mut changed = false;
        for s in &meta.sources {
            if !old.sources.iter().any(|o| o.path == s.path) {
                old.sources.push(s.clone());
                changed = true;
            }
        }
        if old.kept.take().is_some() {
            if old.name.is_empty() {
                old.name = meta.name.clone();
            }
            restored = true;
            changed = true;
        }
        if old.made.is_none() && meta.made.is_some() {
            old.made = meta.made.clone();
            old.image.pixels = meta.image.pixels.clone();
            write_recipe_files(&dest, &old, new.graph.as_ref())?;
            changed = true;
        }
        if changed {
            atomic::write(&dest.join("meta.json"), &old.to_json())?;
        }
        if let Some(note) = new.note.as_deref().filter(|n| !n.trim().is_empty()) {
            append_note(&dest, note)?;
        }
        false
    } else {
        let parent = dest.parent().expect("item dirs have a parent");
        fs::create_dir_all(parent)?;
        fs::rename(staging, &dest).with_context(|| format!("cannot move the new item into {}", dest.display()))?;
        atomic::sync_dir(parent)?;
        true
    };

    let mut catalog = Catalog::load(lib)?;
    let before = catalog.items.clone();
    restored |= catalog.restore(&sha);
    if let Some(folder) = &new.folder {
        catalog.set_folder(&sha, folder.clone());
        if new.file_inputs
            && folder.is_some()
            && let Ok(m) = Meta::load(&dest.join("meta.json"))
        {
            for input in m.upstream() {
                let e = catalog.entry(&input);
                if e.folder.is_none() && e.trashed.is_none() {
                    catalog.set_folder(&input, folder.clone());
                }
            }
        }
    }
    for tag in &new.tags {
        catalog.add_tag(&sha, tag)?;
    }
    if catalog.items != before {
        catalog.save(lib)?;
    }
    journal::append(lib, if created { "add" } else { "update" }, std::slice::from_ref(&sha))?;
    Ok(Added { sha, created, restored })
}

/// Stores `src`, a file uploaded in place of image `sha` (a HEIC or RAW photo converted to PNG), as
/// `uploaded/<its sha256>.<ext>` in that image's item, with `info` on how it was made. Returns its SHA-256 and
/// where it is.
pub fn add_rendition(lib: &Library, sha: &str, src: &Path, mut info: Value) -> Result<(String, PathBuf)> {
    let dir = lib.item_dir(sha);
    if !dir.join("meta.json").exists() {
        bail!("no image {} in rope", short(sha));
    }
    let up = dir.join("uploaded");
    fs::create_dir_all(&up)?;
    let tmp = atomic::temp_path(&up.join("rendition"));
    let (r_sha, len) = hash::copy_hashed(src, &tmp)?;
    let file = format!("{r_sha}.{}", extension(src, None));
    let _lock = lib.lock()?;
    let dest = up.join(&file);
    if dest.exists() {
        fs::remove_file(&tmp)?;
    } else {
        fs::rename(&tmp, &dest)?;
        set_readonly(&dest)?;
    }
    let mut meta = Meta::load(&dir.join("meta.json"))?;
    if !meta.uploaded.contains_key(&r_sha) {
        if let Value::Object(m) = &mut info {
            m.insert("file".into(), Value::from(file));
            m.insert("bytes".into(), Value::from(len));
        }
        meta.uploaded.insert(r_sha.clone(), info);
        atomic::write(&dir.join("meta.json"), &meta.to_json())?;
        journal::append(lib, "update", &[sha.to_owned()])?;
    }
    Ok((r_sha, dest))
}

/// Stores an environment snapshot in `envs/`, once per distinct content, and returns its id.
pub fn add_env(lib: &Library, env: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(&canonical(env))?;
    let id = short(&hash::sha256_bytes(&bytes)).to_owned();
    let path = lib.envs_dir().join(format!("{id}.json"));
    if !path.exists() {
        fs::create_dir_all(lib.envs_dir())?;
        let mut pretty = serde_json::to_vec_pretty(env)?;
        pretty.push(b'\n');
        atomic::write(&path, &pretty)?;
    }
    Ok(id)
}

/// `v` with every object's keys sorted, so equal content hashes equally.
fn canonical(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<_> = m.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), canonical(&m[k]))).collect())
        }
        Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
        _ => v.clone(),
    }
}

// ---------------------------------------------------------------- folders of images

/// Every image under the disk folder `dir`, with the rope folder that mirrors where it sits: a file in `dir/a/b`
/// goes to `<parent>/<dir's name>/a/b`, and `parent` "" is the top. Hidden files and folders are skipped, and a
/// folder without images gets no rope folder, since only the images' folders are named.
pub fn mirror(dir: &Path, parent: &str) -> Vec<(PathBuf, String)> {
    let name = fs::canonicalize(dir)
        .ok()
        .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "import".into());
    let mut out = Vec::new();
    mirror_into(dir, &join(parent, &name), &mut out);
    out
}

fn mirror_into(dir: &Path, folder: &str, out: &mut Vec<(PathBuf, String)>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if name.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            mirror_into(&p, &join(folder, &name), out);
        } else if looks_like_image(&p) {
            out.push((p, folder.to_owned()));
        }
    }
}

fn join(parent: &str, name: &str) -> String {
    let parent = parent.trim_matches('/');
    if parent.is_empty() { name.to_owned() } else { format!("{parent}/{name}") }
}

/// Adds each file into the rope folder named with it (a path like `trips/kyoto`), making the folders that are
/// missing; `new` gives everything else. Returns what became of each file.
pub fn add_to_folders(lib: &Library, files: &[(PathBuf, String)], new: &NewImage) -> Vec<(PathBuf, Result<Added>)> {
    let mut folders: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut out = Vec::new();
    for (file, folder) in files {
        let id = match folders.get(folder) {
            Some(id) => Ok(id.clone()),
            None => edit_catalog(lib, "mkdir", &[], |c| c.mkdir(folder, true)),
        };
        let result = id.and_then(|id| {
            folders.insert(folder.clone(), id.clone());
            add(lib, file, &NewImage { folder: Some(Some(id)), ..new.clone() })
        });
        out.push((file.clone(), result));
    }
    out
}

// ---------------------------------------------------------------- reading

pub fn load(lib: &Library, sha: &str) -> Result<Meta> {
    Meta::load(&lib.item_dir(sha).join("meta.json"))
}

/// The image file of an item.
pub fn image_path(lib: &Library, meta: &Meta) -> PathBuf {
    lib.item_dir(&meta.sha256).join(&meta.file)
}

/// The ids of every item, from the directory listing alone.
pub fn ids(lib: &Library) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let items = lib.items_dir();
    if !items.exists() {
        return Ok(out);
    }
    for shard in fs::read_dir(&items)? {
        let shard = shard?;
        if !shard.file_type()?.is_dir() {
            continue;
        }
        for item in fs::read_dir(shard.path())? {
            let name = item?.file_name().to_string_lossy().into_owned();
            if name.len() == 64 && hash::is_hex(&name) {
                out.push(name);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Every item's metadata, and a message for each meta.json that could not be read.
pub fn scan(lib: &Library) -> Result<(Vec<Meta>, Vec<String>)> {
    let mut metas = Vec::new();
    let mut problems = Vec::new();
    for id in ids(lib)? {
        match load(lib, &id) {
            Ok(m) => metas.push(m),
            Err(e) => problems.push(format!("{}: {e:#}", short(&id))),
        }
    }
    Ok((metas, problems))
}

/// The id named by `text`: an id or a unique prefix of at least 6 hex digits, optionally after `@`; a file in
/// the library; or any file whose bytes the library has.
pub fn resolve(lib: &Library, text: &str) -> Result<String> {
    let bare = text.strip_prefix('@').unwrap_or(text);
    let lower = bare.to_ascii_lowercase();
    if hash::is_hex(&lower) && lower.len() <= 64 && (lower.len() >= 6 || text.starts_with('@')) {
        if lower.len() < 6 {
            bail!("{text}: give at least 6 hex digits of the id");
        }
        let shard = lib.items_dir().join(&lower[..2]);
        let mut found: Vec<String> = match fs::read_dir(&shard) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with(&lower) && n.len() == 64)
                .collect(),
            Err(_) => Vec::new(),
        };
        found.sort();
        match found.len() {
            1 => return Ok(found.remove(0)),
            0 if text.starts_with('@') || !Path::new(text).exists() => bail!("no image {bare} in rope"),
            0 => {}
            n => bail!(
                "{bare} matches {n} images ({}...); give more digits",
                found.iter().take(4).map(|s| short(s)).collect::<Vec<_>>().join(", ")
            ),
        }
    }
    let path = Path::new(text);
    if !path.exists() {
        bail!("no image {text:?}: give an id (at least 6 hex digits, or @id) or a file");
    }
    let canon = fs::canonicalize(path)?;
    if let Ok(items) = fs::canonicalize(lib.items_dir())
        && let Ok(rest) = canon.strip_prefix(&items)
        && let Some(sha) = rest.iter().nth(1).and_then(|s| s.to_str())
        && lib.item_dir(sha).join("meta.json").exists()
    {
        return Ok(sha.to_owned());
    }
    if canon.is_dir() {
        bail!("{} is a folder, not an image", path.display());
    }
    let sha = hash::sha256_file(&canon)?;
    if lib.item_dir(&sha).join("meta.json").exists() {
        return Ok(sha);
    }
    bail!(
        "{} is not in rope; bring it in with: rope import {}",
        path.display(),
        crate::catalog::shell_word(&path.display().to_string())
    )
}

// ---------------------------------------------------------------- changing an item

/// The note, or an empty string.
pub fn note(lib: &Library, sha: &str) -> Result<String> {
    match fs::read_to_string(lib.item_dir(sha).join("note.txt")) {
        Ok(s) => Ok(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(e.into()),
    }
}

/// Replaces the note; an empty one removes note.txt.
pub fn set_note(lib: &Library, sha: &str, text: &str) -> Result<()> {
    let dir = existing_item(lib, sha)?;
    let _lock = lib.lock()?;
    let path = dir.join("note.txt");
    if text.trim().is_empty() {
        if path.exists() {
            fs::remove_file(&path)?;
        }
    } else {
        atomic::write(&path, with_newline(text).as_bytes())?;
    }
    journal::append(lib, "note", &[sha.to_owned()])
}

/// Adds `text` as a new paragraph of the note.
pub fn add_note(lib: &Library, sha: &str, text: &str) -> Result<()> {
    let dir = existing_item(lib, sha)?;
    let _lock = lib.lock()?;
    append_note(&dir, text)?;
    journal::append(lib, "note", &[sha.to_owned()])
}

fn append_note(dir: &Path, text: &str) -> Result<()> {
    let path = dir.join("note.txt");
    let old = fs::read_to_string(&path).unwrap_or_default();
    if old.trim_end().ends_with(text.trim_end()) && !old.is_empty() {
        return Ok(()); // the same note again, e.g. one img2img.py --note run that made these bytes twice
    }
    let new = if old.trim().is_empty() {
        with_newline(text)
    } else {
        format!("{}\n{}", with_newline(&old), with_newline(text))
    };
    atomic::write(&path, new.as_bytes())
}

pub fn set_name(lib: &Library, sha: &str, name: &str) -> Result<()> {
    let dir = existing_item(lib, sha)?;
    let name = name.trim();
    if name.is_empty() {
        bail!("a name cannot be empty");
    }
    let _lock = lib.lock()?;
    let mut meta = Meta::load(&dir.join("meta.json"))?;
    meta.name = name.to_owned();
    atomic::write(&dir.join("meta.json"), &meta.to_json())?;
    journal::append(lib, "name", &[sha.to_owned()])
}

/// Adds a replay record to image `sha`.
pub fn add_replay(lib: &Library, sha: &str, record: Value) -> Result<()> {
    let dir = existing_item(lib, sha)?;
    let _lock = lib.lock()?;
    let mut meta = Meta::load(&dir.join("meta.json"))?;
    meta.replays.push(record);
    atomic::write(&dir.join("meta.json"), &meta.to_json())?;
    journal::append(lib, "replay", &[sha.to_owned()])
}

/// Changes the catalog under the library lock and journals it as `op` on `ids`.
pub fn edit_catalog<T>(
    lib: &Library,
    op: &str,
    ids: &[String],
    change: impl FnOnce(&mut Catalog) -> Result<T>,
) -> Result<T> {
    let _lock = lib.lock()?;
    let mut catalog = Catalog::load(lib)?;
    let out = change(&mut catalog)?;
    catalog.save(lib)?;
    journal::append(lib, op, ids)?;
    Ok(out)
}

/// Removes the folder at `path` with everything below it, moving its images to the trash; returns how many folders
/// went and the images newly trashed (see Catalog::remove_tree).
pub fn remove_folder_tree(lib: &Library, path: &str) -> Result<(usize, Vec<String>)> {
    let _lock = lib.lock()?;
    let mut catalog = Catalog::load(lib)?;
    let out = catalog.remove_tree(path, &time::now())?;
    catalog.save(lib)?;
    journal::append(lib, "rmdir", &out.1)?;
    Ok(out)
}

fn existing_item(lib: &Library, sha: &str) -> Result<PathBuf> {
    let dir = lib.item_dir(sha);
    if !dir.join("meta.json").exists() {
        bail!("no image {} in rope", short(sha));
    }
    Ok(dir)
}

/// Writes repro.sh again for every image with a recipe; returns how many.
pub fn rewrite_repro_scripts(lib: &Library) -> Result<usize> {
    let (metas, _) = scan(lib)?;
    let mut n = 0;
    for m in metas.iter().filter(|m| m.made.is_some() && m.kept.is_none()) {
        let path = lib.item_dir(&m.sha256).join("repro.sh");
        let script = repro::script(m);
        if fs::read_to_string(&path).ok().as_deref() != Some(script.as_str()) {
            atomic::write(&path, script.as_bytes())?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
            n += 1;
        }
    }
    Ok(n)
}

// ---------------------------------------------------------------- pixels

/// SHA-256 over the width and height (big-endian u32) and the RGBA8 pixels. Two files with the same pixels hash
/// the same however they were saved.
pub fn pixel_hash(path: &Path) -> Result<String> {
    let img = decode(path)?;
    let rgba = img.to_rgba8();
    let mut h = Sha256::new();
    h.update(rgba.width().to_be_bytes());
    h.update(rgba.height().to_be_bytes());
    h.update(rgba.as_raw());
    Ok(hash::hex(&h.finalize()))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub identical: bool,
    pub same_size: bool,
    /// The largest difference of any channel of any pixel, 0..=255.
    pub max_diff: u8,
    /// The share of pixels that differ in any channel, 0..=1.
    pub differing: f64,
}

pub fn compare_pixels(a: &Path, b: &Path) -> Result<Comparison> {
    let (a, b) = (decode(a)?.to_rgba8(), decode(b)?.to_rgba8());
    if a.dimensions() != b.dimensions() {
        return Ok(Comparison { identical: false, same_size: false, max_diff: 255, differing: 1.0 });
    }
    let mut max_diff = 0u8;
    let mut differing = 0usize;
    for (p, q) in a.pixels().zip(b.pixels()) {
        let d = p.0.iter().zip(q.0.iter()).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
        max_diff = max_diff.max(d);
        differing += usize::from(d > 0);
    }
    let total = (a.width() as usize * a.height() as usize).max(1);
    Ok(Comparison { identical: differing == 0, same_size: true, max_diff, differing: differing as f64 / total as f64 })
}

fn decode(path: &Path) -> Result<image::DynamicImage> {
    image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))
}

// ---------------------------------------------------------------- helpers

static STAGING: AtomicU64 = AtomicU64::new(0);

fn staging_dir(lib: &Library) -> Result<PathBuf> {
    let n = STAGING.fetch_add(1, Ordering::Relaxed);
    let dir = lib.tmp_dir().join(format!("{}.{n}", std::process::id()));
    fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    Ok(dir)
}

fn image_info(path: &Path, bytes: u64) -> ImageInfo {
    let size = imagesize::size(path).ok();
    let format = fs::File::open(path)
        .ok()
        .and_then(|mut f| {
            use std::io::Read;
            let mut head = vec![0u8; 64];
            let n = f.read(&mut head).ok()?;
            imagesize::image_type(&head[..n]).ok()
        })
        .map(format_name);
    ImageInfo {
        width: size.map(|s| s.width as u32),
        height: size.map(|s| s.height as u32),
        bytes,
        format,
        pixels: None,
    }
}

fn format_name(t: imagesize::ImageType) -> String {
    let debug = format!("{t:?}");
    let base = debug.split('(').next().unwrap_or(&debug);
    base.to_ascii_lowercase()
}

/// The file's suffix in lower case; without a usable one, the detected format's usual suffix.
fn extension(src: &Path, format: Option<&str>) -> String {
    let ext = suffix(src);
    if !ext.is_empty() && ext.len() <= 5 && ext.chars().all(|c| c.is_ascii_alphanumeric()) {
        return ext;
    }
    match format {
        Some("jpeg") => "jpg".into(),
        Some("heif") => "heic".into(),
        Some(f) if !f.is_empty() && f.len() <= 5 && f.chars().all(|c| c.is_ascii_alphanumeric()) => f.into(),
        _ => "bin".into(),
    }
}

fn suffix(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default()
}

fn stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into())
}

fn inside(lib: &Library, path: &Path) -> bool {
    match (fs::canonicalize(path), fs::canonicalize(lib.root())) {
        (Ok(p), Ok(root)) => p.starts_with(root),
        _ => false,
    }
}

fn set_readonly(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o444))
        .with_context(|| format!("cannot chmod {}", path.display()))
}

fn with_newline(s: &str) -> String {
    let t = s.trim_end_matches('\n');
    format!("{t}\n")
}
