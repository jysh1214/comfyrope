//! Thumbnails and previews, decoded by worker threads and kept as textures for the most recently drawn images.
//!
//! Grid thumbnails are also saved as JPEG in the library's .cache/thumbs, so the next start does not decode
//! every photo again. Requests are served newest first: what is on screen now matters more than what was on
//! screen before a scroll.

use crate::decode;
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

/// Grid thumbnails, cached on disk.
pub const GRID: u32 = 256;
/// The preview in the details panel.
pub const PREVIEW: u32 = 1024;
/// The large view: enough for a 4K screen, and within every GPU's texture size limit.
pub const VIEW: u32 = 4096;
const KEEP: usize = 600;
/// How many previews and large views stay on the GPU. A large view takes up to 64 MB, and ComfyUI wants the
/// GPU's memory more.
const KEEP_PREVIEW: usize = 24;
const KEEP_VIEW: usize = 4;
const QUEUE: usize = 300;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Key {
    pub id: String,
    pub size: u32,
}

/// What to decode: the image file, and a file to fall back on when it cannot be decoded here (a camera RAW
/// photo's converted PNG).
#[derive(Clone)]
pub struct Source {
    pub file: PathBuf,
    pub fallback: Option<PathBuf>,
}

struct Request {
    key: Key,
    source: Source,
}

struct Done {
    key: Key,
    image: Option<ColorImage>,
}

type Queue = Arc<(Mutex<VecDeque<Request>>, Condvar)>;

pub struct Thumbs {
    textures: HashMap<Key, (TextureHandle, u64)>,
    pending: HashSet<Key>,
    failed: HashSet<Key>,
    queue: Queue,
    done: Receiver<Done>,
    frame: u64,
}

impl Thumbs {
    pub fn new(ctx: &egui::Context, cache: PathBuf) -> Thumbs {
        let queue: Queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (tx, done) = channel();
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 4).clamp(2, 4));
        for _ in 0..workers {
            let (queue, tx, ctx, cache) = (queue.clone(), tx.clone(), ctx.clone(), cache.clone());
            std::thread::spawn(move || work(queue, tx, ctx, cache));
        }
        Thumbs { textures: HashMap::new(), pending: HashSet::new(), failed: HashSet::new(), queue, done, frame: 0 }
    }

    /// The texture for `key` if it is ready; otherwise asks for it and returns None.
    pub fn get(&mut self, key: &Key, source: impl FnOnce() -> Source) -> Option<TextureHandle> {
        if let Some((tex, used)) = self.textures.get_mut(key) {
            *used = self.frame;
            return Some(tex.clone());
        }
        if !self.pending.contains(key) && !self.failed.contains(key) {
            self.pending.insert(key.clone());
            let (lock, cv) = &*self.queue;
            let mut q = lock.lock().unwrap();
            q.push_back(Request { key: key.clone(), source: source() });
            while q.len() > QUEUE {
                if let Some(old) = q.pop_front() {
                    self.pending.remove(&old.key);
                }
            }
            cv.notify_one();
        }
        None
    }

    #[cfg(test)]
    pub fn ready(&self, key: &Key) -> bool {
        self.textures.contains_key(key)
    }

    pub fn failed(&self, key: &Key) -> bool {
        self.failed.contains(key)
    }

    /// Takes in what the workers finished, and drops the textures not drawn for the longest time.
    pub fn poll(&mut self, ctx: &egui::Context) {
        self.frame += 1;
        while let Ok(d) = self.done.try_recv() {
            self.pending.remove(&d.key);
            match d.image {
                Some(img) => {
                    let name = format!("{}@{}", d.key.id, d.key.size);
                    let tex = ctx.load_texture(name, img, TextureOptions::LINEAR);
                    self.textures.insert(d.key, (tex, self.frame));
                }
                None => {
                    self.failed.insert(d.key);
                }
            }
        }
        for (size, keep) in [(None, KEEP), (Some(PREVIEW), KEEP_PREVIEW), (Some(VIEW), KEEP_VIEW)] {
            let mut by_age: Vec<(u64, Key)> = self
                .textures
                .iter()
                .filter(|(k, _)| size.is_none_or(|s| k.size == s))
                .map(|(k, (_, used))| (*used, k.clone()))
                .collect();
            if by_age.len() > keep {
                by_age.sort_by_key(|(used, _)| *used);
                let drop = by_age.len() - keep;
                for (_, k) in by_age.into_iter().take(drop) {
                    self.textures.remove(&k);
                }
            }
        }
    }
}

fn work(queue: Queue, tx: Sender<Done>, ctx: egui::Context, cache: PathBuf) {
    loop {
        let req = {
            let (lock, cv) = &*queue;
            let mut q = lock.lock().unwrap();
            loop {
                if let Some(r) = q.pop_back() {
                    break r;
                }
                q = cv.wait(q).unwrap();
            }
        };
        let image = load(&req, &cache);
        if tx.send(Done { key: req.key, image }).is_err() {
            return;
        }
        ctx.request_repaint();
    }
}

fn load(req: &Request, cache: &std::path::Path) -> Option<ColorImage> {
    let id = &req.key.id;
    let cached = (req.key.size == GRID).then(|| cache.join("thumbs").join(&id[..2]).join(format!("{id}.jpg")));
    if let Some(c) = cached.as_ref().filter(|c| c.exists())
        && let Ok(img) = image::open(c)
    {
        return Some(decode::fit(&img, GRID));
    }
    let img =
        decode::open(&req.source.file).or_else(|e| req.source.fallback.as_deref().map_or(Err(e), decode::open)).ok()?;
    // scaled down only: an image smaller than the size asked for stays as it is
    let max = req.key.size;
    let small = if img.width() > max || img.height() > max { img.thumbnail(max, max) } else { img };
    if let Some(c) = cached
        && c.parent().is_some_and(|p| std::fs::create_dir_all(p).is_ok())
    {
        let tmp = c.with_extension("tmp.jpg");
        if small.to_rgb8().save(&tmp).is_ok() {
            let _ = std::fs::rename(&tmp, &c);
        }
    }
    Some(decode::fit(&small, req.key.size))
}
