//! Playing a clip in the window. The ffmpeg on the system decodes it to raw frames down a pipe; a thread reads
//! them into memory as they come, and the viewer shows the frame the clock calls for, round and round. A clip
//! whose frames would not fit the budget is decoded smaller.

use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The decoded frames of one clip stay under this many bytes (RGBA).
const BUDGET: u64 = 768 << 20;

#[derive(Default)]
struct Shared {
    frames: Mutex<Vec<ColorImage>>,
    /// ffmpeg has said all it will: the frames are all in, or it could not read the clip.
    done: AtomicBool,
    /// Set when the player goes: the thread stops reading and ends ffmpeg.
    stop: AtomicBool,
}

pub struct Player {
    pub id: String,
    shared: Arc<Shared>,
    /// The frames' size: the clip's own, or smaller to fit the budget.
    pub size: [usize; 2],
    pub fps: f64,
    /// How many frames the clip has by its meta.json; the frames themselves settle it once they are all in.
    total: Option<usize>,
    texture: Option<TextureHandle>,
    /// The frame on the texture.
    pub shown: Option<usize>,
    /// When frame 0 was, or would have been, shown: the clock.
    started: Instant,
    /// The frame held while paused.
    pub paused: Option<usize>,
}

impl Player {
    /// Starts decoding `file`, a clip of `width` x `height` pixels with `frames` frames at `fps`, as far as
    /// those are known.
    pub fn new(
        ctx: &egui::Context,
        id: String,
        file: &Path,
        width: u32,
        height: u32,
        fps: Option<f64>,
        frames: Option<u32>,
    ) -> Player {
        let fps = fps.filter(|f| *f > 0.0).unwrap_or(16.0);
        let (w, h) = fitted(width.max(2), height.max(2), frames.unwrap_or(1));
        let size = [w as usize, h as usize];
        let shared = Arc::new(Shared::default());
        let scaled = (w, h) != (width, height);
        std::thread::spawn({
            let (shared, ctx, file) = (shared.clone(), ctx.clone(), file.to_path_buf());
            move || decode(&shared, &file, size, scaled, &ctx)
        });
        Player {
            id,
            shared,
            size,
            fps,
            total: frames.map(|n| n as usize),
            texture: None,
            shown: None,
            started: Instant::now(),
            paused: None,
        }
    }

    /// The frame to show now, as a texture; None until the first frame is in. Asks for a repaint in time for the
    /// next frame while playing.
    pub fn frame(&mut self, ctx: &egui::Context) -> Option<TextureHandle> {
        let done = self.shared.done.load(Ordering::Relaxed);
        let frames = self.shared.frames.lock().unwrap();
        let have = frames.len();
        if have == 0 {
            if !done {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
            return None;
        }
        let total = if done { have } else { self.total.unwrap_or(have).max(have) };
        let idx = match self.paused {
            Some(i) => i.min(have - 1),
            None => {
                let mut i = (self.started.elapsed().as_secs_f64() * self.fps) as usize % total;
                if i >= have {
                    // the clock is ahead of ffmpeg: hold the last frame there is, and put the clock on it
                    i = have - 1;
                    self.started = Instant::now() - Duration::from_secs_f64(i as f64 / self.fps);
                }
                i
            }
        };
        if self.shown != Some(idx) || self.texture.is_none() {
            let image = frames[idx].clone();
            drop(frames);
            match self.texture.as_mut() {
                Some(t) => t.set(image, TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ctx.load_texture(format!("clip {}", self.id), image, TextureOptions::LINEAR));
                }
            }
            self.shown = Some(idx);
        }
        if self.paused.is_none() {
            ctx.request_repaint_after(Duration::from_secs_f64(0.5 / self.fps));
        }
        self.texture.clone()
    }

    /// How many frames ffmpeg has decoded so far.
    pub fn decoded(&self) -> usize {
        self.shared.frames.lock().unwrap().len()
    }

    /// How many frames there are to play through: the clip's, as far as known yet.
    pub fn total(&self) -> usize {
        let have = self.shared.frames.lock().unwrap().len();
        if self.shared.done.load(Ordering::Relaxed) { have } else { self.total.unwrap_or(have).max(have) }
    }

    /// Pauses on the frame shown, or plays on from it.
    pub fn toggle(&mut self) {
        match self.paused.take() {
            Some(i) => self.started = Instant::now() - Duration::from_secs_f64(i as f64 / self.fps),
            None => self.paused = Some(self.shown.unwrap_or(0)),
        }
    }

    /// Goes to frame `i`: held there when paused, played on from there otherwise.
    pub fn seek(&mut self, i: usize) {
        match self.paused {
            Some(_) => self.paused = Some(i),
            None => self.started = Instant::now() - Duration::from_secs_f64(i as f64 / self.fps),
        }
    }

    /// Whether ffmpeg could not read the clip at all.
    pub fn failed(&self) -> bool {
        self.shared.done.load(Ordering::Relaxed) && self.shared.frames.lock().unwrap().is_empty()
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
    }
}

/// The size to decode at: the clip's own, unless its frames would take more than the budget, then as large as
/// fits, with even sides.
fn fitted(width: u32, height: u32, frames: u32) -> (u32, u32) {
    let need = width as u64 * height as u64 * 4 * frames.max(1) as u64;
    if need <= BUDGET {
        return (width, height);
    }
    let f = (BUDGET as f64 / need as f64).sqrt();
    let even = |x: u32| ((x as f64 * f) as u32).max(2) & !1;
    (even(width), even(height))
}

fn decode(shared: &Shared, file: &Path, size: [usize; 2], scaled: bool, ctx: &egui::Context) {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-nostdin", "-i"]).arg(file);
    if scaled {
        cmd.arg("-vf").arg(format!("scale={}:{}", size[0], size[1]));
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"]);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    let Ok(mut child) = cmd.spawn() else {
        shared.done.store(true, Ordering::Relaxed);
        ctx.request_repaint();
        return;
    };
    let mut out = child.stdout.take().expect("piped");
    let mut buf = vec![0u8; size[0] * size[1] * 4];
    while !shared.stop.load(Ordering::Relaxed) && out.read_exact(&mut buf).is_ok() {
        shared.frames.lock().unwrap().push(ColorImage::from_rgba_premultiplied(size, &buf));
        ctx.request_repaint();
    }
    let _ = child.kill(); // still running only when the player went; nothing to do once it has ended
    let _ = child.wait();
    shared.done.store(true, Ordering::Relaxed);
    ctx.request_repaint();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clip_over_the_budget_is_decoded_smaller_with_even_sides() {
        assert_eq!(fitted(640, 480, 81), (640, 480));
        let (w, h) = fitted(3840, 2160, 300);
        assert!(w < 3840 && h < 2160 && w % 2 == 0 && h % 2 == 0);
        assert!((w as u64 * h as u64 * 4 * 300) <= BUDGET);
        assert!((w as f64 / h as f64 - 16.0 / 9.0).abs() < 0.01, "{w}x{h}");
    }
}
