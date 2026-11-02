//! Playing a clip in the window. The ffmpeg on the system decodes it to raw frames down a pipe; a thread reads
//! them into memory as they come, and the viewer shows the frame the clock calls for, round and round. A clip
//! whose frames would not fit the budget is decoded smaller. A second ffmpeg decodes the clip's sound, if it has
//! any, into samples at the output device's rate; an audio thread plays them in step with the same clock,
//! through cpal (ALSA, so PipeWire or PulseAudio on Linux).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The decoded frames of one clip stay under this many bytes (RGBA).
const BUDGET: u64 = 768 << 20;

#[derive(Default)]
struct Shared {
    frames: Mutex<Vec<ColorImage>>,
    /// ffmpeg has said all it will: the frames are all in, or it could not read the clip.
    done: AtomicBool,
    /// Set when the player goes: the threads stop reading and end their ffmpeg.
    stop: AtomicBool,
}

/// The clip's sound, decoded, and what the audio thread needs to play it in step.
struct Sound {
    /// Interleaved samples at the device's rate and channel count, as ffmpeg decodes them.
    samples: Mutex<Vec<f32>>,
    /// ffmpeg has said all it will: the samples are all in, or the clip has no sound.
    done: AtomicBool,
    /// The gain, as f32 bits: the volume, or 0 while muted.
    gain: AtomicU32,
    /// The clip's clock, as the player last set it.
    clock: Mutex<Clock>,
}

/// Where a clip's clock is, for the audio thread: when frame 0 was, or None while held on a frame, and how long
/// one time round the clip takes.
#[derive(Clone, Copy, Default)]
struct Clock {
    started: Option<Instant>,
    period: f64,
}

/// The audio thread resyncs to the clock once it is this far off, in seconds: a seek, a pause, the clip going
/// round again. Within it the samples run on, so that nothing clicks.
const SLACK: f64 = 0.05;

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
    sound: Arc<Sound>,
    /// The output device and its format, until the stream is opened; None without a device or once it is open.
    device: Option<(cpal::Device, cpal::StreamConfig)>,
    /// The audio stream, open once the first samples are in; it plays as long as it lives.
    stream: Option<cpal::Stream>,
}

impl Player {
    /// Starts decoding `file`, a clip of `size` (width, height) pixels with `frames` frames at `fps`, as far as
    /// those are known, and its sound for the output device, to play at `gain` (0 for muted).
    pub fn new(
        ctx: &egui::Context,
        id: String,
        file: &Path,
        size: (u32, u32),
        fps: Option<f64>,
        frames: Option<u32>,
        gain: f32,
    ) -> Player {
        let (width, height) = size;
        let fps = fps.filter(|f| *f > 0.0).unwrap_or(16.0);
        let (w, h) = fitted(width.max(2), height.max(2), frames.unwrap_or(1));
        let size = [w as usize, h as usize];
        let shared = Arc::new(Shared::default());
        let scaled = (w, h) != (width, height);
        std::thread::spawn({
            let (shared, ctx, file) = (shared.clone(), ctx.clone(), file.to_path_buf());
            move || decode(&shared, &file, size, scaled, &ctx)
        });
        let sound = Arc::new(Sound {
            samples: Mutex::new(Vec::new()),
            done: AtomicBool::new(false),
            gain: AtomicU32::new(gain.to_bits()),
            clock: Mutex::new(Clock::default()),
        });
        let device = output_device();
        match &device {
            Some((_, config)) => {
                let (rate, channels) = (config.sample_rate, config.channels);
                std::thread::spawn({
                    let (shared, sound, ctx, file) = (shared.clone(), sound.clone(), ctx.clone(), file.to_path_buf());
                    move || decode_sound(&shared, &sound, &file, rate, channels, &ctx)
                });
            }
            None => sound.done.store(true, Ordering::Relaxed),
        }
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
            sound,
            device,
            stream: None,
        }
    }

    /// The frame to show now, as a texture; None until the first frame is in. Asks for a repaint in time for the
    /// next frame while playing, keeps the sound in step, and opens the audio stream once there is sound to play.
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
        *self.sound.clock.lock().unwrap() = Clock {
            started: if self.paused.is_some() { None } else { Some(self.started) },
            period: total as f64 / self.fps,
        };
        if self.stream.is_none() && !self.sound.samples.lock().unwrap().is_empty() {
            self.stream =
                self.device.take().and_then(|(device, config)| open_stream(&device, &config, self.sound.clone()));
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

    /// Whether the clip has sound: None until ffmpeg has said, Some(false) for a silent clip or no output device.
    pub fn sound(&self) -> Option<bool> {
        if !self.sound.samples.lock().unwrap().is_empty() {
            return Some(true);
        }
        self.sound.done.load(Ordering::Relaxed).then_some(false)
    }

    /// Sets the gain the sound plays at: the volume, 0 to 1, or 0 for muted.
    pub fn set_gain(&self, gain: f32) {
        self.sound.gain.store(gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
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

// ---------------------------------------------------------------- sound

/// The default output device and the format to open it with: its own rate and channels, f32 samples.
fn output_device() -> Option<(cpal::Device, cpal::StreamConfig)> {
    let device = cpal::default_host().default_output_device()?;
    let config = device.default_output_config().ok()?;
    let config = cpal::StreamConfig {
        channels: config.channels().max(1),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };
    Some((device, config))
}

/// Decodes the clip's sound with a second ffmpeg, into `sound.samples` as they come. A clip without sound leaves
/// them empty, and `done` says so.
fn decode_sound(shared: &Shared, sound: &Sound, file: &Path, rate: u32, channels: u16, ctx: &egui::Context) {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-nostdin", "-i"]).arg(file);
    cmd.args(["-vn", "-f", "f32le", "-ac", &channels.to_string(), "-ar", &rate.to_string(), "-"]);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    if let Ok(mut child) = cmd.spawn() {
        let mut out = child.stdout.take().expect("piped");
        let mut buf = vec![0u8; 64 << 10];
        let mut rest = Vec::new(); // the bytes of a sample cut by the chunk's end
        while !shared.stop.load(Ordering::Relaxed) {
            let n = match out.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            rest.extend_from_slice(&buf[..n]);
            let whole = rest.len() / 4 * 4;
            let samples: Vec<f32> =
                rest[..whole].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            rest.drain(..whole);
            if !samples.is_empty() {
                sound.samples.lock().unwrap().extend_from_slice(&samples);
                ctx.request_repaint(); // the first samples open the stream, from the UI thread
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    }
    sound.done.store(true, Ordering::Relaxed);
    ctx.request_repaint();
}

/// Opens the output stream: its callback plays `sound` from wherever the clock says, round and round.
fn open_stream(device: &cpal::Device, config: &cpal::StreamConfig, sound: Arc<Sound>) -> Option<cpal::Stream> {
    let (rate, channels) = (config.sample_rate as f64, config.channels as usize);
    let mut pos = 0usize;
    let stream = device
        .build_output_stream(
            *config,
            move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let clock = *sound.clock.lock().unwrap();
                let expected = match clock.started {
                    Some(t) if clock.period > 0.0 => Some(((t.elapsed().as_secs_f64() % clock.period) * rate) as usize),
                    _ => None,
                };
                let gain = f32::from_bits(sound.gain.load(Ordering::Relaxed));
                let samples = sound.samples.lock().unwrap();
                fill(out, &samples, channels, &mut pos, expected, (SLACK * rate) as usize, gain);
            },
            |e| eprintln!("rope-gui: audio stream: {e}"),
            None,
        )
        .map_err(|e| eprintln!("rope-gui: cannot open the audio output: {e}"))
        .ok()?;
    stream.play().map_err(|e| eprintln!("rope-gui: cannot play sound: {e}")).ok()?;
    Some(stream)
}

/// Fills `out` from `samples` (interleaved, `channels` per frame) at `pos`, a frame index, which moves on by what
/// was written. `expected` is where the clock says the sound should be, or None while paused, which fills
/// silence; when it is more than `slack` frames away, the position jumps to it, else the samples run on. Past
/// the end there is silence until the clock comes round again.
fn fill(
    out: &mut [f32],
    samples: &[f32],
    channels: usize,
    pos: &mut usize,
    expected: Option<usize>,
    slack: usize,
    gain: f32,
) {
    let Some(expected) = expected else {
        out.fill(0.0);
        return;
    };
    if pos.abs_diff(expected) > slack {
        *pos = expected;
    }
    let frames = out.len() / channels;
    let from = (*pos * channels).min(samples.len());
    let to = ((*pos + frames) * channels).min(samples.len());
    let have = to - from;
    for (o, s) in out[..have].iter_mut().zip(&samples[from..to]) {
        *o = s * gain;
    }
    out[have..].fill(0.0);
    *pos += frames;
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

    #[test]
    fn the_sound_runs_on_near_the_clock_jumps_to_it_when_far_and_falls_silent_past_the_end() {
        let samples: Vec<f32> = (0..20).map(|i| i as f32).collect(); // 10 stereo frames
        let mut out = [9.0f32; 4];
        let mut pos = 0;
        fill(&mut out, &samples, 2, &mut pos, Some(1), 3, 1.0); // 1 frame off, within the slack: runs on from 0
        assert_eq!((out, pos), ([0.0, 1.0, 2.0, 3.0], 2));
        fill(&mut out, &samples, 2, &mut pos, Some(7), 3, 0.5); // 5 off: jumps to 7, at half gain
        assert_eq!((out, pos), ([7.0, 7.5, 8.0, 8.5], 9));
        fill(&mut out, &samples, 2, &mut pos, Some(9), 3, 1.0); // one frame left, then silence
        assert_eq!((out, pos), ([18.0, 19.0, 0.0, 0.0], 11));
        fill(&mut out, &samples, 2, &mut pos, Some(12), 3, 1.0); // past the end: silence, still moving on
        assert_eq!((out, pos), ([0.0; 4], 13));
        fill(&mut out, &samples, 2, &mut pos, None, 3, 1.0); // paused: silence, held
        assert_eq!((out, pos), ([0.0; 4], 13));
        fill(&mut out, &samples, 2, &mut pos, Some(0), 3, 1.0); // round again
        assert_eq!((out, pos), ([0.0, 1.0, 2.0, 3.0], 2));
    }

    /// Needs ffmpeg with libx264 and aac, and an output device: it makes a one-second clip with a tone and plays
    /// it for a moment. `cargo test -- --ignored the_sound`.
    #[test]
    #[ignore = "plays a second of sound through the output device"]
    fn the_sound_of_a_clip_is_decoded_and_opens_the_output() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tone.mp4");
        let made = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=8"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .args(["-c:a", "aac", "-shortest"])
            .arg(&file)
            .status()
            .unwrap();
        assert!(made.success());
        let ctx = egui::Context::default();
        let mut p = Player::new(&ctx, "tone".into(), &file, (64, 64), Some(8.0), Some(8), 1.0);
        let start = Instant::now();
        while p.stream.is_none() && start.elapsed() < Duration::from_secs(5) {
            p.frame(&ctx);
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(p.sound(), Some(true), "ffmpeg decoded no sound");
        assert!(p.stream.is_some(), "the output stream did not open");
        let samples = p.sound.samples.lock().unwrap().len();
        assert!(samples > 0 && samples % 2 == 0, "{samples} samples");
        for _ in 0..30 {
            p.frame(&ctx); // keeps the clock going while it plays
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}
