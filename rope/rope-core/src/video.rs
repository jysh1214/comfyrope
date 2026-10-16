//! Clips: what ffprobe says about a video file, and its first frame as a PNG from ffmpeg, both the ones on the
//! system. Without them a clip is still kept, only with no size, length or poster; `rope check --fix` asks again.

use crate::meta::{ImageInfo, VideoInfo};
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;
use std::process::{Command, Stdio};

/// The clip's size, length, rate, frame count and codec, as far as ffprobe can say; `format` is the container,
/// from the file's suffix.
pub fn file_info(path: &Path, bytes: u64, format: Option<String>) -> (ImageInfo, VideoInfo) {
    let mut image = ImageInfo { width: None, height: None, bytes, format, pixels: None };
    let mut video = VideoInfo::default();
    if let Ok(v) = probe(path) {
        let stream = v["streams"].get(0).cloned().unwrap_or(Value::Null);
        let number = |v: &Value| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64());
        image.width = stream["width"].as_u64().map(|w| w as u32);
        image.height = stream["height"].as_u64().map(|h| h as u32);
        video.codec = stream["codec_name"].as_str().map(str::to_owned);
        video.fps = fraction(&stream["r_frame_rate"]).or_else(|| fraction(&stream["avg_frame_rate"]));
        video.duration = number(&v["format"]["duration"]).or_else(|| number(&stream["duration"]));
        video.frames =
            number(&stream["nb_frames"]).map(|n| n.round() as u32).or_else(|| match (video.duration, video.fps) {
                (Some(d), Some(f)) => Some((d * f).round() as u32),
                _ => None,
            });
    }
    (image, video)
}

fn probe(path: &Path) -> Result<Value> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration")
        .args(["-of", "json"])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .context("ffprobe is not installed")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

/// ffprobe's "16/1" as a number; `None` for "0/0", the rate of a stream without one.
fn fraction(v: &Value) -> Option<f64> {
    let (n, d) = v.as_str()?.split_once('/')?;
    let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
    (n > 0.0 && d > 0.0).then_some(n / d)
}

/// Writes the clip's first frame to `png`.
pub fn poster(clip: &Path, png: &Path) -> Result<()> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(clip)
        .args(["-frames:v", "1", "-update", "1", "-f", "image2", "-vcodec", "png"])
        .arg(png)
        .stdin(Stdio::null())
        .output()
        .context("ffmpeg is not installed")?;
    if !out.status.success() || !png.exists() {
        bail!("ffmpeg could not read a frame: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}
