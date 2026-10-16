//! meta.json: one image's identity, where its bytes were seen, and the recipe that made it.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;

pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub format: u32,
    /// The SHA-256 of the image file: the id, and the item's directory name.
    pub sha256: String,
    /// The image file in the item directory: `<sha12>.<ext>`.
    pub file: String,
    /// The display name; renaming changes only this.
    pub name: String,
    pub added: String,
    pub image: ImageInfo,
    /// Set for a clip (a video): its length, rate, frame count and codec, from ffprobe. `None` for an image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoInfo>,
    /// A clip's first frame as a PNG in the item directory (`poster.png`), for thumbnails; missing when ffmpeg
    /// was not there to make it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poster: Option<String>,
    /// Every path these bytes were seen at, outside the library. A record, never used to find the bytes.
    #[serde(default)]
    pub sources: Vec<Source>,
    /// The recipe, for an image a tool made: flags, prompt, inputs, environment. Kept as written by that tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub made: Option<Value>,
    /// The converted files actually uploaded in place of this image (HEIC or RAW turned into PNG), by SHA-256,
    /// each stored as `uploaded/<sha256>.png`, with how it was made.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub uploaded: Map<String, Value>,
    /// Replays of this image's recipe and how their pixels compared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replays: Vec<Value>,
    /// Set when the image was purged but its bytes stay because other images were made from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kept: Option<Value>,
    /// Fields a newer rope wrote, kept as they are.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// SHA-256 of the decoded pixels (see `store::pixel_hash`), for images a tool made: replays compare these,
    /// since the same pixels saved with a different embedded workflow make a different file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixels: Option<String>,
}

/// What ffprobe says about a clip; each field is `None` when it could not say.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VideoInfo {
    /// Seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    /// ffprobe's name for it: "h264", "vp9", ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub path: String,
    pub at: String,
    /// What brought it in: "import", "img2img.py", "gui".
    pub via: String,
}

/// One input of the command that made an image, in command-line order: image 1, image 2, ...
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    /// The input image's id; `None` when its bytes were never found (a migrated result whose input is gone).
    pub id: Option<String>,
    /// The SHA-256 of the converted file uploaded in its place, if it was converted.
    pub uploaded: Option<String>,
    /// The name the input had on the ComfyUI side, e.g. `img2img/0463485fc53a_photo.jpg`.
    pub upload: Option<String>,
}

impl Meta {
    pub fn load(path: &Path) -> Result<Meta> {
        let text = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        serde_json::from_slice(&text).with_context(|| format!("{} is not a valid meta.json", path.display()))
    }

    pub fn to_json(&self) -> Vec<u8> {
        let mut out = serde_json::to_vec_pretty(self).expect("meta serializes");
        out.push(b'\n');
        out
    }

    pub fn short(&self) -> &str {
        crate::hash::short(&self.sha256)
    }

    /// Whether this is a clip, not an image.
    pub fn is_clip(&self) -> bool {
        self.video.is_some()
    }

    /// The inputs of the recipe, in order; empty for an image nothing made.
    pub fn inputs(&self) -> Vec<Input> {
        let Some(list) = self.made.as_ref().and_then(|m| m.get("inputs")).and_then(Value::as_array) else {
            return Vec::new();
        };
        let text = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).map(str::to_owned);
        list.iter()
            .map(|v| Input { id: text(v, "id"), uploaded: text(v, "uploaded"), upload: text(v, "upload") })
            .collect()
    }

    /// The ids of the images this one was made from, in order; an image used twice appears twice.
    pub fn upstream(&self) -> Vec<String> {
        self.inputs().into_iter().filter_map(|i| i.id).collect()
    }

    /// A field of the recipe as text, e.g. `made_str("prompt")`.
    pub fn made_str(&self, key: &str) -> Option<&str> {
        self.made.as_ref()?.get(key)?.as_str()
    }

    /// The script that made it, by the recipe's "by": img2img.py unless the recipe names another. A bare file
    /// name, so that nothing in a recipe can point repro.sh outside the repo.
    pub fn tool(&self) -> &str {
        match self.made_str("by") {
            Some(by) if by.ends_with(".py") && !by.contains('/') && !by.starts_with('.') => by,
            _ => "img2img.py",
        }
    }

    /// The recipe's flags, as argv items.
    pub fn flags(&self) -> Vec<String> {
        self.made
            .as_ref()
            .and_then(|m| m.get("flags"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_owned).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_fields_survive_a_round_trip() {
        let v = json!({"format": 1, "sha256": "ab", "file": "ab.png", "name": "x", "added": "t",
                       "image": {"bytes": 3}, "future": {"a": 1}});
        let meta: Meta = serde_json::from_value(v).unwrap();
        let back: Value = serde_json::from_slice(&meta.to_json()).unwrap();
        assert_eq!(back["future"], json!({"a": 1}));
    }

    #[test]
    fn inputs_keep_their_order_and_repeats() {
        let v = json!({"format": 1, "sha256": "ab", "file": "ab.png", "name": "x", "added": "t",
                       "image": {"bytes": 3},
                       "made": {"inputs": [{"id": "b"}, {"id": null, "sha12": "c"}, {"id": "b", "uploaded": "d"}]}});
        let meta: Meta = serde_json::from_value(v).unwrap();
        assert_eq!(meta.upstream(), ["b", "b"]);
        assert_eq!(meta.inputs()[2].uploaded.as_deref(), Some("d"));
        assert_eq!(meta.inputs()[1].id, None);
    }
}
