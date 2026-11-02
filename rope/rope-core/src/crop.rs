//! Cutting a rectangle out of an image. The result is a new image in the library, made from the old one, with
//! the rectangle as its recipe: it has a lineage, and a `rope crop` command that makes it again. The original is
//! not touched, as no file in the library ever is.

use crate::store::{self, Added, NewImage};
use crate::{Catalog, Library, Meta, decode, time};
use anyhow::{Context, Result, bail};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

/// A rectangle in the pixels of an image as shown upright (its EXIF orientation applied): the top-left corner
/// and the size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    /// The rectangle a crop's recipe names.
    pub fn from_recipe(made: &Value) -> Option<Rect> {
        serde_json::from_value(made.get("crop")?.clone()).ok()
    }

    /// Whether the rectangle has a size and lies within an image of `width` x `height`.
    pub fn fits(&self, width: u32, height: u32) -> bool {
        self.width >= 1
            && self.height >= 1
            && self.x.checked_add(self.width).is_some_and(|r| r <= width)
            && self.y.checked_add(self.height).is_some_and(|b| b <= height)
    }

    /// As `rope crop` takes it: X Y WIDTH HEIGHT.
    pub fn args(&self) -> [String; 4] {
        [self.x.to_string(), self.y.to_string(), self.width.to_string(), self.height.to_string()]
    }
}

impl std::fmt::Display for Rect {
    /// `300x400 at 10,20`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{} at {},{}", self.width, self.height, self.x, self.y)
    }
}

/// The recipe of a crop of image `sha` to `rect`, made now.
pub fn recipe(sha: &str, rect: Rect) -> Value {
    json!({"by": "rope", "at": time::now(), "crop": rect, "inputs": [{"id": sha}]})
}

/// The name a crop gets: the original's name and the word crop.
pub fn name_for(original: &Meta) -> String {
    let name = if original.name.is_empty() { original.short() } else { &original.name };
    format!("{name} crop")
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Crops image `sha` to `rect` and adds the result as a PNG: a new image made from `sha`, named after it unless
/// `new.name` says otherwise, in its folder unless `new.folder` says otherwise; `new.made` is written here. A
/// clip cannot be cropped. The same crop made again is the same bytes, so it is the one image (`created` is
/// false the second time).
pub fn crop(lib: &Library, sha: &str, rect: Rect, new: &NewImage) -> Result<Added> {
    let meta = store::load(lib, sha)?;
    if meta.is_clip() {
        bail!("{} is a clip; rope crops images only", meta.short());
    }
    let img = pixels(lib, &meta)?;
    if !rect.fits(img.width(), img.height()) {
        bail!("{rect} does not lie within {}, which is {}x{}", meta.short(), img.width(), img.height());
    }
    let out = img.crop_imm(rect.x, rect.y, rect.width, rect.height);
    let tmp = lib.tmp_dir();
    fs::create_dir_all(&tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    let png = tmp.join(format!("crop-{}-{}.png", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
    out.save_with_format(&png, image::ImageFormat::Png).with_context(|| format!("cannot write {}", png.display()))?;
    let folder = match &new.folder {
        Some(f) => Some(f.clone()),
        None => Some(Catalog::load(lib)?.entry(sha).folder),
    };
    let new = NewImage {
        name: Some(new.name.clone().unwrap_or_else(|| name_for(&meta))),
        made: Some(recipe(sha, rect)),
        graph: None,
        via: new.via.clone(),
        folder,
        tags: new.tags.clone(),
        note: new.note.clone(),
        file_inputs: false,
    };
    let result = store::add(lib, &png, &new);
    let _ = fs::remove_file(&png);
    result
}

/// The image's pixels, upright: its file, or, when that cannot be decoded here (a camera RAW photo), the PNG that
/// was uploaded in its place.
fn pixels(lib: &Library, meta: &Meta) -> Result<DynamicImage> {
    let path = store::image_path(lib, meta);
    let Err(e) = decode::open(&path) else { return decode::open(&path) };
    let rendition = meta.uploaded.values().find_map(|u| u.get("file").and_then(Value::as_str));
    match rendition {
        Some(f) => decode::open(&lib.item_dir(&meta.sha256).join("uploaded").join(f)),
        None => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn library() -> (tempfile::TempDir, Library) {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(&dir.path().join("lib")).unwrap();
        (dir, lib)
    }

    /// An 8 x 6 PNG whose every pixel is its own position.
    fn photo(dir: &Path) -> PathBuf {
        let img = image::RgbImage::from_fn(8, 6, |x, y| image::Rgb([x as u8, y as u8, 7]));
        let path = dir.join("photo.png");
        img.save(&path).unwrap();
        path
    }

    use std::path::PathBuf;

    #[test]
    fn a_crop_is_a_new_image_made_from_the_original_in_its_folder() {
        let (dir, lib) = library();
        let folder = store::edit_catalog(&lib, "mkdir", &[], |c| c.mkdir("trips", false)).unwrap();
        let new = NewImage { via: "import".into(), folder: Some(Some(folder.clone())), ..Default::default() };
        let original = store::add(&lib, &photo(dir.path()), &new).unwrap().sha;
        let rect = Rect { x: 2, y: 1, width: 4, height: 3 };
        let added = crop(&lib, &original, rect, &NewImage { via: "gui".into(), ..Default::default() }).unwrap();
        assert!(added.created);
        let meta = store::load(&lib, &added.sha).unwrap();
        assert_eq!(meta.name, "photo crop");
        assert_eq!(meta.upstream(), std::slice::from_ref(&original));
        assert_eq!(meta.crop(), Some(rect));
        assert_eq!((meta.image.width, meta.image.height), (Some(4), Some(3)));
        assert_eq!(meta.image.format.as_deref(), Some("png"));
        assert!(meta.image.pixels.is_some(), "a made image has its pixel hash");
        assert!(meta.sources.is_empty(), "the temporary file is no source");
        let img = decode::open(&store::image_path(&lib, &meta)).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0).0, [2, 1, 7]);
        assert_eq!(img.get_pixel(3, 2).0, [5, 3, 7]);
        assert_eq!(Catalog::load(&lib).unwrap().entry(&added.sha).folder, Some(folder));
        let script = fs::read_to_string(lib.item_dir(&added.sha).join("repro.sh")).unwrap();
        assert!(script.contains(&format!("crop @{} 2 1 4 3 \"$@\"", crate::hash::short(&original))), "{script}");
        assert!(!lib.tmp_dir().join(format!("crop-{}-0.png", std::process::id())).exists());

        // the same crop again is the same bytes: the one image
        let again = crop(&lib, &original, rect, &NewImage { via: "gui".into(), ..Default::default() }).unwrap();
        assert_eq!((again.sha, again.created), (added.sha, false));
    }

    #[test]
    fn a_crop_takes_a_name_and_a_folder_of_its_own_and_refuses_to_leave_the_image() {
        let (dir, lib) = library();
        let original = store::add(&lib, &photo(dir.path()), &NewImage::default()).unwrap().sha;
        let new = NewImage { name: Some("corner".into()), folder: Some(None), ..Default::default() };
        let added = crop(&lib, &original, Rect { x: 0, y: 0, width: 1, height: 1 }, &new).unwrap();
        let meta = store::load(&lib, &added.sha).unwrap();
        assert_eq!(meta.name, "corner");
        assert_eq!(Catalog::load(&lib).unwrap().entry(&added.sha).folder, None);
        for bad in [
            Rect { x: 5, y: 0, width: 4, height: 3 },
            Rect { x: 0, y: 4, width: 4, height: 3 },
            Rect { x: 0, y: 0, width: 0, height: 3 },
            Rect { x: u32::MAX, y: 0, width: 1, height: 1 },
        ] {
            let e = crop(&lib, &original, bad, &NewImage::default()).unwrap_err();
            assert!(e.to_string().contains("8x6"), "{bad:?}: {e}");
        }
    }
}
