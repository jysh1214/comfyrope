//! Decoding an image's pixels, upright: what the GUI shows and what a crop cuts from.

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbaImage};
use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};
use std::path::Path;

/// The image at `path`, with its EXIF orientation applied. HEIC, HEIF and AVIF go through libheif, which applies
/// their rotation itself; the rest through the image crate.
pub fn open(path: &Path) -> Result<DynamicImage> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if matches!(ext.as_str(), "heic" | "heif" | "hif" | "avif") {
        return heif(path);
    }
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("cannot decode {}", path.display()))?;
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

fn heif(path: &Path) -> Result<DynamicImage> {
    let ctx = HeifContext::read_from_file(&path.to_string_lossy())?;
    let handle = ctx.primary_image_handle()?;
    let image = LibHeif::new().decode(&handle, ColorSpace::Rgb(RgbChroma::Rgba), None)?;
    let (w, h) = (image.width(), image.height());
    let planes = image.planes();
    let plane = planes.interleaved.context("libheif returned no interleaved RGBA plane")?;
    let row = w as usize * 4;
    let mut pixels = Vec::with_capacity(row * h as usize);
    for y in 0..h as usize {
        let start = y * plane.stride;
        pixels.extend_from_slice(plane.data.get(start..start + row).context("libheif returned a short plane")?);
    }
    let img = RgbaImage::from_raw(w, h, pixels).context("libheif returned a plane of the wrong size")?;
    Ok(DynamicImage::ImageRgba8(img))
}
