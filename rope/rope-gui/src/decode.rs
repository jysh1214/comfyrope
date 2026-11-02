//! Decoding images for display: upright, as RGBA, and scaled down to a texture's size.

use eframe::egui::ColorImage;
use image::DynamicImage;
pub use rope_core::decode::open;

/// `img` scaled down to fit in `max` x `max` (never up), as an egui image.
pub fn fit(img: &DynamicImage, max: u32) -> ColorImage {
    let small = if img.width() > max || img.height() > max { img.thumbnail(max, max) } else { img.clone() };
    let rgba = small.to_rgba8();
    ColorImage::from_rgba_unmultiplied([rgba.width() as usize, rgba.height() as usize], rgba.as_raw())
}
