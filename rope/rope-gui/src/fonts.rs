//! Chinese text needs a CJK font, which egui does not ship. This uses the system's Noto Sans CJK, Traditional
//! Chinese face, found through fontconfig, instead of bundling 16 MB of font in the repo.

use eframe::egui;
use std::sync::Arc;

pub fn apply(ctx: &egui::Context) {
    let Some((bytes, index)) = find() else {
        eprintln!("rope-gui: no CJK font found (install noto-fonts-cjk); Chinese text will show as boxes");
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    let mut data = egui::FontData::from_owned(bytes);
    data.index = index;
    fonts.font_data.insert("cjk".into(), Arc::new(data));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("cjk".into());
    }
    ctx.set_fonts(fonts);
}

/// The file and face index fontconfig picks for Noto Sans CJK TC.
fn find() -> Option<(Vec<u8>, u32)> {
    let out = std::process::Command::new("fc-match").args(["-f", "%{file}|%{index}", "Noto Sans CJK TC"]).output().ok();
    let (file, index) = out
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.split_once('|').map(|(f, i)| (f.to_owned(), i.parse().unwrap_or(0))))
        .unwrap_or_else(|| ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc".into(), 3));
    std::fs::read(file).ok().map(|b| (b, index))
}
