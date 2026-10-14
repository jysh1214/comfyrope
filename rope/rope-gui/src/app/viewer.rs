//! Looking at one image as large as the window allows: fitted to the window or at 100%, zoomed around the
//! cursor, panned by dragging, stepping through the images of the current view, and trashing the ones not
//! worth keeping on the way.

use super::{Action, App, Dialog};
use crate::model::View;
use crate::thumbs::{self, Key, Source};
use eframe::egui::{self, Color32, Pos2, Rect, RichText, Sense, Vec2};

pub(super) struct Viewer {
    pub id: String,
    /// Screen points per image pixel; `None` fits the image to the window.
    pub zoom: Option<f32>,
    /// How far the image's center is moved from the window's, in points.
    pub pan: Vec2,
    /// The details panel next to the image.
    pub info: bool,
    /// The zoom that fits the image to the window, as last drawn.
    pub fit: f32,
    /// The view's order when the viewer opened. Stepping follows it, so that marking a favorite, which moves the
    /// image to the front of the view, does not change which image comes next.
    pub order: Vec<String>,
}

/// How far one press of + or - zooms.
const STEP: f32 = 1.25;

impl App {
    /// Opens the viewer on `id`.
    pub(super) fn look(&mut self, id: String) {
        let info = self.viewer.as_ref().is_some_and(|v| v.info);
        let order = match self.viewer.take() {
            Some(v) => v.order,
            None => self.visible.clone(),
        };
        self.selected = vec![id.clone()];
        self.anchor = Some(id.clone());
        self.load_detail(&id);
        self.viewer = Some(Viewer { id, zoom: None, pan: Vec2::ZERO, info, fit: 1.0, order });
    }

    /// Goes back to the grid, scrolled to the image last looked at.
    pub(super) fn close_look(&mut self) {
        if let Some(v) = self.viewer.take() {
            self.scroll_to = Some(v.id);
        }
    }

    /// The position of the image being looked at in the current view.
    fn place(&self) -> Option<usize> {
        let v = self.viewer.as_ref()?;
        self.order().iter().position(|id| **id == v.id)
    }

    /// The viewer's order, without the images that have left the view since (trashed, say).
    fn order(&self) -> Vec<&String> {
        let Some(v) = &self.viewer else { return Vec::new() };
        let shown: std::collections::HashSet<&String> = self.visible.iter().collect();
        v.order.iter().filter(|id| shown.contains(id)).collect()
    }

    /// The image `by` places away in the viewer's order, if there is one.
    fn neighbor(&self, by: isize) -> Option<String> {
        let i = self.place()? as isize + by;
        let order = self.order();
        (0..order.len() as isize).contains(&i).then(|| order[i as usize].clone())
    }

    pub(super) fn step(&mut self, by: isize) {
        if let Some(id) = self.neighbor(by) {
            self.look(id);
        }
    }

    /// Moves the image being looked at to the trash and goes on to the next one (or the previous, at the end).
    pub(super) fn trash_and_step(&mut self, ctx: Option<&egui::Context>) {
        let Some(id) = self.viewer.as_ref().map(|v| v.id.clone()) else { return };
        let next = self.neighbor(1).or_else(|| self.neighbor(-1));
        self.apply(Action::Trash(vec![id]), ctx);
        match next {
            Some(n) => self.look(n),
            None => self.viewer = None,
        }
    }

    pub(super) fn viewer_frame(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let gone = self.viewer.as_ref().is_some_and(|v| !self.model().items.contains_key(&v.id));
        if gone {
            self.viewer = None;
            return;
        }
        self.viewer_keys(ctx, actions);
        let Some(v) = &self.viewer else { return };
        let (place, total) = (self.place(), self.order().len());
        let favorite = self.model().catalog.is_favorite(&v.id);
        let item = &self.model().items[&v.id];
        egui::TopBottomPanel::top("viewer-bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Back").on_hover_text("back to the grid (Esc)").clicked() {
                    actions.push(Action::CloseLook);
                }
                ui.separator();
                if ui
                    .add_enabled(place.is_some_and(|p| p > 0), egui::Button::new("‹"))
                    .on_hover_text("previous (←)")
                    .clicked()
                {
                    actions.push(Action::Step(-1));
                }
                ui.label(match place {
                    Some(p) => format!("{} / {total}", p + 1),
                    None => "–".into(),
                });
                if ui
                    .add_enabled(place.is_some_and(|p| p + 1 < total), egui::Button::new("›"))
                    .on_hover_text("next (→)")
                    .clicked()
                {
                    actions.push(Action::Step(1));
                }
                ui.separator();
                if super::star_button(ui, favorite).clicked() {
                    actions.push(Action::ToggleFavorite(vec![v.id.clone()]));
                }
                ui.label(RichText::new(&item.name).strong());
                let size = match (item.width, item.height) {
                    (Some(w), Some(h)) => format!("{w}×{h}"),
                    _ => String::new(),
                };
                ui.label(RichText::new(format!("{}  {size}", rope_core::hash::short(&item.id))).weak());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.selectable_label(v.info, "Info").on_hover_text("the details panel (I)").clicked() {
                        actions.push(Action::ToggleInfo);
                    }
                    if ui.button("Open externally").on_hover_text("in the default image viewer").clicked() {
                        actions.push(Action::External(v.id.clone()));
                    }
                    ui.separator();
                    if ui.selectable_label(v.zoom.is_none(), "Fit").on_hover_text("fit to the window (F)").clicked() {
                        actions.push(Action::Zoom(None));
                    }
                    let actual = 1.0 / ui.ctx().pixels_per_point();
                    if ui
                        .selectable_label(v.zoom == Some(actual), "100%")
                        .on_hover_text("one image pixel per screen pixel (1)")
                        .clicked()
                    {
                        actions.push(Action::Zoom(Some(actual)));
                    }
                    if let Some(z) = v.zoom {
                        ui.label(RichText::new(format!("{:.0}%", z * ui.ctx().pixels_per_point() * 100.0)).weak());
                    }
                });
            });
        });
        if v.info {
            egui::SidePanel::right("viewer-details")
                .resizable(true)
                .default_width(380.0)
                .show(ctx, |ui| self.details(ui, actions));
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::from_gray(18)))
            .show(ctx, |ui| self.viewer_image(ui, actions));
    }

    fn viewer_keys(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        if self.dialog.is_some() || ctx.wants_keyboard_input() {
            return;
        }
        let actual = 1.0 / ctx.pixels_per_point();
        ctx.input(|i| {
            let k = |key| i.key_pressed(key);
            if k(egui::Key::Escape) {
                actions.push(Action::CloseLook);
            }
            if k(egui::Key::ArrowLeft) || k(egui::Key::ArrowUp) {
                actions.push(Action::Step(-1));
            }
            if k(egui::Key::ArrowRight) || k(egui::Key::ArrowDown) || k(egui::Key::Space) {
                actions.push(Action::Step(1));
            }
            if k(egui::Key::F) {
                actions.push(Action::Zoom(None));
            }
            if k(egui::Key::Num1) {
                actions.push(Action::Zoom(Some(actual)));
            }
            if k(egui::Key::Plus) || k(egui::Key::Equals) {
                actions.push(Action::ZoomBy(STEP));
            }
            if k(egui::Key::Minus) {
                actions.push(Action::ZoomBy(1.0 / STEP));
            }
            if k(egui::Key::I) {
                actions.push(Action::ToggleInfo);
            }
            if k(egui::Key::L)
                && let Some(v) = &self.viewer
            {
                actions.push(Action::ToggleFavorite(vec![v.id.clone()]));
            }
            // Backspace too: what an Apple keyboard labels delete sends Backspace
            if k(egui::Key::Delete) || k(egui::Key::Backspace) {
                if self.view == View::Trash {
                    let id = self.viewer.as_ref().map(|v| v.id.clone()).unwrap_or_default();
                    actions.push(Action::Dialog(Dialog::Purge { ids: vec![id] }));
                } else {
                    actions.push(Action::TrashAndStep);
                }
            }
            if i.events.iter().any(|e| matches!(e, egui::Event::Copy))
                && let Some(v) = &self.viewer
            {
                actions.push(Action::CopyPath(v.id.clone()));
            }
        });
    }

    fn viewer_image(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let rect = ui.max_rect();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let Some(v) = &self.viewer else { return };
        let id = v.id.clone();
        let model = self.model.as_ref().expect("loaded");
        let item = &model.items[&id];
        let source = Source { file: item.file.clone(), fallback: item.fallback.clone() };
        let dims = item.width.zip(item.height);
        // the neighbors first: the newest request is decoded first, and that must be this image
        for n in [self.neighbor(1), self.neighbor(-1)].into_iter().flatten() {
            if let Some(i) = model.items.get(&n) {
                let s = Source { file: i.file.clone(), fallback: i.fallback.clone() };
                self.thumbs.get(&Key { id: n, size: thumbs::VIEW }, || s);
            }
        }
        // the full image, or a smaller one stretched to its size while it is decoded
        let full = self.thumbs.get(&Key { id: id.clone(), size: thumbs::VIEW }, || source.clone());
        let shown = full
            .clone()
            .or_else(|| self.thumbs.get(&Key { id: id.clone(), size: thumbs::PREVIEW }, || source.clone()))
            .or_else(|| self.thumbs.get(&Key { id: id.clone(), size: thumbs::GRID }, || source));
        let Some(tex) = shown else {
            let text =
                if self.thumbs.failed(&Key { id, size: thumbs::VIEW }) { "cannot show this image" } else { "…" };
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(16.0),
                Color32::GRAY,
            );
            return;
        };
        let size = image_size(dims, full.as_ref().map(|t| t.size_vec2()), tex.size_vec2());
        let fit = (rect.width() / size.x).min(rect.height() / size.y);
        let v = self.viewer.as_mut().expect("checked");
        v.fit = fit;
        let scale = v.zoom.unwrap_or(fit);
        if v.zoom.is_none() {
            v.pan = Vec2::ZERO;
        }
        let center = rect.center() + v.pan;
        ui.painter_at(rect).image(
            tex.id(),
            Rect::from_center_size(center, size * scale),
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );

        let actual = 1.0 / ui.ctx().pixels_per_point();
        let zoom = ui.input(|i| i.zoom_delta());
        if resp.hovered() && zoom != 1.0 {
            let to = (scale * zoom).clamp(fit.min(actual) / 4.0, actual * 32.0);
            let anchor = resp.hover_pos().unwrap_or(rect.center());
            v.pan = zoom_around(anchor, center, scale, to) - rect.center();
            v.zoom = Some(to);
        }
        if resp.dragged() && v.zoom.is_some() {
            v.pan += resp.drag_delta();
        }
        if resp.double_clicked() {
            match v.zoom {
                None => {
                    let anchor = resp.interact_pointer_pos().unwrap_or(rect.center());
                    v.pan = zoom_around(anchor, center, scale, actual) - rect.center();
                    v.zoom = Some(actual);
                }
                Some(_) => actions.push(Action::Zoom(None)),
            }
        }
    }

    /// Zooms by `factor` around the window's center.
    pub(super) fn zoom_by(&mut self, factor: f32) {
        if let Some(v) = self.viewer.as_mut() {
            let from = v.zoom.unwrap_or(v.fit);
            let to = from * factor;
            v.pan *= to / from;
            v.zoom = Some(to);
        }
    }
}

/// The size to show an image at, in its own pixels: its real size, as far as the texture holds it, or the
/// texture's size when that is all there is.
fn image_size(dims: Option<(u32, u32)>, full: Option<Vec2>, shown: Vec2) -> Vec2 {
    if let Some(full) = full {
        return full;
    }
    match dims {
        Some((w, h)) => {
            let (w, h) = (w as f32, h as f32);
            let cap = thumbs::VIEW as f32 / w.max(h);
            Vec2::new(w, h) * cap.min(1.0)
        }
        None => shown,
    }
}

/// Where the image's center goes when zooming from `from` to `to` points per pixel, so that the image point under
/// `anchor` stays under it.
fn zoom_around(anchor: Pos2, center: Pos2, from: f32, to: f32) -> Pos2 {
    anchor - (anchor - center) / from * to
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zooming_keeps_the_point_under_the_cursor_in_place() {
        let (anchor, center) = (Pos2::new(300.0, 200.0), Pos2::new(500.0, 400.0));
        let new = zoom_around(anchor, center, 0.5, 2.0);
        // the image point under the anchor, before and after
        let before = (anchor - center) / 0.5;
        let after = (anchor - new) / 2.0;
        assert!((before - after).length() < 1e-4);
    }

    #[test]
    fn a_large_image_is_shown_at_the_size_its_texture_holds() {
        assert_eq!(image_size(Some((8000, 4000)), None, Vec2::new(256.0, 128.0)), Vec2::new(4096.0, 2048.0));
        assert_eq!(image_size(Some((1024, 768)), None, Vec2::new(256.0, 192.0)), Vec2::new(1024.0, 768.0));
        assert_eq!(image_size(None, None, Vec2::new(256.0, 192.0)), Vec2::new(256.0, 192.0));
    }
}
