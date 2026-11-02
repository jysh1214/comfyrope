//! Looking at one image as large as the window allows: fitted to the window or at 100%, zoomed around the
//! cursor, panned by dragging, stepping through the images of the current view, and trashing the ones not
//! worth keeping on the way. A clip plays, round and round, with its sound, with Space to pause and a bar below it
//! to scrub, Mute and a volume slider. Crop mode (C) picks a rectangle on the image, dragged out, moved and
//! resized by its handles, kept to a ratio if one is chosen, and Enter makes a new image of it, made from this one.

use super::{Action, App, Dialog};
use crate::model::View;
use crate::player::Player;
use crate::thumbs::{self, Key, Source};
use eframe::egui::{self, Color32, CursorIcon, PointerButton, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2};
use rope_core::crop;
use rope_core::store::NewImage;
use std::collections::{HashMap, HashSet};

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
    /// The view's order when the viewer opened, plus what has entered the view since (see `admit`). Stepping
    /// follows it, so that marking a favorite, which moves the image to the front of the view, does not change
    /// which image comes next.
    pub order: Vec<String>,
    /// Playing the clip being looked at, when it is one.
    pub player: Option<Player>,
    /// Crop mode: picking the rectangle to cut out of the image.
    pub crop: Option<Cropping>,
}

/// Picking the rectangle to crop the image to, entered with C.
pub(super) struct Cropping {
    /// The rectangle so far, in the image's pixels; None until one is dragged out.
    pub rect: Option<Rect>,
    /// What the pointer took hold of when the button went down.
    grab: Option<Grab>,
    pub ratio: Ratio,
    /// The image's size in pixels, as shown upright, and where it was last drawn, in points: the bar reads the
    /// first, the tests the second.
    pub size: Option<(u32, u32)>,
    pub drawn: Rect,
}

impl Default for Cropping {
    fn default() -> Cropping {
        Cropping { rect: None, grab: None, ratio: Ratio::Free, size: None, drawn: Rect::ZERO }
    }
}

/// The ratio a crop keeps, width to height.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) enum Ratio {
    #[default]
    Free,
    /// The image's own.
    Original,
    Square,
    /// 16:9, and 9:16.
    Wide,
    Tall,
    /// 4:3, 3:4, 3:2 and 2:3.
    FourThree,
    ThreeFour,
    ThreeTwo,
    TwoThree,
}

impl Ratio {
    pub const ALL: [Ratio; 9] = [
        Ratio::Free,
        Ratio::Original,
        Ratio::Square,
        Ratio::Wide,
        Ratio::Tall,
        Ratio::FourThree,
        Ratio::ThreeFour,
        Ratio::ThreeTwo,
        Ratio::TwoThree,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Ratio::Free => "Free",
            Ratio::Original => "Original",
            Ratio::Square => "1:1",
            Ratio::Wide => "16:9",
            Ratio::Tall => "9:16",
            Ratio::FourThree => "4:3",
            Ratio::ThreeFour => "3:4",
            Ratio::ThreeTwo => "3:2",
            Ratio::TwoThree => "2:3",
        }
    }

    /// Width over height, for an image of `size` pixels; None for free.
    fn value(self, size: Vec2) -> Option<f32> {
        Some(match self {
            Ratio::Free => return None,
            Ratio::Original => size.x / size.y.max(1.0),
            Ratio::Square => 1.0,
            Ratio::Wide => 16.0 / 9.0,
            Ratio::Tall => 9.0 / 16.0,
            Ratio::FourThree => 4.0 / 3.0,
            Ratio::ThreeFour => 3.0 / 4.0,
            Ratio::ThreeTwo => 3.0 / 2.0,
            Ratio::TwoThree => 2.0 / 3.0,
        })
    }
}

#[derive(Clone, Copy)]
enum Grab {
    /// Dragging a new rectangle out from this corner, in pixels.
    New(Pos2),
    /// Moving the rectangle: the pointer's offset from its top-left corner, in pixels.
    Move(Vec2),
    /// Dragging a corner or an edge.
    Resize(Handle),
}

/// A corner or the middle of an edge of the rectangle: -1, 0 or 1 along each axis, not both 0.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Handle {
    x: i8,
    y: i8,
}

const HANDLES: [Handle; 8] = [
    Handle { x: -1, y: -1 },
    Handle { x: 0, y: -1 },
    Handle { x: 1, y: -1 },
    Handle { x: -1, y: 0 },
    Handle { x: 1, y: 0 },
    Handle { x: -1, y: 1 },
    Handle { x: 0, y: 1 },
    Handle { x: 1, y: 1 },
];

/// How close to a handle, in points, the pointer takes it.
const HANDLE_REACH: f32 = 10.0;

/// What the pointer is over in crop mode.
enum Hit {
    Handle(Handle),
    Inside,
    Outside,
}

/// How far one press of + or - zooms.
const STEP: f32 = 1.25;

/// The height of a clip's bar, in points; the track drawn in it is thinner, the knob a little wider.
const BAR_HEIGHT: f32 = 22.0;

impl Viewer {
    /// Takes into the order the images of `visible` that are not in it yet: the results of an img2img.py run
    /// that finished while looking, an import, an image restored from the trash. Each goes right before the
    /// image that follows it in the view, or at the end, so the order of the rest stays as it was.
    pub fn admit(&mut self, visible: &[String]) {
        let at: HashMap<&String, usize> = self.order.iter().enumerate().map(|(i, id)| (id, i)).collect();
        if visible.iter().all(|id| at.contains_key(id)) {
            return;
        }
        // before[i]: the newcomers to put before order[i], each group in reverse; before[len] goes at the end
        let mut before: Vec<Vec<&String>> = vec![Vec::new(); self.order.len() + 1];
        let mut next = self.order.len();
        for id in visible.iter().rev() {
            match at.get(id) {
                Some(&i) => next = i,
                None => before[next].push(id),
            }
        }
        let mut order = Vec::with_capacity(visible.len().max(self.order.len()));
        for (i, group) in before.iter().enumerate() {
            order.extend(group.iter().rev().map(|id| (*id).clone()));
            order.extend(self.order.get(i).cloned());
        }
        self.order = order;
    }
}

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
        self.viewer = Some(Viewer { id, zoom: None, pan: Vec2::ZERO, info, fit: 1.0, order, player: None, crop: None });
    }

    /// Into crop mode on the image being looked at, or out of it again. A clip cannot be cropped.
    pub(super) fn toggle_crop(&mut self) {
        let Some(v) = self.viewer.as_mut() else { return };
        let clip = self.model.as_ref().is_some_and(|m| m.items.get(&v.id).is_some_and(|i| i.video.is_some()));
        if !clip {
            v.crop = if v.crop.is_some() { None } else { Some(Cropping::default()) };
        }
    }

    /// The ratio the crop keeps; a rectangle already picked takes it, about its center.
    pub(super) fn set_crop_ratio(&mut self, ratio: Ratio) {
        let Some(c) = self.viewer.as_mut().and_then(|v| v.crop.as_mut()) else { return };
        c.ratio = ratio;
        if let (Some(rect), Some((w, h))) = (c.rect, c.size) {
            let size = Vec2::new(w as f32, h as f32);
            if let Some(r) = ratio.value(size) {
                c.rect = Some(with_ratio(rect, r, Rect::from_min_size(Pos2::ZERO, size)));
            }
        }
    }

    /// Makes a new image of the rectangle picked, in the background: `poll` says when it is in, and looks at
    /// it. Nothing happens without a rectangle.
    pub(super) fn apply_crop(&mut self, ctx: Option<&egui::Context>) {
        let Some(v) = self.viewer.as_mut() else { return };
        let Some(c) = &v.crop else { return };
        let Some(rect) = c.rect.zip(c.size).map(|(r, size)| whole(r, size)) else { return };
        let id = v.id.clone();
        v.crop = None;
        let (tx, rx) = std::sync::mpsc::channel();
        self.cropping = Some(rx);
        let lib = self.lib.clone();
        let ctx = ctx.cloned();
        self.say(format!("cropping to {}×{}…", rect.width, rect.height), false);
        std::thread::spawn(move || {
            let new = NewImage { via: "gui".into(), ..Default::default() };
            let _ = tx.send(crop::crop(&lib, &id, rect, &new));
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
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
        let shown: HashSet<&String> = self.visible.iter().collect();
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
        let clip = v.player.is_some();
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
                let mut size = match (item.width, item.height) {
                    (Some(w), Some(h)) => format!("{w}×{h}"),
                    _ => String::new(),
                };
                if let Some(d) = item.video.as_ref().and_then(|c| c.duration) {
                    size = format!("{size}  {d:.1} s");
                }
                ui.label(RichText::new(format!("{}  {size}", rope_core::hash::short(&item.id))).weak());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.selectable_label(v.info, "Info").on_hover_text("the details panel (I)").clicked() {
                        actions.push(Action::ToggleInfo);
                    }
                    if ui.button("Open externally").on_hover_text("in the default image viewer").clicked() {
                        actions.push(Action::External(v.id.clone()));
                    }
                    ui.separator();
                    if ui
                        .add_enabled(!clip, egui::Button::selectable(v.crop.is_some(), "Crop"))
                        .on_hover_text("pick a rectangle to keep as a new image (C)")
                        .on_disabled_hover_text("a clip cannot be cropped")
                        .clicked()
                    {
                        actions.push(Action::ToggleCrop);
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
        if v.crop.is_some() {
            egui::TopBottomPanel::top("viewer-crop").show(ctx, |ui| self.crop_bar(ui, actions));
        }
        let Some(v) = &self.viewer else { return };
        if v.info {
            egui::SidePanel::right("viewer-details")
                .resizable(true)
                .default_width(380.0)
                .show(ctx, |ui| self.details(ui, actions));
        }
        if clip {
            egui::TopBottomPanel::bottom("viewer-clip").show(ctx, |ui| {
                if let Some(p) = self.viewer.as_ref().and_then(|v| v.player.as_ref()) {
                    clip_bar(ui, p, self.volume, self.muted, actions);
                }
            });
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
        if self.viewer.as_ref().is_some_and(|v| v.crop.is_some()) {
            // crop mode: Enter crops, Esc or C leaves it, zooming and the details panel go on; nothing else, so
            // that a key for the grid does not step away from or trash the image under the rectangle
            ctx.input(|i| {
                let k = |key| i.key_pressed(key);
                if k(egui::Key::Escape) || k(egui::Key::C) {
                    actions.push(Action::ToggleCrop);
                }
                if k(egui::Key::Enter) {
                    actions.push(Action::CropApply);
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
            });
            return;
        }
        ctx.input(|i| {
            let k = |key| i.key_pressed(key);
            if k(egui::Key::Escape) {
                actions.push(Action::CloseLook);
            }
            if k(egui::Key::C) {
                actions.push(Action::ToggleCrop);
            }
            if k(egui::Key::ArrowLeft) || k(egui::Key::ArrowUp) {
                actions.push(Action::Step(-1));
            }
            if k(egui::Key::ArrowRight) || k(egui::Key::ArrowDown) {
                actions.push(Action::Step(1));
            }
            if k(egui::Key::Space) {
                // a clip pauses and plays on Space; for an image it is the next one, as the arrows are
                let clip = self.viewer.as_ref().is_some_and(|v| v.player.is_some());
                actions.push(if clip { Action::PlayPause } else { Action::Step(1) });
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
            if k(egui::Key::S) && self.viewer.as_ref().is_some_and(|v| v.player.is_some()) {
                actions.push(Action::Mute);
            }
            if k(egui::Key::L)
                && let Some(v) = &self.viewer
            {
                actions.push(Action::ToggleFavorite(vec![v.id.clone()]));
            }
            if k(egui::Key::M)
                && let Some(v) = &self.viewer
            {
                actions.push(Action::Dialog(Dialog::move_to(vec![v.id.clone()])));
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
        let mut dims = item.width.zip(item.height);
        // the neighbors first: the newest request is decoded first, and that must be this image
        for n in [self.neighbor(1), self.neighbor(-1)].into_iter().flatten() {
            if let Some(i) = model.items.get(&n) {
                let s = Source { file: i.file.clone(), fallback: i.fallback.clone() };
                self.thumbs.get(&Key { id: n, size: thumbs::VIEW }, || s);
            }
        }
        // a clip plays: the frame the clock calls for, once ffmpeg has it; its poster frame until then
        let mut full = None;
        let mut cannot_play = false;
        if let (Some(clip), Some((w, h))) = (&item.video, dims) {
            let (file, fps, frames) = (item.file.clone(), clip.fps, clip.frames);
            let gain = self.gain();
            let v = self.viewer.as_mut().expect("checked");
            let p = v.player.get_or_insert_with(|| Player::new(ui.ctx(), id.clone(), &file, (w, h), fps, frames, gain));
            full = p.frame(ui.ctx());
            dims = Some((p.size[0] as u32, p.size[1] as u32));
            cannot_play = p.failed();
        }
        // the full image, or a smaller one stretched to its size while it is decoded
        if full.is_none() {
            full = self.thumbs.get(&Key { id: id.clone(), size: thumbs::VIEW }, || source.clone());
        }
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
        if cannot_play {
            ui.painter().text(
                rect.center_bottom() - Vec2::new(0.0, 24.0),
                egui::Align2::CENTER_CENTER,
                "ffmpeg could not decode this clip; this is its poster frame",
                egui::FontId::proportional(14.0),
                Color32::GRAY,
            );
        }
        let size = image_size(dims, full.as_ref().map(|t| t.size_vec2()), tex.size_vec2());
        let fit = (rect.width() / size.x).min(rect.height() / size.y);
        let v = self.viewer.as_mut().expect("checked");
        v.fit = fit;
        let scale = v.zoom.unwrap_or(fit);
        if v.zoom.is_none() {
            v.pan = Vec2::ZERO;
        }
        let center = rect.center() + v.pan;
        let drawn = Rect::from_center_size(center, size * scale);
        ui.painter_at(rect).image(tex.id(), drawn, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);

        let actual = 1.0 / ui.ctx().pixels_per_point();
        let zoom = ui.input(|i| i.zoom_delta());
        if resp.hovered() && zoom != 1.0 {
            let to = (scale * zoom).clamp(fit.min(actual) / 4.0, actual * 32.0);
            let anchor = resp.hover_pos().unwrap_or(rect.center());
            v.pan = zoom_around(anchor, center, scale, to) - rect.center();
            v.zoom = Some(to);
        }
        if let Some(c) = v.crop.as_mut() {
            // crop mode: the left button works the rectangle, the right or middle one pans
            c.size = Some(decoded_size(dims, tex.size_vec2()));
            c.drawn = drawn;
            crop_interact(ui, &resp, c);
            paint_crop(ui.painter_at(rect), rect, c);
            if (resp.dragged_by(PointerButton::Secondary) || resp.dragged_by(PointerButton::Middle)) && v.zoom.is_some()
            {
                v.pan += resp.drag_delta();
            }
            return;
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

    /// Under the bar in crop mode: the ratio, the rectangle's size and place in pixels, Crop and Cancel.
    fn crop_bar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let Some(c) = self.viewer.as_ref().and_then(|v| v.crop.as_ref()) else { return };
        ui.horizontal(|ui| {
            ui.label(RichText::new("Crop").strong());
            egui::ComboBox::from_id_salt("crop-ratio").selected_text(c.ratio.label()).width(96.0).show_ui(ui, |ui| {
                for r in Ratio::ALL {
                    if ui.selectable_label(c.ratio == r, r.label()).clicked() {
                        actions.push(Action::CropRatio(r));
                    }
                }
            });
            match c.rect.zip(c.size).map(|(r, size)| whole(r, size)) {
                Some(r) => ui.label(format!("{}×{} at {}, {}", r.width, r.height, r.x, r.y)),
                None => ui.label(RichText::new("drag on the image to pick the area to keep").weak()),
            };
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Cancel").on_hover_text("leave crop mode (Esc)").clicked() {
                    actions.push(Action::ToggleCrop);
                }
                if ui
                    .add_enabled(c.rect.is_some(), egui::Button::new("Crop"))
                    .on_hover_text("a new image of the area, made from this one, as PNG (Enter)")
                    .clicked()
                {
                    actions.push(Action::CropApply);
                }
                ui.label(
                    RichText::new("the original stays; drag inside to move, the handles to resize").weak().small(),
                );
            });
        });
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

/// Crop mode's pointer work on the image: a left drag from a handle resizes the rectangle, from inside it moves
/// it, from anywhere else drags a new one out. The cursor says which beforehand.
fn crop_interact(ui: &egui::Ui, resp: &egui::Response, c: &mut Cropping) {
    let Some((w, h)) = c.size else { return };
    let px = Vec2::new(w as f32, h as f32);
    let bounds = Rect::from_min_size(Pos2::ZERO, px);
    let ratio = c.ratio.value(px);
    let shown = c.rect.map(|r| Rect::from_min_max(to_screen(r.min, c.drawn, px), to_screen(r.max, c.drawn, px)));
    if resp.drag_started_by(PointerButton::Primary) {
        let origin = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos());
        if let Some(origin) = origin {
            c.grab = Some(match (hit(origin, shown), c.rect) {
                (Hit::Handle(handle), Some(_)) => Grab::Resize(handle),
                (Hit::Inside, Some(rect)) => Grab::Move(to_pixels(origin, c.drawn, px) - rect.min),
                _ => Grab::New(to_pixels(origin, c.drawn, px)),
            });
        }
    }
    if resp.dragged_by(PointerButton::Primary)
        && let (Some(grab), Some(pointer)) = (c.grab, resp.interact_pointer_pos())
    {
        let p = to_pixels(pointer, c.drawn, px);
        let rect = c.rect.unwrap_or(bounds);
        c.rect = Some(match grab {
            Grab::New(anchor) => dragged_out(anchor, p, ratio, bounds),
            Grab::Move(offset) => moved(rect, p - offset, bounds),
            Grab::Resize(handle) => resized(rect, handle, p, ratio, bounds),
        });
    }
    if resp.drag_stopped_by(PointerButton::Primary) {
        c.grab = None;
        // a rectangle without a pixel in it is none
        if c.rect.is_some_and(|r| r.width() < 1.0 || r.height() < 1.0) {
            c.rect = None;
        }
    }
    if let Some(p) = resp.hover_pos() {
        let icon = match c.grab {
            Some(Grab::New(_)) => CursorIcon::Crosshair,
            Some(Grab::Move(_)) => CursorIcon::Grabbing,
            Some(Grab::Resize(handle)) => resize_cursor(handle),
            None => match hit(p, shown) {
                Hit::Handle(handle) => resize_cursor(handle),
                Hit::Inside => CursorIcon::Grab,
                Hit::Outside => CursorIcon::Crosshair,
            },
        };
        ui.ctx().set_cursor_icon(icon);
    }
}

/// The rest of the image dimmed, the rectangle's edge and its handles; a word on what to do while there is no
/// rectangle.
fn paint_crop(painter: egui::Painter, rect: Rect, c: &Cropping) {
    let Some((w, h)) = c.size else { return };
    let px = Vec2::new(w as f32, h as f32);
    let Some(r) = c.rect else {
        painter.text(
            rect.center_bottom() - Vec2::new(0.0, 24.0),
            egui::Align2::CENTER_CENTER,
            "drag on the image to pick the area to keep",
            egui::FontId::proportional(14.0),
            Color32::GRAY,
        );
        return;
    };
    let s = Rect::from_min_max(to_screen(r.min, c.drawn, px), to_screen(r.max, c.drawn, px));
    let d = c.drawn;
    let dim = Color32::from_black_alpha(140);
    for part in [
        Rect::from_min_max(d.min, Pos2::new(d.max.x, s.min.y)),
        Rect::from_min_max(Pos2::new(d.min.x, s.max.y), d.max),
        Rect::from_min_max(Pos2::new(d.min.x, s.min.y), Pos2::new(s.min.x, s.max.y)),
        Rect::from_min_max(Pos2::new(s.max.x, s.min.y), Pos2::new(d.max.x, s.max.y)),
    ] {
        if part.is_positive() {
            painter.rect_filled(part, 0.0, dim);
        }
    }
    painter.rect_stroke(s, 0.0, Stroke::new(1.5, Color32::WHITE), StrokeKind::Outside);
    for handle in HANDLES {
        let square = Rect::from_center_size(handle_pos(s, handle), Vec2::splat(8.0));
        painter.rect_filled(square, 1.0, Color32::WHITE);
        painter.rect_stroke(square, 1.0, Stroke::new(1.0, Color32::from_gray(40)), StrokeKind::Outside);
    }
}

/// What the pointer at `pos` is over, with the rectangle drawn at `shown`.
fn hit(pos: Pos2, shown: Option<Rect>) -> Hit {
    let Some(r) = shown else { return Hit::Outside };
    if let Some(handle) = HANDLES.into_iter().find(|h| handle_pos(r, *h).distance(pos) <= HANDLE_REACH) {
        return Hit::Handle(handle);
    }
    if r.contains(pos) { Hit::Inside } else { Hit::Outside }
}

fn handle_pos(r: Rect, handle: Handle) -> Pos2 {
    let at = |lo: f32, mid: f32, hi: f32, i: i8| match i {
        -1 => lo,
        0 => mid,
        _ => hi,
    };
    Pos2::new(at(r.min.x, r.center().x, r.max.x, handle.x), at(r.min.y, r.center().y, r.max.y, handle.y))
}

fn resize_cursor(handle: Handle) -> CursorIcon {
    match (handle.x, handle.y) {
        (_, 0) => CursorIcon::ResizeHorizontal,
        (0, _) => CursorIcon::ResizeVertical,
        (x, y) if x == y => CursorIcon::ResizeNwSe,
        _ => CursorIcon::ResizeNeSw,
    }
}

/// From the image's pixels to the window, with the image drawn at `drawn` and `px` pixels in size, and back.
fn to_screen(p: Pos2, drawn: Rect, px: Vec2) -> Pos2 {
    drawn.min + p.to_vec2() / px * drawn.size()
}

fn to_pixels(p: Pos2, drawn: Rect, px: Vec2) -> Pos2 {
    Pos2::ZERO + (p - drawn.min) / drawn.size() * px
}

/// The image's size in pixels as shown upright: what meta.json says, the two swapped when the texture, which is
/// decoded upright, is the other way round (a photo shot in portrait, whose file says landscape); the texture's
/// own size when meta.json has none.
fn decoded_size(dims: Option<(u32, u32)>, tex: Vec2) -> (u32, u32) {
    match dims {
        Some((w, h)) if (w >= h) == (tex.x >= tex.y) => (w, h),
        Some((w, h)) => (h, w),
        None => (tex.x as u32, tex.y as u32),
    }
}

/// The rectangle dragged out from `anchor` to `pointer`, within `bounds`: with a `ratio` (width over height),
/// the smallest rectangle of that ratio that reaches the pointer, shrunk to the room there is in the drag's
/// direction.
fn dragged_out(anchor: Pos2, pointer: Pos2, ratio: Option<f32>, bounds: Rect) -> Rect {
    let p = bounds.clamp(pointer);
    let d = p - anchor;
    let sign = Vec2::new(if d.x < 0.0 { -1.0 } else { 1.0 }, if d.y < 0.0 { -1.0 } else { 1.0 });
    let mut size = d.abs();
    if let Some(r) = ratio {
        let room = Vec2::new(
            if sign.x > 0.0 { bounds.max.x - anchor.x } else { anchor.x - bounds.min.x },
            if sign.y > 0.0 { bounds.max.y - anchor.y } else { anchor.y - bounds.min.y },
        );
        size = fit_ratio(size, r, room);
    }
    Rect::from_two_pos(anchor, anchor + size * sign)
}

/// `size` taken to `ratio` (width over height), growing the lesser side so that both are covered, then shrunk
/// to fit `room`, keeping the ratio.
fn fit_ratio(size: Vec2, ratio: f32, room: Vec2) -> Vec2 {
    let mut w = size.x.max(size.y * ratio);
    let mut h = w / ratio;
    if w > room.x {
        w = room.x;
        h = w / ratio;
    }
    if h > room.y {
        h = room.y;
        w = h * ratio;
    }
    Vec2::new(w, h)
}

/// The rectangle moved so that its top-left corner is at `to`, as far as `bounds` allow.
fn moved(rect: Rect, to: Pos2, bounds: Rect) -> Rect {
    let max = bounds.max - rect.size();
    let min =
        Pos2::new(to.x.clamp(bounds.min.x, max.x.max(bounds.min.x)), to.y.clamp(bounds.min.y, max.y.max(bounds.min.y)));
    Rect::from_min_size(min, rect.size())
}

/// The rectangle with `handle` dragged to `pointer`, within `bounds` and never less than a pixel each way. An
/// edge moves on its own and a corner takes its two edges; with a `ratio`, a corner is a fresh drag from the
/// opposite corner, and an edge sets its side while the other follows, about the rectangle's middle.
fn resized(rect: Rect, handle: Handle, pointer: Pos2, ratio: Option<f32>, bounds: Rect) -> Rect {
    let p = bounds.clamp(pointer);
    let Some(ratio) = ratio else {
        let mut r = rect;
        match handle.x {
            -1 => r.min.x = p.x.min(rect.max.x - 1.0),
            1 => r.max.x = p.x.max(rect.min.x + 1.0),
            _ => {}
        }
        match handle.y {
            -1 => r.min.y = p.y.min(rect.max.y - 1.0),
            1 => r.max.y = p.y.max(rect.min.y + 1.0),
            _ => {}
        }
        return r;
    };
    let anchor = Pos2::new(
        if handle.x < 0 { rect.max.x } else { rect.min.x },
        if handle.y < 0 { rect.max.y } else { rect.min.y },
    );
    if handle.x != 0 && handle.y != 0 {
        return dragged_out(anchor, p, Some(ratio), bounds);
    }
    if handle.x != 0 {
        let room = if handle.x < 0 { anchor.x - bounds.min.x } else { bounds.max.x - anchor.x };
        let size = fit_ratio(Vec2::new((p.x - anchor.x).abs().max(1.0), 0.0), ratio, Vec2::new(room, bounds.height()));
        let x = if handle.x < 0 { anchor.x - size.x } else { anchor.x };
        let y = (rect.center().y - size.y / 2.0).clamp(bounds.min.y, bounds.max.y - size.y);
        Rect::from_min_size(Pos2::new(x, y), size)
    } else {
        let room = if handle.y < 0 { anchor.y - bounds.min.y } else { bounds.max.y - anchor.y };
        let size = fit_ratio(Vec2::new(0.0, (p.y - anchor.y).abs().max(1.0)), ratio, Vec2::new(bounds.width(), room));
        let y = if handle.y < 0 { anchor.y - size.y } else { anchor.y };
        let x = (rect.center().x - size.x / 2.0).clamp(bounds.min.x, bounds.max.x - size.x);
        Rect::from_min_size(Pos2::new(x, y), size)
    }
}

/// The rectangle taken to `ratio` about its center, as large by area as before, as far as `bounds` allow.
fn with_ratio(rect: Rect, ratio: f32, bounds: Rect) -> Rect {
    let w = (rect.area().max(1.0) * ratio).sqrt();
    let size = fit_ratio(Vec2::new(w, w / ratio), ratio, bounds.size());
    let c = rect.center();
    let min = Pos2::new(
        (c.x - size.x / 2.0).clamp(bounds.min.x, bounds.max.x - size.x),
        (c.y - size.y / 2.0).clamp(bounds.min.y, bounds.max.y - size.y),
    );
    Rect::from_min_size(min, size)
}

/// The rectangle in whole pixels of an image of `size`: its edges rounded, at least a pixel each way.
pub(super) fn whole(rect: Rect, size: (u32, u32)) -> crop::Rect {
    let (w, h) = (size.0 as f32, size.1 as f32);
    let x0 = rect.min.x.round().clamp(0.0, (w - 1.0).max(0.0));
    let y0 = rect.min.y.round().clamp(0.0, (h - 1.0).max(0.0));
    let x1 = rect.max.x.round().clamp(x0 + 1.0, w.max(x0 + 1.0));
    let y1 = rect.max.y.round().clamp(y0 + 1.0, h.max(y0 + 1.0));
    crop::Rect { x: x0 as u32, y: y0 as u32, width: (x1 - x0) as u32, height: (y1 - y0) as u32 }
}

/// Under a clip: Play/Pause, the progress bar across the width, and the position. The bar shows how far the
/// clip is and how much of it ffmpeg has decoded so far, and goes to the frame it is clicked or dragged to, the
/// frame following the pointer while the button is down.
fn clip_bar(ui: &mut egui::Ui, p: &Player, volume: f32, muted: bool, actions: &mut Vec<Action>) {
    let (shown, have, total) = (p.shown.unwrap_or(0), p.decoded(), p.total().max(1));
    ui.horizontal(|ui| {
        let label = if p.paused.is_some() { "Play" } else { "Pause" };
        if ui.button(label).on_hover_text("play or pause (Space)").clicked() {
            actions.push(Action::PlayPause);
        }
        // the sound: Mute and a volume slider, or a word for a clip that has none
        if p.sound() == Some(false) {
            ui.label(RichText::new("no sound").weak());
        } else {
            let label = if muted { "Unmute" } else { "Mute" };
            if ui.button(label).on_hover_text("sound on or off (S)").clicked() {
                actions.push(Action::Mute);
            }
            let mut v = volume;
            ui.spacing_mut().slider_width = 90.0;
            let slider = egui::Slider::new(&mut v, 0.0..=1.0).show_value(false);
            if ui.add_enabled(!muted, slider).on_hover_text(format!("volume {:.0}%", volume * 100.0)).changed() {
                actions.push(Action::Volume(v));
            }
        }
        // the position goes right, in the room its widest value takes, so the bar does not shift as digits change
        let time = |i: usize| format!("{} / {total}  {:.1} s", i + 1, i as f64 / p.fps);
        let font = egui::TextStyle::Body.resolve(ui.style());
        let width = ui.painter().layout_no_wrap(time(total - 1), font, Color32::WHITE).size().x;
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let size = Vec2::new(width, BAR_HEIGHT);
            ui.allocate_ui_with_layout(size, egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.label(RichText::new(time(shown)).weak());
            });
            let (rect, resp) =
                ui.allocate_exact_size(Vec2::new(ui.available_width().max(1.0), BAR_HEIGHT), Sense::click_and_drag());
            let track = Rect::from_center_size(rect.center(), Vec2::new(rect.width(), 6.0));
            let x = |i: usize| track.left() + track.width() * i as f32 / (total - 1).max(1) as f32;
            let painter = ui.painter();
            painter.rect_filled(track, 3.0, Color32::from_gray(60));
            if have > 0 {
                painter.rect_filled(track.with_max_x(x(have - 1)), 3.0, Color32::from_gray(100)); // decoded so far
            }
            painter.rect_filled(track.with_max_x(x(shown)), 3.0, ui.visuals().selection.bg_fill);
            let held = resp.hovered() || resp.dragged();
            let knob = Pos2::new(x(shown), track.center().y);
            painter.circle_filled(knob, if held { 8.0 } else { 6.0 }, Color32::from_gray(if held { 255 } else { 220 }));
            if resp.is_pointer_button_down_on()
                && let Some(pos) = resp.interact_pointer_pos()
            {
                actions.push(Action::Seek(frame_at(pos.x, track, total)));
            }
        });
    });
}

/// The frame the bar's point `x` stands for, of `total` (at least 1): the track's ends are the first and the
/// last frame, and so is anything beyond them, where a drag may go.
fn frame_at(x: f32, track: Rect, total: usize) -> usize {
    let t = ((x - track.left()) / track.width().max(1.0)).clamp(0.0, 1.0);
    ((t * (total - 1) as f32).round() as usize).min(total - 1)
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
    fn the_bar_ends_are_the_first_and_last_frame_and_so_is_beyond_them() {
        let track = Rect::from_min_max(Pos2::new(100.0, 0.0), Pos2::new(300.0, 6.0));
        assert_eq!(frame_at(100.0, track, 81), 0);
        assert_eq!(frame_at(300.0, track, 81), 80);
        assert_eq!(frame_at(200.0, track, 81), 40);
        assert_eq!(frame_at(20.0, track, 81), 0);
        assert_eq!(frame_at(500.0, track, 81), 80);
        assert_eq!(frame_at(250.0, track, 1), 0);
    }

    fn bounds() -> Rect {
        Rect::from_min_size(Pos2::ZERO, Vec2::new(4000.0, 3000.0))
    }

    #[test]
    fn a_rectangle_dragged_out_covers_the_drag_keeps_its_ratio_and_stays_in_the_image() {
        let free = dragged_out(Pos2::new(1000.0, 1000.0), Pos2::new(400.0, 1800.0), None, bounds());
        assert_eq!(free, Rect::from_min_max(Pos2::new(400.0, 1000.0), Pos2::new(1000.0, 1800.0)));
        // 16:9 from a mostly vertical drag: the height is the drag's, the width follows
        let wide = dragged_out(Pos2::new(1000.0, 1000.0), Pos2::new(1100.0, 1900.0), Some(16.0 / 9.0), bounds());
        assert_eq!(wide.min, Pos2::new(1000.0, 1000.0));
        assert!((wide.height() - 900.0).abs() < 0.01 && (wide.width() - 1600.0).abs() < 0.01, "{wide:?}");
        // beyond the image's edge the drag is cut, and the ratio shrinks the other side to match
        let cut = dragged_out(Pos2::new(3000.0, 1000.0), Pos2::new(5000.0, 2900.0), Some(1.0), bounds());
        assert_eq!(cut, Rect::from_min_max(Pos2::new(3000.0, 1000.0), Pos2::new(4000.0, 2000.0)));
        // dragging up and left from the anchor, the anchor is the bottom-right corner
        let back = dragged_out(Pos2::new(1000.0, 1000.0), Pos2::new(-50.0, 200.0), Some(2.0), bounds());
        assert_eq!(back, Rect::from_min_max(Pos2::new(0.0, 500.0), Pos2::new(1000.0, 1000.0)));
    }

    #[test]
    fn moving_and_resizing_keep_the_rectangle_in_the_image_and_a_pixel_wide() {
        let r = Rect::from_min_size(Pos2::new(100.0, 100.0), Vec2::new(800.0, 600.0));
        assert_eq!(moved(r, Pos2::new(3800.0, -20.0), bounds()), Rect::from_min_size(Pos2::new(3200.0, 0.0), r.size()));
        // free: an edge on its own, a corner both, never past the opposite edge
        let left = Handle { x: -1, y: 0 };
        assert_eq!(resized(r, left, Pos2::new(300.0, 2000.0), None, bounds()).min, Pos2::new(300.0, 100.0));
        assert_eq!(resized(r, left, Pos2::new(2000.0, 100.0), None, bounds()).min.x, 899.0);
        let corner = resized(r, Handle { x: 1, y: 1 }, Pos2::new(1500.0, 1200.0), None, bounds());
        assert_eq!(corner, Rect::from_min_max(Pos2::new(100.0, 100.0), Pos2::new(1500.0, 1200.0)));
        // with a ratio, a corner keeps the opposite corner and an edge the rectangle's middle
        let corner = resized(r, Handle { x: 1, y: 1 }, Pos2::new(1100.0, 1200.0), Some(2.0), bounds());
        assert_eq!(corner.min, Pos2::new(100.0, 100.0));
        assert!((corner.width() - 2200.0).abs() < 0.01 && (corner.height() - 1100.0).abs() < 0.01, "{corner:?}");
        let right = resized(r, Handle { x: 1, y: 0 }, Pos2::new(1300.0, 0.0), Some(2.0), bounds());
        assert_eq!(right.min.x, 100.0);
        assert!((right.width() - 1200.0).abs() < 0.01 && (right.height() - 600.0).abs() < 0.01, "{right:?}");
        assert!((right.center().y - r.center().y).abs() < 0.01, "about the middle");
        // taken to a square about its center, with the same area
        let square = with_ratio(r, 1.0, bounds());
        assert!((square.width() - square.height()).abs() < 0.01 && (square.area() - r.area()).abs() < 1.0);
        assert!((square.center() - r.center()).length() < 0.01);
        // whole pixels, within the image, at least one each way
        assert_eq!(whole(r, (4000, 3000)), crop::Rect { x: 100, y: 100, width: 800, height: 600 });
        let thin = Rect::from_min_size(Pos2::new(3999.6, 10.2), Vec2::new(0.1, 0.1));
        assert_eq!(whole(thin, (4000, 3000)), crop::Rect { x: 3999, y: 10, width: 1, height: 1 });
        // a portrait photo whose file says landscape: the texture, upright, tells
        assert_eq!(decoded_size(Some((4032, 3024)), Vec2::new(3072.0, 4096.0)), (3024, 4032));
        assert_eq!(decoded_size(Some((4032, 3024)), Vec2::new(4096.0, 3072.0)), (4032, 3024));
        assert_eq!(decoded_size(None, Vec2::new(640.0, 480.0)), (640, 480));
    }

    #[test]
    fn a_large_image_is_shown_at_the_size_its_texture_holds() {
        assert_eq!(image_size(Some((8000, 4000)), None, Vec2::new(256.0, 128.0)), Vec2::new(4096.0, 2048.0));
        assert_eq!(image_size(Some((1024, 768)), None, Vec2::new(256.0, 192.0)), Vec2::new(1024.0, 768.0));
        assert_eq!(image_size(None, None, Vec2::new(256.0, 192.0)), Vec2::new(256.0, 192.0));
    }
}
