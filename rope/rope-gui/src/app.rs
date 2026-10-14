//! The window: rope's folders and tags on the left, the images of the chosen one in the middle, the chosen image
//! on the right. Everything it changes goes through rope-core, like the CLI, so both can be used at once; it sees
//! changes made elsewhere through the journal.

mod viewer;

use crate::model::{Model, View};
use crate::thumbs::{self, Key, Source, Thumbs};
use eframe::egui::{self, Align2, Color32, FontId, Rect, RichText, Sense, Vec2};
use rope_core::hash::short;
use rope_core::store::{self, NewImage};
use rope_core::{Library, Meta, purge, repro, time};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant, SystemTime};

/// Why the repro buttons of an image are disabled.
const NO_RECIPE: &str = "no recipe: rope did not see this image being made, so there is nothing to repeat";

const THUMB: f32 = 168.0;
const LABEL: f32 = 22.0;
const SMALL: f32 = 72.0;
const IMAGE_SUFFIXES: &[&str] = &[
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "heic", "heif", "avif", "jxl", "psd", "dng", "cr2",
    "cr3", "nef", "arw", "raf", "orf", "rw2",
];

struct Detail {
    id: String,
    meta: Meta,
    note: String,
    note_saved: String,
    note_mtime: Option<SystemTime>,
    note_conflict: bool,
    name: String,
    new_tag: String,
}

/// Work a dialog asks for, done once the dialog is drawn.
type Deferred = Box<dyn FnOnce(&mut App) -> anyhow::Result<()>>;

enum Dialog {
    NewFolder { parent: Option<String>, name: String },
    RenameFolder { id: String, name: String },
    DeleteFolder { id: String },
    Purge { ids: Vec<String> },
    Tag { ids: Vec<String>, name: String },
}

#[derive(Clone, Copy, PartialEq)]
enum Pick {
    One,
    Toggle,
    Range,
}

enum Action {
    View(View),
    Select(String, Pick),
    /// Select an image wherever it is: switch to its folder first.
    Open(String),
    SelectAll,
    Trash(Vec<String>),
    Restore(Vec<String>),
    MoveTo(Vec<String>, Option<String>),
    AddTag(Vec<String>, String),
    RemoveTag(Vec<String>, String),
    Dialog(Dialog),
    CopyPath(String),
    /// A folder's rope path, e.g. trips/京都, for --into and rope commands.
    CopyFolder(String),
    CopyRepro(String),
    CopyCommand(String),
    External(String),
    SaveName,
    SaveNote {
        force: bool,
    },
    ReloadNote,
    Import {
        folders: bool,
    },
    Undo,
    /// Look at an image as large as the window allows.
    Look(String),
    CloseLook,
    /// Look at the image this many places away in the view.
    Step(isize),
    TrashAndStep,
    /// Screen points per image pixel; `None` fits the image to the window.
    Zoom(Option<f32>),
    ZoomBy(f32),
    ToggleInfo,
    /// Mark these images as favorites, or, when all of them are already, unmark them.
    ToggleFavorite(Vec<String>),
}

pub struct App {
    lib: Library,
    model: Option<Model>,
    loading: Option<Receiver<anyhow::Result<Model>>>,
    view: View,
    search: String,
    visible: Vec<String>,
    visible_for: Option<(u64, View, String)>,
    cols: usize,
    selected: Vec<String>,
    anchor: Option<String>,
    detail: Option<Detail>,
    thumbs: Thumbs,
    changes: Receiver<()>,
    _watcher: Option<notify::RecommendedWatcher>,
    status: Option<(String, bool, Instant)>,
    undo: Option<(String, Vec<String>)>,
    dialog: Option<Dialog>,
    tasks: Vec<Receiver<Result<String, String>>>,
    scroll_to: Option<String>,
    open_on_load: Option<String>,
    /// Looking at one image as large as the window allows.
    viewer: Option<viewer::Viewer>,
}

impl App {
    pub fn new(ctx: &egui::Context, lib: Library, open: Option<String>) -> App {
        crate::fonts::apply(ctx);
        let ctx = ctx.clone();
        let (tx, loading) = channel();
        let l = lib.clone();
        let c = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Model::load(&l));
            c.request_repaint();
        });
        let (tx, changes) = channel();
        let c = ctx.clone();
        let watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if res.is_ok_and(|e| e.paths.iter().any(|p| p.file_name().is_some_and(|n| n == "journal.jsonl"))) {
                let _ = tx.send(());
                c.request_repaint();
            }
        })
        .and_then(|mut w| {
            use notify::Watcher;
            w.watch(lib.root(), notify::RecursiveMode::NonRecursive).map(|()| w)
        });
        let mut app = App {
            thumbs: Thumbs::new(&ctx, lib.cache_dir()),
            lib,
            model: None,
            loading: Some(loading),
            view: View::All,
            search: String::new(),
            visible: Vec::new(),
            visible_for: None,
            cols: 1,
            selected: Vec::new(),
            anchor: None,
            detail: None,
            changes,
            _watcher: None,
            status: None,
            undo: None,
            dialog: None,
            tasks: Vec::new(),
            scroll_to: None,
            open_on_load: open,
            viewer: None,
        };
        match watcher {
            Ok(w) => app._watcher = Some(w),
            Err(e) => app.say(format!("cannot watch the library for changes ({e}); press F5 to refresh"), true),
        }
        app
    }

    fn say(&mut self, text: impl Into<String>, error: bool) {
        self.status = Some((text.into(), error, Instant::now()));
    }

    fn model(&self) -> &Model {
        self.model.as_ref().expect("loaded")
    }

    // ---------------------------------------------------------------- keeping up

    fn poll(&mut self) {
        if let Some(rx) = &self.loading
            && let Ok(result) = rx.try_recv()
        {
            self.loading = None;
            match result {
                Ok(m) => {
                    self.model = Some(m);
                    if let Some(id) = self.open_on_load.take() {
                        match store::resolve(&self.lib, &id) {
                            Ok(sha) => self.apply(Action::Open(sha), None),
                            Err(e) => self.say(format!("{e:#}"), true),
                        }
                    }
                }
                Err(e) => self.say(format!("cannot read the library: {e:#}"), true),
            }
        }
        let mut changed = false;
        while self.changes.try_recv().is_ok() {
            changed = true;
        }
        if changed {
            self.refresh();
        }
        let mut messages = Vec::new();
        self.tasks.retain(|rx| match rx.try_recv() {
            Ok(m) => {
                messages.push(m);
                false
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => true,
            Err(_) => false,
        });
        for m in messages {
            match m {
                Ok(text) => self.say(text, false),
                Err(text) => self.say(text, true),
            }
        }
    }

    /// Brings the model up to date with the journal; called after every change this window makes too.
    fn refresh(&mut self) {
        let Some(model) = self.model.as_mut() else { return };
        match model.refresh(&self.lib) {
            Ok(changed) => {
                let model = self.model.as_ref().expect("loaded");
                self.selected.retain(|id| model.items.contains_key(id));
                if let Some(d) = &self.detail {
                    if !model.items.contains_key(&d.id) {
                        self.detail = None;
                    } else if changed.contains(&d.id) {
                        let id = d.id.clone();
                        self.load_detail(&id);
                    }
                }
            }
            Err(e) => self.say(format!("cannot read the journal: {e:#}"), true),
        }
    }

    fn update_visible(&mut self) {
        let key = (self.model().version, self.view.clone(), self.search.clone());
        if self.visible_for.as_ref() != Some(&key) {
            self.visible = self.model().view(&self.view, &self.search);
            self.visible_for = Some(key);
        }
    }

    fn load_detail(&mut self, id: &str) {
        let meta = match store::load(&self.lib, id) {
            Ok(m) => m,
            Err(e) => {
                self.detail = None;
                return self.say(format!("{e:#}"), true);
            }
        };
        let note_path = self.lib.item_dir(id).join("note.txt");
        let note = store::note(&self.lib, id).unwrap_or_default();
        let mtime = std::fs::metadata(&note_path).and_then(|m| m.modified()).ok();
        // a note being edited is kept; the user decides when the file changed under it
        if let Some(d) = self.detail.as_mut().filter(|d| d.id == id && d.note != d.note_saved) {
            d.note_conflict = note != d.note_saved;
            d.meta = meta;
            return;
        }
        self.detail = Some(Detail {
            id: id.to_owned(),
            name: meta.name.clone(),
            meta,
            note_saved: note.clone(),
            note,
            note_mtime: mtime,
            note_conflict: false,
            new_tag: String::new(),
        });
    }

    // ---------------------------------------------------------------- actions

    fn apply(&mut self, action: Action, ctx: Option<&egui::Context>) {
        let result = self.try_apply(action, ctx);
        if let Err(e) = result {
            self.say(format!("{e:#}"), true);
        }
    }

    fn try_apply(&mut self, action: Action, ctx: Option<&egui::Context>) -> anyhow::Result<()> {
        match action {
            Action::View(v) => {
                if v != self.view {
                    self.view = v;
                    self.selected.clear();
                    self.anchor = None;
                }
            }
            Action::Select(id, pick) => {
                match pick {
                    Pick::One => self.selected = vec![id.clone()],
                    Pick::Toggle => {
                        if let Some(i) = self.selected.iter().position(|s| *s == id) {
                            self.selected.remove(i);
                        } else {
                            self.selected.push(id.clone());
                        }
                    }
                    Pick::Range => {
                        let pos = |x: &str| self.visible.iter().position(|v| v == x);
                        let a = self.anchor.as_deref().and_then(pos).unwrap_or(0);
                        let b = pos(&id).unwrap_or(a);
                        let (lo, hi) = (a.min(b), a.max(b));
                        self.selected = self.visible[lo..=hi].to_vec();
                    }
                }
                if pick != Pick::Range {
                    self.anchor = Some(id.clone());
                }
                if self.selected.contains(&id) {
                    self.load_detail(&id);
                }
            }
            Action::Open(id) => {
                let model = self.model();
                let entry = model.catalog.entry(&id);
                let view = if entry.trashed.is_some() {
                    View::Trash
                } else if let Some(f) = entry.folder {
                    View::Folder(f)
                } else if model.items.get(&id).is_some_and(|i| i.kept) {
                    return Err(anyhow::anyhow!(
                        "{} was purged; only its bytes are kept, for the images made from it",
                        short(&id)
                    ));
                } else {
                    View::Unfiled
                };
                self.view = view;
                self.search.clear();
                self.selected = vec![id.clone()];
                self.anchor = Some(id.clone());
                self.load_detail(&id);
                self.scroll_to = Some(id);
            }
            Action::SelectAll => self.selected = self.visible.clone(),
            Action::Trash(ids) => {
                let now = time::now();
                let n = store::edit_catalog(&self.lib, "rm", &ids, |c| {
                    Ok(ids.iter().filter(|s| c.trash(s, &now)).count())
                })?;
                self.refresh();
                self.say(format!("moved {n} image(s) to the trash"), false);
                self.undo = Some(("restore".into(), ids));
            }
            Action::Restore(ids) => {
                let n = store::edit_catalog(&self.lib, "restore", &ids, |c| {
                    Ok(ids.iter().filter(|s| c.restore(s)).count())
                })?;
                self.refresh();
                self.say(format!("restored {n} image(s)"), false);
            }
            Action::MoveTo(ids, folder) => {
                store::edit_catalog(&self.lib, "mv", &ids, |c| {
                    for s in &ids {
                        c.set_folder(s, folder.clone());
                    }
                    Ok(())
                })?;
                self.refresh();
                let place = folder.map(|f| self.model().catalog.folder_path(&f)).unwrap_or_else(|| "unfiled".into());
                self.say(format!("moved {} image(s) to {place}", ids.len()), false);
            }
            Action::AddTag(ids, tag) => {
                let tag = tag.trim().to_owned();
                if !tag.is_empty() {
                    store::edit_catalog(&self.lib, "tag", &ids, |c| {
                        ids.iter().try_for_each(|s| c.add_tag(s, &tag).map(drop))
                    })?;
                    self.refresh();
                }
            }
            Action::RemoveTag(ids, tag) => {
                store::edit_catalog(&self.lib, "tag", &ids, |c| {
                    ids.iter().for_each(|s| {
                        c.remove_tag(s, &tag);
                    });
                    Ok(())
                })?;
                self.refresh();
            }
            Action::Dialog(d) => self.dialog = Some(d),
            Action::CopyPath(id) => {
                let path = self.model().items.get(&id).map(|i| i.file.display().to_string()).unwrap_or_default();
                self.copy(ctx, path, "the path");
            }
            Action::CopyFolder(id) => {
                let path = self.folder_arg(&id);
                self.copy(ctx, path, "the folder path");
            }
            Action::CopyRepro(id) => {
                let meta = store::load(&self.lib, &id)?;
                let cmd = repro::shell_join(&repro::replay_argv(&self.lib, &meta)?);
                self.copy(ctx, cmd, "the repro command");
            }
            Action::CopyCommand(id) => {
                let meta = store::load(&self.lib, &id)?;
                let catalog = &self.model().catalog;
                let folder = catalog.entry(&id).folder.map(|f| catalog.folder_path(&f));
                let cmd = repro::shell_join(&repro::command_argv(&self.lib, &meta, folder.as_deref())?);
                self.copy(ctx, cmd, "the img2img.py command");
            }
            Action::External(id) => {
                if let Some(item) = self.model().items.get(&id) {
                    std::process::Command::new("xdg-open").arg(&item.file).spawn()?;
                }
            }
            Action::SaveName => {
                if let Some(d) = &self.detail
                    && d.name.trim() != d.meta.name
                    && !d.name.trim().is_empty()
                {
                    store::set_name(&self.lib, &d.id, &d.name)?;
                    self.refresh();
                }
            }
            Action::SaveNote { force } => self.save_note(force)?,
            Action::ReloadNote => {
                if let Some(d) = self.detail.as_mut() {
                    d.note = d.note_saved.clone();
                    d.note_conflict = false;
                }
                if let Some(id) = self.detail.as_ref().map(|d| d.id.clone()) {
                    self.detail = None;
                    self.load_detail(&id);
                }
            }
            Action::Import { folders } => self.import(folders, ctx),
            Action::Look(id) => self.look(id),
            Action::CloseLook => self.close_look(),
            Action::Step(by) => self.step(by),
            Action::TrashAndStep => self.trash_and_step(ctx),
            Action::Zoom(zoom) => {
                if let Some(v) = self.viewer.as_mut() {
                    (v.zoom, v.pan) = (zoom, Vec2::ZERO);
                }
            }
            Action::ZoomBy(factor) => self.zoom_by(factor),
            Action::ToggleFavorite(ids) => {
                let on = !ids.iter().all(|s| self.model().catalog.is_favorite(s));
                store::edit_catalog(&self.lib, "fav", &ids, |c| {
                    ids.iter().for_each(|s| {
                        c.set_favorite(s, on);
                    });
                    Ok(())
                })?;
                self.refresh();
            }
            Action::ToggleInfo => {
                if let Some(v) = self.viewer.as_mut() {
                    v.info = !v.info;
                }
            }
            Action::Undo => {
                if let Some((what, ids)) = self.undo.take()
                    && what == "restore"
                {
                    self.apply(Action::Restore(ids), ctx);
                }
            }
        }
        Ok(())
    }

    fn copy(&mut self, ctx: Option<&egui::Context>, text: String, what: &str) {
        if let Some(ctx) = ctx {
            ctx.copy_text(text);
            self.say(format!("copied {what}"), false);
        }
    }

    fn save_note(&mut self, force: bool) -> anyhow::Result<()> {
        let Some(d) = self.detail.as_mut() else { return Ok(()) };
        if d.note == d.note_saved {
            return Ok(());
        }
        let path = self.lib.item_dir(&d.id).join("note.txt");
        let now = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if now != d.note_mtime && !force {
            d.note_conflict = true;
            return Ok(());
        }
        store::set_note(&self.lib, &d.id, &d.note)?;
        d.note_saved = d.note.clone();
        d.note_mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        d.note_conflict = false;
        self.refresh();
        Ok(())
    }

    /// Imports what a file dialog picks, into the folder (or with the tag) being viewed, in the background.
    fn import(&mut self, folders: bool, ctx: Option<&egui::Context>) {
        let dialog = rfd::FileDialog::new().set_title("Import into rope");
        let picked: Vec<PathBuf> = if folders {
            dialog.pick_folders().unwrap_or_default()
        } else {
            dialog.add_filter("Images", IMAGE_SUFFIXES).pick_files().unwrap_or_default()
        };
        if !picked.is_empty() {
            self.import_paths(picked, folders, ctx);
        }
    }

    /// Imports in the background. Files go into the folder being viewed, or get the tag being viewed. A folder
    /// becomes a rope folder of the same name at the top, with its subfolders, wherever the window is.
    fn import_paths(&mut self, picked: Vec<PathBuf>, folders: bool, ctx: Option<&egui::Context>) {
        let new = NewImage {
            via: "gui".into(),
            folder: match &self.view {
                View::Folder(f) if !folders => Some(Some(f.clone())),
                _ => None,
            },
            tags: match &self.view {
                View::Tag(t) if !folders => vec![t.clone()],
                _ => Vec::new(),
            },
            ..Default::default()
        };
        let (tx, rx) = channel();
        self.tasks.push(rx);
        let lib = self.lib.clone();
        let ctx = ctx.cloned();
        self.say(format!("importing {} item(s)…", picked.len()), false);
        std::thread::spawn(move || {
            let results = if folders {
                let files: Vec<(PathBuf, String)> = picked.iter().flat_map(|d| store::mirror(d, "")).collect();
                store::add_to_folders(&lib, &files, &new)
            } else {
                let mut files = Vec::new();
                for p in picked {
                    collect(&p, &mut files);
                }
                files.into_iter().map(|f| (f.clone(), store::add(&lib, &f, &new))).collect()
            };
            let (mut added, mut known, mut failed) = (0, 0, Vec::new());
            for (f, r) in &results {
                match r {
                    Ok(a) if a.created => added += 1,
                    Ok(_) => known += 1,
                    Err(e) => failed.push(format!("{}: {e:#}", f.display())),
                }
            }
            let text = format!("imported {added} new image(s), {known} already in rope");
            let _ = tx.send(if failed.is_empty() {
                Ok(text)
            } else {
                Err(format!("{text}; failed: {}", failed.join("; ")))
            });
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
    }

    // ---------------------------------------------------------------- drawing

    fn sidebar(&self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let model = self.model();
        let counts = &model.counts;
        let row = |ui: &mut egui::Ui, view: View, text: String, actions: &mut Vec<Action>| {
            if ui.selectable_label(self.view == view, text).clicked() {
                actions.push(Action::View(view));
            }
        };
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            row(ui, View::All, format!("All  ({})", model.shown), actions);
            row(ui, View::Unfiled, format!("Unfiled  ({})", counts.get(&None).copied().unwrap_or(0)), actions);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Folders").strong());
                if ui.small_button("+").on_hover_text("New folder").clicked() {
                    actions.push(Action::Dialog(Dialog::NewFolder { parent: None, name: String::new() }));
                }
            });
            self.folder_rows(ui, None, counts, actions);
            let tags = &model.tags;
            if !tags.is_empty() {
                ui.add_space(8.0);
                ui.label(RichText::new("Tags").strong());
                for (t, n) in tags {
                    row(ui, View::Tag(t.clone()), format!("#{t}  ({n})"), actions);
                }
            }
            ui.add_space(8.0);
            let trash = ui.selectable_label(self.view == View::Trash, format!("Trash  ({})", model.trashed));
            if trash.clicked() {
                actions.push(Action::View(View::Trash));
            }
            trash.context_menu(|ui| {
                let clean = ui.add_enabled(model.trashed > 0, egui::Button::new("Clean"));
                if clean.on_hover_text("delete everything in the trash for good").clicked() {
                    actions.push(Action::Dialog(Dialog::Purge { ids: self.trashed() }));
                    ui.close();
                }
            });
        });
    }

    /// Makes the folder at `path` and the missing folders above it: `a/b` makes `b` inside `a`.
    fn new_folder(&mut self, path: &str) -> anyhow::Result<()> {
        if rope_core::catalog::split(path).next().is_none() {
            anyhow::bail!("a folder needs a name");
        }
        if self.model().catalog.find(path).is_some() {
            anyhow::bail!("folder {path:?} exists");
        }
        store::edit_catalog(&self.lib, "mkdir", &[], |c| c.mkdir(path, true).map(drop))?;
        self.refresh();
        Ok(())
    }

    /// A folder's rope path as a command takes it: `trips/京都`, quoted for the shell when it has a space or a
    /// shell character, like `'trips/day 1'`.
    fn folder_arg(&self, id: &str) -> String {
        rope_core::catalog::shell_word(&self.model().catalog.folder_path(id))
    }

    /// Deletes a folder with everything in it: its subfolders go, its images go to the trash.
    fn delete_folder(&mut self, id: &str) -> anyhow::Result<()> {
        let path = self.model().catalog.folder_path(id);
        let gone = self.model().catalog.subtree(id);
        let (folders, trashed) = store::remove_folder_tree(&self.lib, &path)?;
        if matches!(&self.view, View::Folder(f) if gone.contains(f)) {
            self.view = View::All;
        }
        self.refresh();
        self.say(format!("deleted {folders} folder(s); moved {} image(s) to the trash", trashed.len()), false);
        Ok(())
    }

    /// Every image in the trash.
    fn trashed(&self) -> Vec<String> {
        let model = self.model();
        let mut ids: Vec<String> =
            model.items.values().filter(|i| !i.kept && model.catalog.is_trashed(&i.id)).map(|i| i.id.clone()).collect();
        ids.sort();
        ids
    }

    fn folder_rows(
        &self,
        ui: &mut egui::Ui,
        parent: Option<&str>,
        counts: &std::collections::HashMap<Option<String>, usize>,
        actions: &mut Vec<Action>,
    ) {
        let model = self.model();
        for (id, folder) in model.catalog.children(parent) {
            let n = counts.get(&Some(id.to_owned())).copied().unwrap_or(0);
            let text = format!("{}  ({n})", folder.name);
            let selected = self.view == View::Folder(id.to_owned());
            let menu = |resp: &egui::Response, actions: &mut Vec<Action>| {
                resp.context_menu(|ui| {
                    if ui.button("Copy path").on_hover_text("for --into and rope commands").clicked() {
                        actions.push(Action::CopyFolder(id.to_owned()));
                        ui.close();
                    }
                    if ui.button("New subfolder…").clicked() {
                        actions.push(Action::Dialog(Dialog::NewFolder {
                            parent: Some(id.to_owned()),
                            name: String::new(),
                        }));
                        ui.close();
                    }
                    if ui.button("Rename…").clicked() {
                        actions.push(Action::Dialog(Dialog::RenameFolder {
                            id: id.to_owned(),
                            name: folder.name.clone(),
                        }));
                        ui.close();
                    }
                    if ui.button("Delete…").clicked() {
                        actions.push(Action::Dialog(Dialog::DeleteFolder { id: id.to_owned() }));
                        ui.close();
                    }
                });
            };
            if model.catalog.children(Some(id)).is_empty() {
                ui.horizontal(|ui| {
                    ui.add_space(ui.spacing().indent);
                    let resp = ui.selectable_label(selected, text);
                    if resp.clicked() {
                        actions.push(Action::View(View::Folder(id.to_owned())));
                    }
                    menu(&resp, actions);
                });
            } else {
                let state = egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    ui.make_persistent_id(("folder", id)),
                    true,
                );
                state
                    .show_header(ui, |ui| {
                        let resp = ui.selectable_label(selected, text);
                        if resp.clicked() {
                            actions.push(Action::View(View::Folder(id.to_owned())));
                        }
                        menu(&resp, actions);
                    })
                    .body(|ui| self.folder_rows(ui, Some(id), counts, actions));
            }
        }
    }

    fn grid(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let n = self.visible.len();
        if n == 0 {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new(if self.search.is_empty() { "No images here" } else { "Nothing matches" }).weak(),
                );
            });
            return;
        }
        let gap = ui.spacing().item_spacing;
        let cell = Vec2::new(THUMB, THUMB + LABEL);
        self.cols = (((ui.available_width() + gap.x) / (cell.x + gap.x)).floor() as usize).max(1);
        let cols = self.cols;
        let rows = n.div_ceil(cols);
        let mut area = egui::ScrollArea::vertical().auto_shrink([false, false]);
        if let Some(id) = self.scroll_to.take()
            && let Some(i) = self.visible.iter().position(|v| *v == id)
        {
            area = area.vertical_scroll_offset(((i / cols) as f32 * (cell.y + gap.y) - cell.y).max(0.0));
        }
        area.show_rows(ui, cell.y, rows, |ui, range| {
            for row in range {
                ui.horizontal(|ui| {
                    for i in row * cols..((row + 1) * cols).min(n) {
                        let id = self.visible[i].clone();
                        self.cell(ui, &id, cell, actions);
                    }
                });
            }
        });
    }

    fn cell(&mut self, ui: &mut egui::Ui, id: &str, size: Vec2, actions: &mut Vec<Action>) {
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
        if !ui.is_rect_visible(rect) {
            return;
        }
        let Some(item) = self.model.as_ref().and_then(|m| m.items.get(id)) else { return };
        let selected = self.selected.iter().any(|s| s == id);
        let visuals = ui.visuals();
        if selected {
            ui.painter().rect_filled(rect, 4.0, visuals.selection.bg_fill);
        } else if resp.hovered() {
            ui.painter().rect_filled(rect, 4.0, visuals.widgets.hovered.weak_bg_fill);
        }
        let square = Rect::from_min_size(rect.min, Vec2::splat(THUMB)).shrink(4.0);
        let key = Key { id: id.to_owned(), size: thumbs::GRID };
        let (file, fallback) = (item.file.clone(), item.fallback.clone());
        match self.thumbs.get(&key, || Source { file, fallback }) {
            Some(tex) => {
                let fitted = fit(tex.size_vec2(), square.size());
                egui::Image::new(&tex).paint_at(ui, Rect::from_center_size(square.center(), fitted));
            }
            None => {
                let text = if self.thumbs.failed(&key) { "cannot show" } else { "…" };
                ui.painter().text(
                    square.center(),
                    Align2::CENTER_CENTER,
                    text,
                    FontId::proportional(13.0),
                    visuals.weak_text_color(),
                );
            }
        }
        let favorite = self.model.as_ref().is_some_and(|m| m.catalog.is_favorite(id));
        let star_rect = Rect::from_min_size(egui::pos2(square.max.x - 26.0, square.min.y + 4.0), Vec2::splat(22.0));
        let star = ui.interact(star_rect, ui.id().with(("favorite", id)), Sense::click());
        if favorite || resp.contains_pointer() {
            paint_star(ui.painter(), star_rect.shrink(2.0), favorite, star.hovered());
        }
        if star.on_hover_text(if favorite { "remove from favorites (L)" } else { "add to favorites (L)" }).clicked() {
            actions.push(Action::ToggleFavorite(vec![id.to_owned()]));
        }
        let item = self.model.as_ref().and_then(|m| m.items.get(id)).expect("present");
        let label = Rect::from_min_max(
            egui::pos2(rect.min.x + 4.0, rect.max.y - LABEL),
            egui::pos2(rect.max.x - 4.0, rect.max.y),
        );
        let color = if selected { visuals.selection.stroke.color } else { visuals.text_color() };
        let mark = if item.made { "" } else { "◦ " };
        ui.painter().with_clip_rect(label).text(
            label.left_center(),
            Align2::LEFT_CENTER,
            format!("{mark}{}", item.name),
            FontId::proportional(12.5),
            color,
        );
        let name = item.name.clone();
        let resp = resp.on_hover_text(name);
        if resp.clicked() {
            let m = ui.input(|i| i.modifiers);
            let pick = if m.shift {
                Pick::Range
            } else if m.command {
                Pick::Toggle
            } else {
                Pick::One
            };
            actions.push(Action::Select(id.to_owned(), pick));
        }
        if resp.double_clicked() {
            actions.push(Action::Look(id.to_owned()));
        }
        let targets: Vec<String> = if selected { self.selected.clone() } else { vec![id.to_owned()] };
        let trash_view = self.view == View::Trash;
        let model = self.model.as_ref().expect("loaded");
        let made = model.items.get(id).is_some_and(|i| i.made);
        resp.context_menu(|ui| {
            if !selected {
                actions.push(Action::Select(id.to_owned(), Pick::One));
            }
            if ui.button("Copy path").clicked() {
                actions.push(Action::CopyPath(id.to_owned()));
                ui.close();
            }
            let repro = ui.add_enabled(made, egui::Button::new("Copy repro command"));
            if repro.on_disabled_hover_text(NO_RECIPE).clicked() {
                actions.push(Action::CopyRepro(id.to_owned()));
                ui.close();
            }
            if ui.button("View large").clicked() {
                actions.push(Action::Look(id.to_owned()));
                ui.close();
            }
            if ui.button("Open in external viewer").clicked() {
                actions.push(Action::External(id.to_owned()));
                ui.close();
            }
            ui.separator();
            ui.menu_button("Move to", |ui| {
                if ui.button("/ (unfiled)").clicked() {
                    actions.push(Action::MoveTo(targets.clone(), None));
                    ui.close();
                }
                for (path, fid) in &model.paths {
                    if ui.button(path).clicked() {
                        actions.push(Action::MoveTo(targets.clone(), Some(fid.clone())));
                        ui.close();
                    }
                }
            });
            if ui.button("Add tag…").clicked() {
                actions.push(Action::Dialog(Dialog::Tag { ids: targets.clone(), name: String::new() }));
                ui.close();
            }
            let all_favorite = targets.iter().all(|s| model.catalog.is_favorite(s));
            if ui.button(if all_favorite { "Remove from favorites" } else { "Add to favorites" }).clicked() {
                actions.push(Action::ToggleFavorite(targets.clone()));
                ui.close();
            }
            ui.separator();
            if trash_view {
                if ui.button("Restore").clicked() {
                    actions.push(Action::Restore(targets.clone()));
                    ui.close();
                }
                if ui.button("Delete for good…").clicked() {
                    actions.push(Action::Dialog(Dialog::Purge { ids: targets.clone() }));
                    ui.close();
                }
            } else if ui.button("Move to trash").clicked() {
                actions.push(Action::Trash(targets.clone()));
                ui.close();
            }
        });
    }

    fn details(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        if self.selected.len() > 1 {
            return self.bulk(ui, actions);
        }
        let Some(mut d) = self.detail.take().filter(|d| self.selected.contains(&d.id)) else {
            ui.centered_and_justified(|ui| ui.label(RichText::new("Select an image").weak()));
            return;
        };
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.detail_body(ui, &mut d, actions));
        self.detail = Some(d);
    }

    fn detail_body(&mut self, ui: &mut egui::Ui, d: &mut Detail, actions: &mut Vec<Action>) {
        let id = d.id.clone();
        let width = ui.available_width();
        let Some(item) = self.model.as_ref().and_then(|m| m.items.get(&id)) else { return };
        let source = Source { file: item.file.clone(), fallback: item.fallback.clone() };
        let box_size = Vec2::new(width, width.min(420.0));
        let (rect, resp) = ui.allocate_exact_size(box_size, Sense::click());
        // the preview, or the grid thumbnail while the preview is being decoded
        let preview = self.thumbs.get(&Key { id: id.clone(), size: thumbs::PREVIEW }, || source.clone());
        let shown = preview.or_else(|| self.thumbs.get(&Key { id: id.clone(), size: thumbs::GRID }, || source));
        if let Some(tex) = shown {
            egui::Image::new(&tex)
                .paint_at(ui, Rect::from_center_size(rect.center(), fit(tex.size_vec2(), rect.size())));
        }
        if self.viewer.is_none() && resp.on_hover_text("click: view large").clicked() {
            actions.push(Action::Look(id.clone()));
        }
        let m = &d.meta;
        let catalog = &self.model.as_ref().expect("loaded").catalog;
        let entry = catalog.entry(&id);

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if star_button(ui, entry.favorite).clicked() {
                actions.push(Action::ToggleFavorite(vec![id.clone()]));
            }
            let name = ui.add(
                egui::TextEdit::singleline(&mut d.name).desired_width(f32::INFINITY).font(egui::TextStyle::Heading),
            );
            if name.lost_focus() {
                actions.push(Action::SaveName);
            }
        });
        let size = match (m.image.width, m.image.height) {
            (Some(w), Some(h)) => format!("{w}×{h}  "),
            _ => String::new(),
        };
        ui.label(
            RichText::new(format!(
                "{}  {size}{}  {}  {}",
                m.short(),
                m.image.format.as_deref().unwrap_or("?"),
                human(m.image.bytes),
                m.added.get(..16).unwrap_or(&m.added).replace('T', " ")
            ))
            .weak(),
        );
        if m.kept.is_some() {
            ui.colored_label(
                Color32::from_rgb(200, 140, 40),
                "Purged: only its bytes are kept, for the images made from it.",
            );
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Copy path").clicked() {
                actions.push(Action::CopyPath(id.clone()));
            }
            ui.add_enabled_ui(m.made.is_some(), |ui| {
                if ui
                    .button("Copy repro")
                    .on_hover_text("the command that makes this image again (img2img.py --replay)")
                    .on_disabled_hover_text(NO_RECIPE)
                    .clicked()
                {
                    actions.push(Action::CopyRepro(id.clone()));
                }
                if ui
                    .button("Copy command")
                    .on_hover_text("the full img2img.py command, to change and run as a variation")
                    .on_disabled_hover_text(NO_RECIPE)
                    .clicked()
                {
                    actions.push(Action::CopyCommand(id.clone()));
                }
            });
            if entry.trashed.is_some() {
                if ui.button("Restore").clicked() {
                    actions.push(Action::Restore(vec![id.clone()]));
                }
                if ui.button("Delete for good…").clicked() {
                    actions.push(Action::Dialog(Dialog::Purge { ids: vec![id.clone()] }));
                }
            } else if m.kept.is_none() && ui.button("Delete").on_hover_text("move it to the trash").clicked() {
                actions.push(Action::Trash(vec![id.clone()]));
            }
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Folder");
            let current = entry.folder.clone();
            let shown = current.as_ref().map(|f| catalog.folder_path(f)).unwrap_or_else(|| "/ (unfiled)".into());
            egui::ComboBox::from_id_salt("folder").selected_text(shown).width(ui.available_width() - 8.0).show_ui(
                ui,
                |ui| {
                    if ui.selectable_label(current.is_none(), "/ (unfiled)").clicked() && current.is_some() {
                        actions.push(Action::MoveTo(vec![id.clone()], None));
                    }
                    for (path, fid) in self.model.as_ref().expect("loaded").paths.clone() {
                        if ui.selectable_label(current.as_deref() == Some(fid.as_str()), path).clicked()
                            && current.as_deref() != Some(fid.as_str())
                        {
                            actions.push(Action::MoveTo(vec![id.clone()], Some(fid)));
                        }
                    }
                },
            );
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Tags");
            for t in &entry.tags {
                if ui.small_button(format!("#{t}  ×")).on_hover_text("remove this tag").clicked() {
                    actions.push(Action::RemoveTag(vec![id.clone()], t.clone()));
                }
            }
            let add = ui.add(egui::TextEdit::singleline(&mut d.new_tag).hint_text("add a tag").desired_width(110.0));
            if add.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !d.new_tag.trim().is_empty() {
                actions.push(Action::AddTag(vec![id.clone()], std::mem::take(&mut d.new_tag)));
            }
        });
        if !d.new_tag.trim().is_empty() {
            let typed = d.new_tag.trim().to_lowercase();
            let all = &self.model.as_ref().expect("loaded").tags;
            let matches: Vec<&String> =
                all.keys().filter(|t| t.to_lowercase().contains(&typed) && !entry.tags.contains(t)).take(8).collect();
            if !matches.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    for t in matches {
                        if ui.small_button(format!("#{t}")).clicked() {
                            actions.push(Action::AddTag(vec![id.clone()], t.clone()));
                            d.new_tag.clear();
                        }
                    }
                });
            }
        }

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Note");
            let dirty = d.note != d.note_saved;
            if ui.add_enabled(dirty, egui::Button::new("Save").small()).clicked() {
                actions.push(Action::SaveNote { force: false });
            }
            if dirty && ui.small_button("Revert").clicked() {
                d.note = d.note_saved.clone();
            }
        });
        if d.note_conflict {
            ui.colored_label(Color32::from_rgb(220, 120, 40), "The note was changed elsewhere since you opened it.");
            ui.horizontal(|ui| {
                if ui.small_button("Keep mine").clicked() {
                    actions.push(Action::SaveNote { force: true });
                }
                if ui.small_button("Take theirs").clicked() {
                    actions.push(Action::ReloadNote);
                }
            });
        }
        let note = ui.add(egui::TextEdit::multiline(&mut d.note).desired_rows(5).desired_width(f32::INFINITY));
        if note.lost_focus() && d.note != d.note_saved {
            actions.push(Action::SaveNote { force: false });
        }

        let inputs = m.inputs();
        if !inputs.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new("Made from").strong());
            for (slot, input) in inputs.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}", slot + 1));
                    match &input.id {
                        Some(up) => self.link(ui, up, actions),
                        None => {
                            ui.label(RichText::new("missing: rope never had these bytes").weak());
                        }
                    }
                });
            }
        }
        let children: Vec<String> = self.model.as_ref().expect("loaded").children(&id).to_vec();
        if !children.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new(format!("Made from this  ({})", children.len())).strong());
            ui.horizontal_wrapped(|ui| {
                for c in &children {
                    self.small_thumb(ui, c, actions);
                }
            });
        }

        if let Some(made) = &m.made {
            ui.add_space(6.0);
            ui.label(RichText::new("Recipe").strong());
            ui.label(RichText::new(repro::shell_join(&m.flags())).monospace().small());
            if let Some(p) = made.get("prompt").and_then(|v| v.as_str()) {
                ui.add(egui::Label::new(p).wrap());
            }
            if let Some(p) = made.get("enhanced_prompt").and_then(|v| v.as_str()).filter(|p| !p.is_empty()) {
                ui.label(RichText::new("enhanced prompt").weak().small());
                ui.add(egui::Label::new(RichText::new(p).small()).wrap());
            }
            if let Some(s) = made.get("server").and_then(|v| v.as_str()) {
                ui.label(RichText::new(s).weak().small());
            }
        }
        if !m.replays.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new("Replays").strong());
            for r in &m.replays {
                let at = r["at"].as_str().unwrap_or("?");
                let result = r["result"].as_str().unwrap_or("?");
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{} {result}", at.get(..16).unwrap_or(at).replace('T', " "))).small(),
                    );
                    if let Some(new) = r.get("id").and_then(|v| v.as_str()) {
                        self.link(ui, new, actions);
                    }
                });
            }
        }
        if !m.sources.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new("Seen at").strong());
            for s in &m.sources {
                let gone = !Path::new(&s.path).exists();
                let text = RichText::new(format!("{}{}", s.path, if gone { "  (gone)" } else { "" })).small();
                ui.add(egui::Label::new(if gone { text.weak() } else { text }).wrap());
            }
        }
    }

    /// A small thumbnail with the image's name; clicking it goes to that image.
    fn link(&mut self, ui: &mut egui::Ui, id: &str, actions: &mut Vec<Action>) {
        self.small_thumb(ui, id, actions);
        let model = self.model.as_ref().expect("loaded");
        let name = model.items.get(id).map(|i| if i.kept { format!("{} (purged)", short(id)) } else { i.name.clone() });
        ui.label(RichText::new(name.unwrap_or_else(|| format!("{} (not in rope)", short(id)))).small());
    }

    fn small_thumb(&mut self, ui: &mut egui::Ui, id: &str, actions: &mut Vec<Action>) {
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(SMALL), Sense::click());
        let Some(item) = self.model.as_ref().and_then(|m| m.items.get(id)) else {
            ui.painter().rect_stroke(rect, 3.0, ui.visuals().widgets.inactive.bg_stroke, egui::StrokeKind::Inside);
            return;
        };
        let (file, fallback, name) = (item.file.clone(), item.fallback.clone(), item.name.clone());
        if let Some(tex) = self.thumbs.get(&Key { id: id.to_owned(), size: thumbs::GRID }, || Source { file, fallback })
        {
            egui::Image::new(&tex)
                .paint_at(ui, Rect::from_center_size(rect.center(), fit(tex.size_vec2(), rect.size())));
        }
        if resp.on_hover_text(name).clicked() {
            actions.push(Action::Open(id.to_owned()));
        }
    }

    fn bulk(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let ids = self.selected.clone();
        ui.heading(format!("{} images selected", ids.len()));
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Move to");
            egui::ComboBox::from_id_salt("bulk-folder").selected_text("choose a folder").show_ui(ui, |ui| {
                if ui.selectable_label(false, "/ (unfiled)").clicked() {
                    actions.push(Action::MoveTo(ids.clone(), None));
                }
                for (path, fid) in self.model().paths.clone() {
                    if ui.selectable_label(false, path).clicked() {
                        actions.push(Action::MoveTo(ids.clone(), Some(fid)));
                    }
                }
            });
        });
        ui.horizontal(|ui| {
            if ui.button("Add tag…").clicked() {
                actions.push(Action::Dialog(Dialog::Tag { ids: ids.clone(), name: String::new() }));
            }
            if self.view == View::Trash {
                if ui.button("Restore").clicked() {
                    actions.push(Action::Restore(ids.clone()));
                }
                if ui.button("Delete for good…").clicked() {
                    actions.push(Action::Dialog(Dialog::Purge { ids: ids.clone() }));
                }
            } else if ui.button("Move to trash").clicked() {
                actions.push(Action::Trash(ids.clone()));
            }
        });
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.dialog.take() else { return };
        let mut done = false;
        let mut run: Option<Deferred> = None;
        let enter = ctx.input(|i| i.key_pressed(egui::Key::Enter));
        let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        let model = self.model.as_ref().expect("loaded");
        egui::Modal::new(egui::Id::new("dialog")).show(ctx, |ui| {
            ui.set_width(380.0);
            match &mut dialog {
                Dialog::NewFolder { parent, name } => {
                    ui.heading("New folder");
                    if let Some(p) = parent {
                        ui.label(format!("in {}", model.catalog.folder_path(p)));
                    }
                    ui.text_edit_singleline(name).request_focus();
                    ui.label(RichText::new("a/b makes b inside a, and a too if it is missing").weak().small());
                    let path = match parent {
                        Some(p) => format!("{}/{}", model.catalog.folder_path(p), name.trim()),
                        None => name.trim().to_owned(),
                    };
                    if buttons(ui, "Create", enter && !name.trim().is_empty(), &mut done) {
                        run = Some(Box::new(move |app| app.new_folder(&path)));
                    }
                }
                Dialog::RenameFolder { id, name } => {
                    ui.heading("Rename folder");
                    ui.text_edit_singleline(name).request_focus();
                    let old = model.catalog.folder_path(id);
                    let parent = old.rsplit_once('/').map(|(p, _)| p.to_owned()).unwrap_or_default();
                    let new =
                        if parent.is_empty() { name.trim().to_owned() } else { format!("{parent}/{}", name.trim()) };
                    if buttons(ui, "Rename", enter && !name.trim().is_empty(), &mut done) {
                        run = Some(Box::new(move |app| {
                            store::edit_catalog(&app.lib, "mv", &[], |c| c.move_folder(&old, &new))?;
                            app.refresh();
                            Ok(())
                        }));
                    }
                }
                Dialog::DeleteFolder { id } => {
                    let path = model.catalog.folder_path(id);
                    let (folders, images) = model.catalog.tree_size(id);
                    ui.heading("Delete folder");
                    let below = match folders - 1 {
                        0 => String::new(),
                        n => format!(" and the {n} folder(s) inside it"),
                    };
                    ui.label(format!("Delete the folder {path}{below}?"));
                    if images > 0 {
                        ui.label(format!(
                            "Its {images} image(s) move to the trash. Restored from there, they come back unfiled; \
                             emptying the trash deletes them for good."
                        ));
                    }
                    if buttons(ui, "Delete", enter, &mut done) {
                        let id = id.clone();
                        run = Some(Box::new(move |app| app.delete_folder(&id)));
                    }
                }
                Dialog::Tag { ids, name } => {
                    ui.heading(format!("Tag {} image(s)", ids.len()));
                    ui.text_edit_singleline(name).request_focus();
                    if buttons(ui, "Add", enter && !name.trim().is_empty(), &mut done) {
                        let (ids, tag) = (ids.clone(), name.clone());
                        run = Some(Box::new(move |app| app.try_apply(Action::AddTag(ids, tag), None)));
                    }
                }
                Dialog::Purge { ids } => {
                    let gone: std::collections::HashSet<&String> = ids.iter().collect();
                    let kept = ids.iter().filter(|i| model.children(i).iter().any(|c| !gone.contains(c))).count();
                    ui.heading("Delete for good");
                    ui.label(format!(
                        "Delete {} image(s) for good: the files, their notes and their records.",
                        ids.len()
                    ));
                    if kept > 0 {
                        ui.label(format!(
                            "{kept} of them were used to make other images. Their bytes stay, so those can still be \
                                          replayed, until the images made from them are deleted too."
                        ));
                    }
                    if buttons(ui, "Delete for good", false, &mut done) {
                        let ids = ids.clone();
                        run = Some(Box::new(move |app| {
                            let (tx, rx) = channel();
                            app.tasks.push(rx);
                            let lib = app.lib.clone();
                            std::thread::spawn(move || {
                                let r = purge::purge(&lib, &ids, false).map_err(|e| format!("{e:#}")).map(|p| {
                                    format!(
                                        "deleted {} image(s); kept the bytes of {} for their descendants",
                                        p.deleted.len(),
                                        p.kept.len()
                                    )
                                });
                                let _ = tx.send(r);
                            });
                            Ok(())
                        }));
                    }
                }
            }
            if escape {
                done = true;
            }
        });
        if let Some(f) = run {
            done = true;
            if let Err(e) = f(self) {
                self.say(format!("{e:#}"), true);
            }
        }
        if !done {
            self.dialog = Some(dialog);
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        if self.dialog.is_some() || ctx.wants_keyboard_input() {
            return;
        }
        let (del, all, esc, copy, f5, arrows) = ctx.input(|i| {
            let dir = if i.key_pressed(egui::Key::ArrowRight) {
                1
            } else if i.key_pressed(egui::Key::ArrowLeft) {
                -1
            } else if i.key_pressed(egui::Key::ArrowDown) {
                self.cols as isize
            } else if i.key_pressed(egui::Key::ArrowUp) {
                -(self.cols as isize)
            } else {
                0
            };
            (
                // Backspace too: what an Apple keyboard labels delete sends Backspace
                i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                i.modifiers.command && i.key_pressed(egui::Key::A),
                i.key_pressed(egui::Key::Escape),
                i.events.iter().any(|e| matches!(e, egui::Event::Copy)),
                i.key_pressed(egui::Key::F5),
                dir,
            )
        });
        if del && !self.selected.is_empty() {
            let ids = self.selected.clone();
            actions.push(if self.view == View::Trash {
                Action::Dialog(Dialog::Purge { ids })
            } else {
                Action::Trash(ids)
            });
        }
        if all {
            actions.push(Action::SelectAll);
        }
        if esc {
            self.selected.clear();
        }
        if copy && let Some(id) = self.selected.last() {
            actions.push(Action::CopyPath(id.clone()));
        }
        if ctx.input(|i| i.key_pressed(egui::Key::L)) && !self.selected.is_empty() {
            actions.push(Action::ToggleFavorite(self.selected.clone()));
        }
        let look = ctx.input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space));
        if look && let Some(id) = self.selected.last() {
            actions.push(Action::Look(id.clone()));
        }
        if f5 {
            self.model = None;
            let (tx, rx) = channel();
            self.loading = Some(rx);
            let lib = self.lib.clone();
            let c = ctx.clone();
            std::thread::spawn(move || {
                let _ = tx.send(Model::load(&lib));
                c.request_repaint();
            });
        }
        if arrows != 0 && !self.visible.is_empty() {
            let cur = self.selected.last().and_then(|s| self.visible.iter().position(|v| v == s));
            let next = match cur {
                Some(i) => (i as isize + arrows).clamp(0, self.visible.len() as isize - 1) as usize,
                None => 0,
            };
            let id = self.visible[next].clone();
            actions.push(Action::Select(id.clone(), Pick::One));
            self.scroll_to = Some(id);
        }
    }

    fn title(&self) -> String {
        let model = self.model();
        match &self.view {
            View::All => "All images".into(),
            View::Unfiled => "Unfiled".into(),
            View::Folder(f) => model.catalog.folder_path(f),
            View::Tag(t) => format!("#{t}"),
            View::Trash => "Trash".into(),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame(ctx);
    }
}

impl App {
    /// Draws one frame and applies what was clicked in it.
    pub fn frame(&mut self, ctx: &egui::Context) {
        self.thumbs.poll(ctx);
        self.poll();
        let mut actions = Vec::new();
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some((text, error, at)) = &self.status {
                    let fresh = at.elapsed() < Duration::from_secs(8) || *error;
                    if fresh {
                        let color = if *error { Color32::from_rgb(220, 90, 70) } else { ui.visuals().text_color() };
                        ui.colored_label(color, text);
                        if !*error
                            && self.undo.is_some()
                            && at.elapsed() < Duration::from_secs(8)
                            && ui.small_button("Undo").clicked()
                        {
                            actions.push(Action::Undo);
                        }
                    }
                    if !fresh {
                        ui.ctx().request_repaint_after(Duration::from_secs(1));
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(self.lib.root().display().to_string()).weak().small());
                });
            });
        });
        if self.model.is_none() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    if self.loading.is_some() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("reading the library…");
                        });
                    } else {
                        ui.label("The library could not be read; see the message below.");
                    }
                });
            });
            return;
        }
        self.update_visible();
        if self.viewer.is_some() {
            self.viewer_frame(ctx, &mut actions);
            self.dialogs(ctx);
            for a in actions {
                self.apply(a, Some(ctx));
            }
            return;
        }
        self.shortcuts(ctx, &mut actions);
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Import…").on_hover_text("into the folder shown, or unfiled").clicked() {
                    actions.push(Action::Import { folders: false });
                }
                if ui
                    .button("Import folder…")
                    .on_hover_text("as a folder of the same name at the top, with its subfolders")
                    .clicked()
                {
                    actions.push(Action::Import { folders: true });
                }
                ui.separator();
                ui.add(
                    egui::TextEdit::singleline(&mut self.search)
                        .hint_text("search names, prompts, ids")
                        .desired_width(260.0),
                );
                ui.separator();
                ui.label(RichText::new(self.title()).strong());
                ui.label(RichText::new(format!("{} image(s)", self.visible.len())).weak());
            });
        });
        egui::SidePanel::left("folders")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| self.sidebar(ui, &mut actions));
        egui::SidePanel::right("details")
            .resizable(true)
            .default_width(380.0)
            .show(ctx, |ui| self.details(ui, &mut actions));
        egui::CentralPanel::default().show(ctx, |ui| self.grid(ui, &mut actions));
        self.dialogs(ctx);
        for a in actions {
            self.apply(a, Some(ctx));
        }
    }
}

/// A five-pointed star filling `rect`: filled gold for a favorite, else an outline, brighter when `hot`. Drawn
/// with lines rather than taken from a font, so it is crisp at this size on every machine.
pub(crate) fn paint_star(painter: &egui::Painter, rect: Rect, favorite: bool, hot: bool) {
    let c = rect.center();
    let outer = rect.width().min(rect.height()) / 2.0;
    let inner = outer * 0.42;
    let points: Vec<egui::Pos2> = (0..10)
        .map(|k| {
            let angle = -std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::PI / 5.0;
            let r = if k % 2 == 0 { outer } else { inner };
            egui::pos2(c.x + r * angle.cos(), c.y + r * angle.sin())
        })
        .collect();
    if favorite {
        let fill = Color32::from_rgb(250, 190, 40);
        // a fan from the middle: a star is star-shaped around its center
        let mut mesh = egui::Mesh::default();
        mesh.colored_vertex(c, fill);
        for p in &points {
            mesh.colored_vertex(*p, fill);
        }
        for k in 0..10u32 {
            mesh.add_triangle(0, 1 + k, 1 + (k + 1) % 10);
        }
        painter.add(egui::Shape::mesh(mesh));
        painter.add(egui::Shape::closed_line(points, egui::Stroke::new(1.0, Color32::from_rgb(140, 95, 10))));
    } else {
        // an outline that reads on a light image as well as a dark one
        painter.add(egui::Shape::closed_line(points.clone(), egui::Stroke::new(2.5, Color32::from_black_alpha(110))));
        let edge = if hot { Color32::WHITE } else { Color32::from_white_alpha(190) };
        painter.add(egui::Shape::closed_line(points, egui::Stroke::new(1.2, edge)));
    }
}

/// A clickable star for marking a favorite.
fn star_button(ui: &mut egui::Ui, favorite: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(22.0), Sense::click());
    paint_star(ui.painter(), rect.shrink(2.0), favorite, resp.hovered());
    resp.on_hover_text(if favorite { "remove from favorites (L)" } else { "add to favorites (L)" })
}

/// OK and Cancel; returns whether OK was pressed (or `enter_ok`). Cancel sets `done`.
fn buttons(ui: &mut egui::Ui, ok: &str, enter_ok: bool, done: &mut bool) -> bool {
    let mut pressed = enter_ok;
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        pressed |= ui.button(ok).clicked();
        if ui.button("Cancel").clicked() {
            *done = true;
        }
    });
    pressed
}

/// `size` scaled to fit in `area`, keeping its aspect ratio.
fn fit(size: Vec2, area: Vec2) -> Vec2 {
    let scale = (area.x / size.x).min(area.y / size.y);
    size * scale
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> =
            std::fs::read_dir(path).into_iter().flatten().flatten().map(|e| e.path()).collect();
        entries.sort();
        for p in entries {
            if !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
                collect(&p, out);
            }
        }
    } else if store::looks_like_image(path) {
        out.push(path.to_path_buf());
    }
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rope_core::store::NewImage;
    use serde_json::json;

    fn png(dir: &Path, name: &str, seed: u8) -> PathBuf {
        let path = dir.join(name);
        image::RgbImage::from_fn(300, 200, |x, y| image::Rgb([seed, x as u8, y as u8])).save(&path).unwrap();
        path
    }

    /// Draws frames until `done` holds, or fails after a few seconds.
    fn frames_until(ctx: &egui::Context, app: &mut App, done: impl Fn(&App) -> bool) {
        for _ in 0..500 {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(1400.0, 900.0))),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| app.frame(ctx));
            if done(app) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the window never got there");
    }

    /// Draws one frame in which `key` is pressed.
    fn press(ctx: &egui::Context, app: &mut App, key: egui::Key) {
        let event = egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(1400.0, 900.0))),
            events: vec![event],
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| app.frame(ctx));
    }

    #[test]
    fn the_large_view_steps_zooms_trashes_and_goes_back() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(&dir.path().join("lib")).unwrap();
        let new = || NewImage { via: "import".into(), ..Default::default() };
        // three images, the newest first in the grid: c, b, a
        let ids: Vec<String> =
            (1..=3).map(|i| store::add(&lib, &png(dir.path(), &format!("{i}.png"), i), &new()).unwrap().sha).collect();
        let (a, b, c) = (ids[0].clone(), ids[1].clone(), ids[2].clone());
        let ctx = egui::Context::default();
        let mut app = App::new(&ctx, lib.clone(), None);
        frames_until(&ctx, &mut app, |a| a.model.is_some() && a.visible.len() == 3);
        let order = app.visible.clone();

        // Enter on the selection opens it large, and the full-size texture arrives
        app.apply(Action::Select(order[0].clone(), Pick::One), Some(&ctx));
        press(&ctx, &mut app, egui::Key::Enter);
        assert_eq!(app.viewer.as_ref().map(|v| v.id.clone()), Some(order[0].clone()));
        frames_until(&ctx, &mut app, |a| a.thumbs.ready(&Key { id: order[0].clone(), size: thumbs::VIEW }));

        // arrows step through the view, and stop at its ends
        press(&ctx, &mut app, egui::Key::ArrowRight);
        assert_eq!(app.viewer.as_ref().unwrap().id, order[1]);
        press(&ctx, &mut app, egui::Key::ArrowLeft);
        press(&ctx, &mut app, egui::Key::ArrowLeft);
        assert_eq!(app.viewer.as_ref().unwrap().id, order[0]);

        // 1 is 100%, + zooms in, F fits again; I shows the details next to the image
        press(&ctx, &mut app, egui::Key::Num1);
        assert_eq!(app.viewer.as_ref().unwrap().zoom, Some(1.0 / ctx.pixels_per_point()));
        press(&ctx, &mut app, egui::Key::Plus);
        assert!(app.viewer.as_ref().unwrap().zoom.unwrap() > 1.0 / ctx.pixels_per_point());
        press(&ctx, &mut app, egui::Key::F);
        assert_eq!(app.viewer.as_ref().unwrap().zoom, None);
        press(&ctx, &mut app, egui::Key::I);
        frames_until(&ctx, &mut app, |a| a.viewer.as_ref().is_some_and(|v| v.info));

        // Delete trashes the image and goes on to the next one
        press(&ctx, &mut app, egui::Key::Delete);
        assert!(rope_core::Catalog::load(&lib).unwrap().is_trashed(&order[0]));
        assert_eq!(app.viewer.as_ref().unwrap().id, order[1]);
        frames_until(&ctx, &mut app, |a| a.visible.len() == 2);

        // Backspace does the same, for keyboards whose delete key sends it (Apple's)
        press(&ctx, &mut app, egui::Key::Backspace);
        assert!(rope_core::Catalog::load(&lib).unwrap().is_trashed(&order[1]));
        assert_eq!(app.viewer.as_ref().unwrap().id, order[2]);
        frames_until(&ctx, &mut app, |a| a.visible.len() == 1);

        // Esc goes back to the grid, on the image last looked at
        press(&ctx, &mut app, egui::Key::Escape);
        assert!(app.viewer.is_none());
        assert_eq!(app.selected, [order[2].clone()]);

        // in the grid, Backspace trashes the selection
        press(&ctx, &mut app, egui::Key::Backspace);
        assert!(rope_core::Catalog::load(&lib).unwrap().is_trashed(&order[2]));
        let _ = (a, b, c);
    }

    #[test]
    fn favorites_come_first_and_the_large_view_keeps_its_order_when_one_is_marked() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(&dir.path().join("lib")).unwrap();
        let new = || NewImage { via: "import".into(), ..Default::default() };
        for i in 1..=3 {
            store::add(&lib, &png(dir.path(), &format!("{i}.png"), i), &new()).unwrap();
        }
        let ctx = egui::Context::default();
        let mut app = App::new(&ctx, lib.clone(), None);
        frames_until(&ctx, &mut app, |a| a.model.is_some() && a.visible.len() == 3);
        let order = app.visible.clone();

        // L in the grid marks the selection; the favorite moves to the front of the view
        app.apply(Action::Select(order[2].clone(), Pick::One), Some(&ctx));
        press(&ctx, &mut app, egui::Key::L);
        assert!(rope_core::Catalog::load(&lib).unwrap().is_favorite(&order[2]));
        frames_until(&ctx, &mut app, |a| a.visible[0] == order[2]);
        assert_eq!(app.visible, [order[2].clone(), order[0].clone(), order[1].clone()]);

        // in the large view, marking a favorite does not change which image comes next
        let now = app.visible.clone();
        app.apply(Action::Look(now[1].clone()), Some(&ctx));
        press(&ctx, &mut app, egui::Key::L);
        frames_until(&ctx, &mut app, |a| a.visible[1] == now[1] && a.visible[0] == now[0] || a.visible[0] == now[1]);
        assert!(rope_core::Catalog::load(&lib).unwrap().is_favorite(&now[1]));
        press(&ctx, &mut app, egui::Key::ArrowRight);
        assert_eq!(app.viewer.as_ref().unwrap().id, now[2], "the next one in the order the viewer opened with");

        // L again unmarks; the menu's toggle unmarks a whole selection when all are favorites
        press(&ctx, &mut app, egui::Key::Escape);
        app.apply(Action::ToggleFavorite(vec![now[0].clone(), now[1].clone()]), Some(&ctx));
        let c = rope_core::Catalog::load(&lib).unwrap();
        assert!(!c.is_favorite(&now[0]) && !c.is_favorite(&now[1]));
    }

    #[test]
    fn the_window_draws_every_view_and_its_actions_reach_the_library() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(&dir.path().join("lib")).unwrap();
        let folder = store::edit_catalog(&lib, "mkdir", &[], |c| c.mkdir("trips/京都", true)).unwrap();
        let photo = store::add(
            &lib,
            &png(dir.path(), "photo.png", 1),
            &NewImage {
                via: "import".into(),
                folder: Some(Some(folder.clone())),
                tags: vec!["海邊".into()],
                ..Default::default()
            },
        )
        .unwrap()
        .sha;
        let made = NewImage {
            name: Some("result".into()),
            via: "img2img.py".into(),
            made: Some(
                json!({"flags": ["--model=2511", "--seed=1"], "prompt": "把背景換成海邊", "inputs": [{"id": photo}]}),
            ),
            ..Default::default()
        };
        let result = store::add(&lib, &png(dir.path(), "result.png", 2), &made).unwrap().sha;

        let ctx = egui::Context::default();
        let mut app = App::new(&ctx, lib.clone(), Some(result[..8].to_owned()));
        frames_until(&ctx, &mut app, |a| a.detail.as_ref().is_some_and(|d| d.id == result));
        assert_eq!(app.view, View::Unfiled, "opening an image shows its folder");
        // the grid thumbnail and the preview get decoded and cached
        frames_until(&ctx, &mut app, |a| {
            [thumbs::PREVIEW, thumbs::GRID].iter().all(|&size| a.thumbs.ready(&Key { id: result.clone(), size }))
        });
        assert!(lib.cache_dir().join("thumbs").join(&result[..2]).join(format!("{result}.jpg")).exists());

        // upstream link: opening the photo switches to its folder
        app.apply(Action::Open(photo.clone()), Some(&ctx));
        assert_eq!(app.view, View::Folder(folder.clone()));
        frames_until(&ctx, &mut app, |a| a.visible == [photo.clone()]);

        // every view draws
        for v in [View::All, View::Tag("海邊".into()), View::Trash, View::Unfiled, View::Folder(folder.clone())] {
            app.apply(Action::View(v), Some(&ctx));
            frames_until(&ctx, &mut app, |_| true);
        }

        // note, name, tags, folder moves, trash and undo, all through the library
        app.apply(Action::Select(result.clone(), Pick::One), Some(&ctx));
        app.detail.as_mut().unwrap().note = "第一行\n".into();
        app.apply(Action::SaveNote { force: false }, Some(&ctx));
        assert_eq!(store::note(&lib, &result).unwrap(), "第一行\n");
        app.detail.as_mut().unwrap().name = "renamed".into();
        app.apply(Action::SaveName, Some(&ctx));
        assert_eq!(store::load(&lib, &result).unwrap().name, "renamed");
        app.apply(Action::AddTag(vec![result.clone()], "夕陽".into()), Some(&ctx));
        app.apply(Action::MoveTo(vec![result.clone()], Some(folder.clone())), Some(&ctx));
        let c = rope_core::Catalog::load(&lib).unwrap();
        assert_eq!(c.entry(&result).tags, ["夕陽"]);
        assert_eq!(c.entry(&result).folder.as_deref(), Some(folder.as_str()));
        app.apply(Action::Trash(vec![result.clone()]), Some(&ctx));
        assert!(rope_core::Catalog::load(&lib).unwrap().is_trashed(&result));
        app.apply(Action::Undo, Some(&ctx));
        assert!(!rope_core::Catalog::load(&lib).unwrap().is_trashed(&result));

        // a note changed elsewhere while being edited is not overwritten silently
        app.apply(Action::Select(result.clone(), Pick::One), Some(&ctx));
        app.detail.as_mut().unwrap().note = "mine\n".into();
        std::thread::sleep(Duration::from_millis(20));
        store::set_note(&lib, &result, "theirs\n").unwrap();
        app.apply(Action::SaveNote { force: false }, Some(&ctx));
        assert!(app.detail.as_ref().unwrap().note_conflict);
        assert_eq!(store::note(&lib, &result).unwrap(), "theirs\n");

        // dialogs draw
        for d in [
            Dialog::NewFolder { parent: Some(folder.clone()), name: "x".into() },
            Dialog::RenameFolder { id: folder.clone(), name: "y".into() },
            Dialog::DeleteFolder { id: folder.clone() },
            Dialog::Tag { ids: vec![result.clone()], name: String::new() },
            Dialog::Purge { ids: vec![photo.clone()] },
        ] {
            app.dialog = Some(d);
            frames_until(&ctx, &mut app, |_| true);
            app.dialog = None;
        }

        // an image added by someone else (the CLI, img2img.py) shows up through the journal
        let other = store::add(
            &lib,
            &png(dir.path(), "other.png", 3),
            &NewImage { via: "import".into(), ..Default::default() },
        )
        .unwrap()
        .sha;
        app.apply(Action::View(View::All), Some(&ctx));
        frames_until(&ctx, &mut app, |a| a.visible.contains(&other));

        // Import folder: a rope folder of the same name at the top, whatever folder is shown
        let disk = dir.path().join("北海道");
        std::fs::create_dir_all(disk.join("day 2")).unwrap();
        png(&disk, "a.png", 7);
        png(&disk.join("day 2"), "b.png", 8);
        app.apply(Action::View(View::Folder(folder.clone())), Some(&ctx));
        app.import_paths(vec![disk.clone()], true, Some(&ctx));
        frames_until(&ctx, &mut app, |a| a.model().catalog.find("北海道/day 2").is_some_and(|f| f.is_some()));
        let c = rope_core::Catalog::load(&lib).unwrap();
        assert!(c.find("北海道").is_some() && c.find("trips/京都/北海道").is_none());

        // Clean: the trash's menu asks to purge everything in it
        app.apply(Action::Trash(vec![other.clone()]), Some(&ctx));
        assert_eq!(app.trashed(), vec![other.clone()]);
        app.apply(Action::Restore(vec![other.clone()]), Some(&ctx));
        app.apply(Action::View(View::All), Some(&ctx));
        frames_until(&ctx, &mut app, |a| a.visible.len() == 5);

        // New folder: a path makes the missing folders on the way, inside an existing one too
        app.new_folder("旅行/2026/春").unwrap();
        app.new_folder("trips/京都/day 1/morning").unwrap();
        let c = rope_core::Catalog::load(&lib).unwrap();
        for path in ["旅行", "旅行/2026", "旅行/2026/春", "trips/京都/day 1", "trips/京都/day 1/morning"] {
            assert!(c.find(path).is_some_and(|f| f.is_some()), "{path}");
        }
        assert_eq!(c.find("trips/京都"), Some(Some(folder.clone())), "an existing folder is reused");
        assert!(app.new_folder("旅行/2026").unwrap_err().to_string().contains("exists"));
        assert!(app.new_folder(" / ").is_err());

        // deleting a folder with what is in it: its subfolders go, its images go to the trash, the view moves on
        let gone = app.model().catalog.find("旅行").unwrap().unwrap();
        let inside = store::add(
            &lib,
            &png(dir.path(), "in.png", 9),
            &NewImage {
                via: "import".into(),
                folder: Some(app.model().catalog.find("旅行/2026/春").unwrap()),
                ..Default::default()
            },
        )
        .unwrap()
        .sha;
        app.refresh();
        app.apply(Action::View(View::Folder(app.model().catalog.find("旅行/2026").unwrap().unwrap())), Some(&ctx));
        app.delete_folder(&gone).unwrap();
        let c = rope_core::Catalog::load(&lib).unwrap();
        assert!(c.find("旅行").is_none() && c.find("旅行/2026/春").is_none());
        assert!(c.is_trashed(&inside) && c.entry(&inside).folder.is_none());
        assert_eq!(app.view, View::All);
        app.apply(Action::Restore(vec![inside.clone()]), Some(&ctx));
        assert_eq!(rope_core::Catalog::load(&lib).unwrap().entry(&inside), Default::default(), "restored: unfiled");

        // a folder's Copy path gives what a command takes
        assert_eq!(app.folder_arg(&folder), "trips/京都");
        let day = app.model().catalog.find("trips/京都/day 1").unwrap().unwrap();
        assert_eq!(app.folder_arg(&day), "'trips/京都/day 1'");
        app.apply(Action::CopyFolder(day), Some(&ctx));
        assert_eq!(app.status.as_ref().map(|s| s.0.as_str()), Some("copied the folder path"));

        // several selected: the bulk panel draws
        app.apply(Action::SelectAll, Some(&ctx));
        frames_until(&ctx, &mut app, |_| true);
        assert_eq!(app.selected.len(), 5);
    }
}
