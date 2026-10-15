//! The Prompts page: prompts worth keeping, each with a title and its text. The list is in the middle and the
//! editor on the right, as images and their details are.

use super::{Action, App, Dialog};
use eframe::egui::{self, RichText, Sense};
use rope_core::prompts::{self, Prompt};

/// The prompt being edited: an existing one, or a new one until it is saved.
pub(super) struct PromptEdit {
    pub id: Option<String>,
    pub title: String,
    pub text: String,
    /// What is saved, to know when the editor is dirty.
    pub saved: (String, String),
}

impl PromptEdit {
    pub fn dirty(&self) -> bool {
        (self.title.as_str(), self.text.as_str()) != (self.saved.0.as_str(), self.saved.1.as_str())
    }
}

impl App {
    /// Opens the editor on the prompt `id`, or, without one, on a new prompt with `title` and `text`.
    pub(super) fn edit_prompt(&mut self, id: Option<String>, title: String, text: String) {
        let (title, text, saved) = match id.as_deref().and_then(|i| self.model().prompts.get(i)) {
            Some(p) => (p.title.clone(), p.text.clone(), (p.title.clone(), p.text.clone())),
            None => (title, text, (String::new(), String::new())),
        };
        self.prompt = Some(PromptEdit { id, title, text, saved });
    }

    /// Writes the editor's prompt: a new one is added, an existing one changed.
    pub(super) fn save_prompt(&mut self) -> anyhow::Result<()> {
        let Some(e) = self.prompt.as_mut() else { return Ok(()) };
        let p = match &e.id {
            Some(id) => prompts::edit(&self.lib, id, Some(&e.title), Some(&e.text))?,
            None => prompts::add(&self.lib, &e.title, &e.text)?,
        };
        *e = PromptEdit { id: Some(p.id), title: p.title.clone(), text: p.text.clone(), saved: (p.title, p.text) };
        self.refresh();
        self.say("prompt saved", false);
        Ok(())
    }

    pub(super) fn delete_prompt(&mut self, id: &str) -> anyhow::Result<()> {
        let p = prompts::remove(&self.lib, id)?;
        if self.prompt.as_ref().is_some_and(|e| e.id.as_deref() == Some(id)) {
            self.prompt = None;
        }
        self.refresh();
        self.say(format!("removed the prompt {:?}", p.title), false);
        Ok(())
    }

    /// After a change from elsewhere: a prompt deleted there leaves the editor, one changed there is taken when
    /// nothing was typed here yet.
    pub(super) fn follow_prompts(&mut self) {
        let Some(e) = self.prompt.as_mut() else { return };
        let Some(id) = &e.id else { return };
        match self.model.as_ref().expect("loaded").prompts.get(id) {
            None => self.prompt = None,
            Some(p) if !e.dirty() && (p.title != e.title || p.text != e.text) => {
                e.title = p.title.clone();
                e.text = p.text.clone();
                e.saved = (p.title.clone(), p.text.clone());
            }
            Some(_) => {}
        }
    }

    /// The prompts that match the search box, in the order they were added.
    pub(super) fn prompts_shown(&self) -> Vec<Prompt> {
        let search = self.search.trim().to_lowercase();
        self.model()
            .prompts
            .prompts
            .iter()
            .filter(|p| {
                search.is_empty() || p.title.to_lowercase().contains(&search) || p.text.to_lowercase().contains(&search)
            })
            .cloned()
            .collect()
    }

    /// The middle of the window on the Prompts page: the list.
    pub(super) fn prompt_list(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("New prompt").clicked() {
                actions.push(Action::EditPrompt(None));
            }
            ui.label(
                RichText::new("Prompts to use again. Copy one into the next img2img.py command, or in a shell: \"$(rope prompt show TITLE)\"")
                    .weak()
                    .small(),
            );
        });
        ui.add_space(4.0);
        let shown = self.prompts_shown();
        if shown.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new(if self.search.trim().is_empty() { "No prompts yet" } else { "Nothing matches" })
                        .weak(),
                );
            });
            return;
        }
        let current = self.prompt.as_ref().and_then(|e| e.id.clone());
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for p in &shown {
                let selected = current.as_deref() == Some(p.id.as_str());
                let fill = if selected {
                    ui.visuals().selection.bg_fill.gamma_multiply(0.35)
                } else {
                    ui.visuals().faint_bg_color
                };
                let resp = egui::Frame::group(ui.style())
                    .fill(fill)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&p.title).strong());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("Copy").on_hover_text("copy the text").clicked() {
                                    actions.push(Action::CopyText(p.text.clone()));
                                }
                            });
                        });
                        ui.add(egui::Label::new(RichText::new(&p.text).small()).wrap());
                    })
                    .response;
                if resp.interact(Sense::click()).clicked() {
                    actions.push(Action::EditPrompt(Some(p.id.clone())));
                }
                ui.add_space(4.0);
            }
        });
    }

    /// The right side on the Prompts page: the editor.
    pub(super) fn prompt_editor(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let Some(e) = self.prompt.as_mut() else {
            ui.centered_and_justified(|ui| ui.label(RichText::new("Select a prompt, or make a new one").weak()));
            return;
        };
        let dirty = e.dirty();
        ui.heading(if e.id.is_some() { "Prompt" } else { "New prompt" });
        ui.add_space(4.0);
        ui.label("Title");
        let title =
            ui.add(egui::TextEdit::singleline(&mut e.title).desired_width(f32::INFINITY).hint_text("e.g. 黃昏海邊"));
        if e.id.is_none() && e.title.is_empty() && e.text.is_empty() {
            title.request_focus();
        }
        ui.add_space(4.0);
        ui.label("Text");
        ui.add(
            egui::TextEdit::multiline(&mut e.text)
                .desired_rows(10)
                .desired_width(f32::INFINITY)
                .hint_text("the prompt"),
        );
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(dirty && !e.title.trim().is_empty(), egui::Button::new("Save")).clicked() {
                actions.push(Action::SavePrompt);
            }
            if dirty && ui.button("Revert").clicked() {
                (e.title, e.text) = e.saved.clone();
            }
            if ui.button("Copy").on_hover_text("copy the text").clicked() {
                actions.push(Action::CopyText(e.text.clone()));
            }
            if let Some(id) = &e.id
                && ui.button("Delete…").clicked()
            {
                actions.push(Action::Dialog(Dialog::DeletePrompt { id: id.clone() }));
            }
            if ui.button("Close").clicked() {
                actions.push(Action::ClosePrompt);
            }
        });
        if dirty {
            ui.label(RichText::new("not saved yet").weak().small());
        }
    }
}
