//! The toolbar "File" menu: save/load the scene document through platform
//! file dialogs (rfd).
//!
//! Native uses blocking OS dialogs, so every action resolves within the
//! click's frame. The web has no save dialog — saving is a browser download
//! with a user-typed filename — and its file picker is asynchronous, so a
//! picked file is parked in [`FileMenu::picked`] and reported on a later
//! frame. Both paths surface the outcome as a [`FileEvent`]; the app applies
//! it without knowing which platform produced it.

use caso_kernel::scene::SceneDocument;
use eframe::egui;

#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

/// A file delivered by the async browser picker: (filename, contents).
#[cfg(target_arch = "wasm32")]
type PickedFile = Rc<RefCell<Option<(String, Vec<u8>)>>>;

/// Outcome of a File-menu action, ready for the app to apply.
pub enum FileEvent {
    /// A scene file was read and parsed successfully.
    Loaded {
        name: String,
        document: SceneDocument,
    },
    /// A save finished or an action failed — a status-bar message.
    Status(String),
}

#[derive(Default)]
pub struct FileMenu {
    /// Filename for the browser download (empty means `scene.json`).
    #[cfg(target_arch = "wasm32")]
    download_name: String,
    /// File handed back by the async browser picker, consumed on the next
    /// frame (wasm is single-threaded, so `Rc<RefCell>` suffices).
    #[cfg(target_arch = "wasm32")]
    picked: PickedFile,
}

impl FileMenu {
    /// Renders the menu button and returns at most one event per frame —
    /// either from a click this frame or from a web pick that completed
    /// earlier.
    pub fn ui(&mut self, ui: &mut egui::Ui, document: &SceneDocument) -> Option<FileEvent> {
        let mut event = None;
        ui.menu_button("File", |ui| event = self.menu_contents(ui, document));
        event.or_else(|| self.take_picked())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn menu_contents(
        &mut self,
        ui: &mut egui::Ui,
        document: &SceneDocument,
    ) -> Option<FileEvent> {
        let mut event = None;
        if ui.button("Save As…").clicked() {
            ui.close();
            if let Some(path) = scene_dialog().set_file_name("scene.json").save_file() {
                let result = serialize(document).and_then(|text| {
                    std::fs::write(&path, text).map_err(|error| error.to_string())
                });
                event = Some(FileEvent::Status(match result {
                    Ok(()) => format!("Saved {}", path.display()),
                    Err(error) => format!("Save failed: {error}"),
                }));
            }
        }
        if ui.button("Open…").clicked() {
            ui.close();
            if let Some(path) = scene_dialog().pick_file() {
                event = Some(match std::fs::read(&path) {
                    Ok(bytes) => parse_scene(path.display().to_string(), &bytes),
                    Err(error) => FileEvent::Status(format!("Load failed: {error}")),
                });
            }
        }
        event
    }

    #[cfg(target_arch = "wasm32")]
    fn menu_contents(
        &mut self,
        ui: &mut egui::Ui,
        document: &SceneDocument,
    ) -> Option<FileEvent> {
        let mut event = None;
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.add(
                egui::TextEdit::singleline(&mut self.download_name)
                    .desired_width(140.0)
                    .hint_text("scene.json"),
            );
        });
        if ui.button("Save (download)").clicked() {
            ui.close();
            let name = download_name(&self.download_name);
            let result = serialize(document).and_then(|text| {
                crate::web_download_bytes(&name, text.as_bytes())
                    .map_err(|error| format!("{error:?}"))
            });
            event = Some(FileEvent::Status(match result {
                Ok(()) => format!("Downloaded {name}"),
                Err(error) => format!("Save failed: {error}"),
            }));
        }
        if ui.button("Open…").clicked() {
            ui.close();
            let picked = self.picked.clone();
            let ctx = ui.ctx().clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(file) = rfd::AsyncFileDialog::new()
                    .add_filter("Scene JSON", &["json"])
                    .pick_file()
                    .await
                {
                    let bytes = file.read().await;
                    *picked.borrow_mut() = Some((file.file_name(), bytes));
                    ctx.request_repaint();
                }
            });
        }
        event
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn take_picked(&mut self) -> Option<FileEvent> {
        None
    }

    #[cfg(target_arch = "wasm32")]
    fn take_picked(&mut self) -> Option<FileEvent> {
        let (name, bytes) = self.picked.borrow_mut().take()?;
        Some(parse_scene(name, &bytes))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_dialog() -> rfd::FileDialog {
    rfd::FileDialog::new().add_filter("Scene JSON", &["json"])
}

/// The download filename: trimmed user entry with `.json` appended when
/// missing; `scene.json` when empty.
#[cfg(target_arch = "wasm32")]
fn download_name(raw: &str) -> String {
    let name = raw.trim();
    if name.is_empty() {
        "scene.json".to_string()
    } else if name.ends_with(".json") {
        name.to_string()
    } else {
        format!("{name}.json")
    }
}

fn serialize(document: &SceneDocument) -> Result<String, String> {
    caso_kernel::serialization::save_scene_to_string(document).map_err(|error| error.to_string())
}

fn parse_scene(name: String, bytes: &[u8]) -> FileEvent {
    let result = std::str::from_utf8(bytes)
        .map_err(|error| error.to_string())
        .and_then(|text| {
            caso_kernel::serialization::load_scene_from_str(text)
                .map_err(|error| error.to_string())
        });
    match result {
        Ok(document) => FileEvent::Loaded { name, document },
        Err(error) => FileEvent::Status(format!("Load failed ({name}): {error}")),
    }
}
