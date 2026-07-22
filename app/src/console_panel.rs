//! Console Draw editor, output pane, and `.rhai` file workflow.

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use eframe::egui;

use crate::console_draw_runner::{run_console_draw, EXAMPLE_SCRIPT};
use crate::state::AppState;

#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

const MAX_SCRIPT_BYTES: usize = 1024 * 1024;

struct LoadedScript {
    name: String,
    source: String,
}

#[cfg(target_arch = "wasm32")]
enum PickedScript {
    Cancelled,
    File(String, Vec<u8>),
}

#[cfg(target_arch = "wasm32")]
type PickedFile = Rc<RefCell<Option<PickedScript>>>;

pub struct ConsolePanel {
    script: String,
    saved_script: String,
    filename: String,
    output: String,
    confirm_open: bool,
    pending_script: Option<LoadedScript>,
    #[cfg(target_arch = "wasm32")]
    picked: PickedFile,
    #[cfg(target_arch = "wasm32")]
    opening: bool,
    #[cfg(target_arch = "wasm32")]
    open_source: String,
}

impl Default for ConsolePanel {
    fn default() -> Self {
        let script = EXAMPLE_SCRIPT.to_string();
        Self {
            saved_script: script.clone(),
            script,
            filename: "console_draw.rhai".to_string(),
            output: String::new(),
            confirm_open: false,
            pending_script: None,
            #[cfg(target_arch = "wasm32")]
            picked: PickedFile::default(),
            #[cfg(target_arch = "wasm32")]
            opening: false,
            #[cfg(target_arch = "wasm32")]
            open_source: String::new(),
        }
    }
}

impl ConsolePanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        self.take_picked();
        let mut run = false;
        ui.horizontal_wrapped(|ui| {
            run |= ui.button("Run").clicked();
            if ui.button("Open .rhai…").clicked() {
                if self.dirty() {
                    self.confirm_open = true;
                } else {
                    self.begin_open(ui);
                }
            }
            if ui.button("Save As .rhai…").clicked() {
                self.save_as(state);
            }
            if ui.button("Clear Output").clicked() {
                self.output.clear();
            }
        });

        #[cfg(target_arch = "wasm32")]
        ui.horizontal(|ui| {
            ui.label("File");
            ui.add(
                egui::TextEdit::singleline(&mut self.filename)
                    .desired_width(150.0)
                    .hint_text("console_draw.rhai"),
            );
            if self.dirty() {
                ui.strong("*");
            }
            if self.opening {
                ui.spinner();
                ui.weak("Opening…");
            }
        });
        #[cfg(not(target_arch = "wasm32"))]
        ui.label(format!(
            "File: {}{}",
            self.filename,
            if self.dirty() { " *" } else { "" }
        ));

        if self.confirm_open {
            ui.group(|ui| {
                ui.label("Discard unsaved console changes and open another script?");
                ui.horizontal(|ui| {
                    if ui.button("Discard and Open").clicked() {
                        self.confirm_open = false;
                        if let Some(script) = self.pending_script.take() {
                            self.install(script);
                        } else {
                            self.begin_open(ui);
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm_open = false;
                        self.pending_script = None;
                    }
                });
            });
        }

        ui.separator();
        ui.label("Code (Ctrl+Enter to run)");
        let editor = ui.add_sized(
            [ui.available_width(), 300.0],
            egui::TextEdit::multiline(&mut self.script)
                .code_editor()
                .desired_width(f32::INFINITY),
        );
        run |= editor.has_focus()
            && ui.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::Enter));
        if run {
            self.run(state);
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Output");
            if self.output.is_empty() {
                ui.weak("(empty)");
            }
        });
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(&self.output).monospace())
                        .wrap()
                        .selectable(true),
                );
            });
    }

    fn dirty(&self) -> bool {
        self.script != self.saved_script
    }

    fn run(&mut self, state: &mut AppState) {
        match run_console_draw(state, &self.script) {
            Ok(result) => {
                self.output = if result.output.is_empty() {
                    "Run completed.".to_string()
                } else {
                    result.output
                };
                state.status = if result.mutated {
                    "Console Draw: committed as one undo step".to_string()
                } else {
                    "Console Draw: completed (scene unchanged)".to_string()
                };
            }
            Err(failure) => {
                self.output = failure.display_output();
                state.status = "Console Draw: discarded after error".to_string();
            }
        }
    }

    fn install(&mut self, script: LoadedScript) {
        self.filename = script.name;
        self.saved_script = script.source.clone();
        self.script = script.source;
        self.output = "Script opened.".to_string();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn begin_open(&mut self, _ui: &egui::Ui) {
        let Some(path) = script_dialog().pick_file() else {
            return;
        };
        match std::fs::read(&path)
            .map_err(|failure| failure.to_string())
            .and_then(|bytes| parse_script(display_name(&path), &bytes))
        {
            Ok(script) => self.install(script),
            Err(failure) => self.output = format!("Open failed: {failure}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn begin_open(&mut self, ui: &egui::Ui) {
        if self.opening {
            return;
        }
        self.opening = true;
        self.open_source = self.script.clone();
        let picked = self.picked.clone();
        let context = ui.ctx().clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = match rfd::AsyncFileDialog::new()
                .add_filter("Rhai script", &["rhai"])
                .pick_file()
                .await
            {
                Some(file) => PickedScript::File(file.file_name(), file.read().await),
                None => PickedScript::Cancelled,
            };
            *picked.borrow_mut() = Some(result);
            context.request_repaint();
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_as(&mut self, state: &mut AppState) {
        let Some(path) = script_dialog().set_file_name(&self.filename).save_file() else {
            return;
        };
        let path = ensure_rhai_path(path);
        match std::fs::write(&path, self.script.as_bytes()) {
            Ok(()) => {
                self.filename = display_name(&path);
                self.saved_script = self.script.clone();
                state.status = format!("Saved {}", path.display());
            }
            Err(failure) => self.output = format!("Save failed: {failure}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn save_as(&mut self, state: &mut AppState) {
        let name = rhai_filename(&self.filename);
        match crate::web_download_bytes(&name, self.script.as_bytes()) {
            Ok(()) => {
                self.filename = name.clone();
                self.saved_script = self.script.clone();
                state.status = format!("Downloaded {name}");
            }
            Err(failure) => self.output = format!("Save failed: {failure:?}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn take_picked(&mut self) {}

    #[cfg(target_arch = "wasm32")]
    fn take_picked(&mut self) {
        let Some(picked) = self.picked.borrow_mut().take() else {
            return;
        };
        self.opening = false;
        let PickedScript::File(name, bytes) = picked else {
            return;
        };
        let script = match parse_script(name, &bytes) {
            Ok(script) => script,
            Err(failure) => {
                self.output = format!("Open failed: {failure}");
                return;
            }
        };
        if self.script == self.open_source {
            self.install(script);
        } else {
            self.pending_script = Some(script);
            self.confirm_open = true;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn script_dialog() -> rfd::FileDialog {
    rfd::FileDialog::new().add_filter("Rhai script", &["rhai"])
}

fn parse_script(name: String, bytes: &[u8]) -> Result<LoadedScript, String> {
    if bytes.len() > MAX_SCRIPT_BYTES {
        return Err("script is larger than 1 MiB".to_string());
    }
    let source = std::str::from_utf8(bytes)
        .map_err(|_| "script is not valid UTF-8".to_string())?
        .to_string();
    Ok(LoadedScript { name, source })
}

#[cfg(any(target_arch = "wasm32", test))]
fn rhai_filename(raw: &str) -> String {
    let name = raw.trim();
    if name.is_empty() {
        "console_draw.rhai".to_string()
    } else if name.ends_with(".rhai") {
        name.to_string()
    } else {
        format!("{name}.rhai")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_rhai_path(mut path: PathBuf) -> PathBuf {
    if path.extension().and_then(|extension| extension.to_str()) != Some("rhai") {
        path.set_extension("rhai");
    }
    path
}

#[cfg(not(target_arch = "wasm32"))]
fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("console_draw.rhai")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filenames_get_the_rhai_extension() {
        assert_eq!(rhai_filename(""), "console_draw.rhai");
        assert_eq!(rhai_filename("part"), "part.rhai");
        assert_eq!(rhai_filename("part.rhai"), "part.rhai");
        assert_eq!(
            ensure_rhai_path(PathBuf::from("part.txt")),
            PathBuf::from("part.rhai")
        );
    }

    #[test]
    fn loaded_scripts_are_bounded_utf8() {
        assert!(parse_script("bad.rhai".to_string(), &[0xff]).is_err());
        assert!(parse_script("big.rhai".to_string(), &vec![b'x'; MAX_SCRIPT_BYTES + 1]).is_err());
        assert_eq!(
            parse_script("ok.rhai".to_string(), b"print(42);")
                .unwrap()
                .source,
            "print(42);"
        );
    }
}
