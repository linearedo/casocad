//! casoCAD visual theme for egui, from `app/assets/theme.qss` and the
//! `docs/` web mock: bg #0f1216, panel #161b22, header #1c232c, line #2a323c,
//! text #d7dde4, dim #8a96a3, accent #4aa8ff, selection #1d2a38/#2d6fb0.

use eframe::egui;

#[allow(dead_code)]
pub const BG: egui::Color32 = egui::Color32::from_rgb(0x0f, 0x12, 0x16);
pub const PANEL: egui::Color32 = egui::Color32::from_rgb(0x16, 0x1b, 0x22);
pub const HEADER: egui::Color32 = egui::Color32::from_rgb(0x1c, 0x23, 0x2c);
pub const LINE: egui::Color32 = egui::Color32::from_rgb(0x2a, 0x32, 0x3c);
pub const TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(0xd7, 0xdd, 0xe4);
#[allow(dead_code)]
pub const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(0x8a, 0x96, 0xa3);
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x4a, 0xa8, 0xff);
pub const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(0x2d, 0x6f, 0xb0);
pub const INPUT_BG: egui::Color32 = egui::Color32::from_rgb(0x11, 0x16, 0x1c);
pub const SELECTION: egui::Color32 = egui::Color32::from_rgb(0x1d, 0x2a, 0x38);

pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|style| {
        let visuals = &mut style.visuals;
        *visuals = egui::Visuals::dark();
        visuals.panel_fill = PANEL;
        visuals.window_fill = PANEL;
        visuals.extreme_bg_color = INPUT_BG;
        visuals.faint_bg_color = HEADER;
        visuals.selection.bg_fill = ACCENT_DIM;
        visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
        visuals.hyperlink_color = ACCENT;
        visuals.override_text_color = Some(TEXT_COLOR);
        visuals.widgets.noninteractive.bg_fill = PANEL;
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, LINE);
        visuals.widgets.inactive.bg_fill = HEADER;
        visuals.widgets.hovered.bg_fill = SELECTION;
        visuals.widgets.active.bg_fill = ACCENT_DIM;
        visuals.window_stroke = egui::Stroke::new(1.0, LINE);
    });
}
