//! casoWASM application entry (native). The same `CasoApp` runs on the web
//! through eframe's WebRunner (see `lib.rs`).

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 860.0])
            .with_title("casoWASM - Programmable SDF CAD"),
        ..Default::default()
    };
    eframe::run_native(
        "casoWASM",
        options,
        Box::new(|creation_context| Ok(Box::new(caso_app::CasoApp::new(creation_context)))),
    )
}

/// On wasm the entry point is `caso_app::start` (wasm-bindgen); this binary
/// target is only meaningful natively.
#[cfg(target_arch = "wasm32")]
fn main() {}
