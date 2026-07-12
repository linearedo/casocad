//! Load a scene.json and resave it through the Rust serializer.
//!
//!     cargo run -p caso-kernel --example resave -- <input.json> <output.json>

use caso_kernel::serialization::{load_scene_from_str, save_scene_to_string};

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("usage: resave <input.json> <output.json>");
    let output = args.next().expect("usage: resave <input.json> <output.json>");
    let text = std::fs::read_to_string(&input).expect("read input scene");
    let document = load_scene_from_str(&text).expect("load scene");
    let saved = save_scene_to_string(&document).expect("save scene");
    std::fs::write(&output, saved).expect("write output scene");
    println!("resaved {input} -> {output}");
}
