//! Phase 5 acceptance helper: perform the draw→subtract→domain workflow and
//! save the scene for the Python interop check.
use caso_kernel::scene::SceneDocument;
use caso_kernel::serialization::save_scene_to_string;
use caso_kernel::vec3::vec3;

fn main() {
    let path = std::env::args().nth(1).expect("usage: acceptance_save <out.json>");
    let mut d = SceneDocument::new();
    let a = d.add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0).unwrap();
    let b = d.add_primitive_from_drag("cylinder", vec3(-0.3, -0.3, 0.0), vec3(0.3, 0.3, 0.0), 1.0).unwrap();
    let r = d.combine(a, b, "difference").unwrap();
    d.set_domain_root(r, caso_kernel::roles::DomainKind::Fluid).unwrap();
    std::fs::write(&path, save_scene_to_string(&d).unwrap()).unwrap();
    println!("saved {path}");
}
