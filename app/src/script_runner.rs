//! Rhai mesher-script runner — the WASM-compatible replacement for the
//! Python-subprocess scripting page. Scripts receive the same surface as the
//! Python contract: `domains` (lookup by name or unique kind) and
//! `emit(element_type, vertices, tag_name)`.
//!
//! Example script:
//! ```rhai
//! let fluid = domains.get("fluid");
//! let b = fluid.bounds();          // [xmin, xmax, ymin, ymax, zmin, zmax]
//! let dx = 0.16;
//! let x = b[0] + dx/2;
//! while x < b[1] {
//!     let y = b[2] + dx/2;
//!     while y < b[3] {
//!         if fluid.sdf(x, y, 0.0) < 0.0 {
//!             emit("point", [[x, y, 0.0]], "fluid_internal");
//!         }
//!         y += dx;
//!     }
//!     x += dx;
//! }
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use caso_kernel::meshing::{
    meshable_domains_from_document, MeshableBoundaryRegion, MeshableDomain, MeshableDomains,
};
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::vec3;
use caso_meshing::MeshElement;
use rhai::{Array, Dynamic, Engine, Scope};

#[derive(Clone)]
struct DomainsHandle(Rc<MeshableDomains>);

#[derive(Clone)]
struct DomainHandle(Rc<MeshableDomain>);

#[derive(Clone)]
struct RegionHandle(Rc<MeshableBoundaryRegion>);

fn dynamic_to_vertex(value: &Dynamic) -> Result<[f64; 3], String> {
    let array = value
        .read_lock::<Array>()
        .ok_or("each vertex must be an [x, y, z] array")?;
    if array.len() != 3 {
        return Err("each vertex must have exactly three components".to_string());
    }
    let mut out = [0.0; 3];
    for (slot, component) in out.iter_mut().zip(array.iter()) {
        *slot = component
            .as_float()
            .or_else(|_| component.as_int().map(|value| value as f64))
            .map_err(|_| "vertex components must be numbers".to_string())?;
    }
    Ok(out)
}

/// Run a mesher script against the document's declared Domains; returns the
/// emitted elements. The exactness compile gate runs before any script code.
pub fn run_mesher_script(
    document: &SceneDocument,
    script: &str,
) -> Result<Vec<MeshElement>, String> {
    let domains = meshable_domains_from_document(document).map_err(|error| error.to_string())?;
    let domains = DomainsHandle(Rc::new(domains));
    let emitted: Rc<RefCell<Vec<MeshElement>>> = Rc::new(RefCell::new(Vec::new()));

    let mut engine = Engine::new();
    engine.set_max_operations(200_000_000); // generous but bounded

    engine
        .register_type_with_name::<DomainsHandle>("Domains")
        .register_fn(
            "get",
            |handle: &mut DomainsHandle,
             key: &str|
             -> Result<DomainHandle, Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .get(key)
                    .map(|domain| DomainHandle(Rc::new(domain.clone())))
                    .map_err(|error| error.to_string().into())
            },
        )
        .register_fn("len", |handle: &mut DomainsHandle| handle.0.len() as i64)
        .register_fn("keys", |handle: &mut DomainsHandle| {
            handle
                .0
                .keys()
                .into_iter()
                .map(Dynamic::from)
                .collect::<Array>()
        });

    engine
        .register_type_with_name::<DomainHandle>("Domain")
        .register_get("name", |handle: &mut DomainHandle| handle.0.name.clone())
        .register_get("kind", |handle: &mut DomainHandle| {
            handle.0.kind.as_str().to_string()
        })
        .register_get("dimension", |handle: &mut DomainHandle| {
            handle.0.dimension as i64
        })
        .register_fn("bounds", |handle: &mut DomainHandle| {
            let bounds = &handle.0.bounds;
            [
                bounds.x_min,
                bounds.x_max,
                bounds.y_min,
                bounds.y_max,
                bounds.z_min,
                bounds.z_max,
            ]
            .iter()
            .copied()
            .map(Dynamic::from)
            .collect::<Array>()
        })
        .register_fn("sdf", |handle: &mut DomainHandle, x: f64, y: f64, z: f64| {
            handle.0.domain_sdf(&[vec3(x, y, z)])[0]
        })
        .register_fn("regions", |handle: &mut DomainHandle| {
            handle
                .0
                .boundary_regions
                .iter()
                .map(|region| Dynamic::from(RegionHandle(Rc::new(region.clone()))))
                .collect::<Array>()
        })
        .register_fn(
            "region",
            |handle: &mut DomainHandle,
             name: &str|
             -> Result<RegionHandle, Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .region_by_name(name)
                    .map(|region| RegionHandle(Rc::new(region.clone())))
                    .map_err(|error| error.to_string().into())
            },
        );

    engine
        .register_type_with_name::<RegionHandle>("BoundaryRegion")
        .register_get("name", |handle: &mut RegionHandle| handle.0.name.clone())
        .register_get("tag", |handle: &mut RegionHandle| {
            handle.0.tag.clone().unwrap_or_default()
        })
        .register_fn(
            "contains",
            |handle: &mut RegionHandle, x: f64, y: f64, z: f64| {
                handle
                    .0
                    .contains(&[vec3(x, y, z)])
                    .map(|mask| mask[0])
                    .unwrap_or(false)
            },
        )
        .register_fn(
            "owner_sdf",
            |handle: &mut RegionHandle, x: f64, y: f64, z: f64| {
                handle.0.owner_sdf(&[vec3(x, y, z)])[0]
            },
        );

    let sink = emitted.clone();
    engine.register_fn(
        "emit",
        move |element_type: &str, vertices: Array, tag_name: &str| -> Result<(), Box<rhai::EvalAltResult>> {
            let mut points = Vec::with_capacity(vertices.len());
            for vertex in &vertices {
                points.push(dynamic_to_vertex(vertex).map_err(|error| error.to_string())?);
            }
            sink.borrow_mut().push(MeshElement {
                element_type: element_type.to_string(),
                vertices: points,
                tag_name: tag_name.to_string(),
            });
            Ok(())
        },
    );

    let mut scope = Scope::new();
    scope.push("domains", domains);
    engine
        .run_with_scope(&mut scope, script)
        .map_err(|error| error.to_string())?;
    drop(engine);
    Ok(Rc::try_unwrap(emitted)
        .map(RefCell::into_inner)
        .unwrap_or_default())
}

/// The example script preloaded in the Meshing workspace.
pub const EXAMPLE_SCRIPT: &str = r#"// casoWASM mesher script (Rhai).
// `domains` exposes the declared Fluid/Solid Domains;
// `emit(element_type, vertices, tag_name)` streams mesh elements.
let fluid = domains.get("fluid");
let b = fluid.bounds();
let dx = 0.16;
let regions = fluid.regions();
let z = (b[4] + b[5]) / 2.0;
let x = b[0] + dx/2.0;
while x < b[1] {
    let y = b[2] + dx/2.0;
    while y < b[3] {
        if fluid.sdf(x, y, z) < 0.0 {
            emit("point", [[x, y, z]], "fluid_internal");
        }
        y += dx;
    }
    x += dx;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_script_emits_interior_lattice_points() {
        let document = SceneDocument::default_scene().expect("default scene");
        let elements = run_mesher_script(&document, EXAMPLE_SCRIPT).expect("script runs");
        assert!(!elements.is_empty());
        // Every emitted point is strictly inside the fluid (not in the
        // cylinder obstacle, not outside the flow box).
        let domains = meshable_domains_from_document(&document).expect("domains");
        let fluid = domains.get("fluid").expect("fluid");
        for element in &elements {
            assert_eq!(element.element_type, "point");
            assert_eq!(element.tag_name, "fluid_internal");
            let point = element.vertices[0];
            assert!(fluid.domain_sdf(&[vec3(point[0], point[1], point[2])])[0] < 0.0);
        }
    }

    #[test]
    fn boundary_regions_are_scriptable() {
        let mut document = SceneDocument::default_scene().expect("default scene");
        let mut box_id = 0;
        for (id, _parent) in document.walk() {
            if matches!(
                document.object(id).expect("object").payload,
                caso_kernel::scene::ScenePayload::Box3(_)
            ) {
                box_id = id;
            }
        }
        document
            .add_boundary_region(box_id, None, Some("-X"), None)
            .expect("region");
        let script = r#"
            let fluid = domains.get("fluid");
            let regions = fluid.regions();
            let r = regions[0];
            for candidate in regions {
                if candidate.name.contains("-X") {
                    r = candidate;
                }
            }
            if r.contains(0.0, 0.0, 0.5) {
                emit("point", [[0.0, 0.0, 0.5]], r.name);
            }
            if !r.contains(4.5, 0.0, 0.5) {
                emit("point", [[9.0, 9.0, 9.0]], "not_in_region");
            }
        "#;
        let elements = run_mesher_script(&document, script).expect("script runs");
        assert_eq!(elements.len(), 2);
        assert!(elements[0].tag_name.contains("-X"));
        assert_eq!(elements[1].tag_name, "not_in_region");
    }

    #[test]
    fn script_errors_are_reported() {
        let document = SceneDocument::default_scene().expect("default scene");
        assert!(run_mesher_script(&document, "nonsense(").is_err());
        assert!(run_mesher_script(&document, r#"domains.get("nope");"#)
            .unwrap_err()
            .contains("unknown meshable domain"));
    }
}
