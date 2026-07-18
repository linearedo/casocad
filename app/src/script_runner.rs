//! Rhai mesher-script runner. Scripts receive exact meshable `domains` plus
//! a topology-preserving `mesh` builder that emits MeshIR.

use std::cell::RefCell;
use std::rc::Rc;

use caso_kernel::meshing::{
    meshable_domains_from_document, MeshableBoundaryRegion, MeshableDomain, MeshableDomainSpace,
    MeshableDomains,
};
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::vec3;
use caso_meshing::{MeshIr, MeshIrBuilder};
use rhai::{Array, Dynamic, Engine, Scope};

#[derive(Clone)]
struct DomainsHandle(Rc<MeshableDomains>);

#[derive(Clone)]
struct DomainHandle(Rc<MeshableDomain>);

#[derive(Clone)]
struct RegionHandle(Rc<MeshableBoundaryRegion>);

#[derive(Clone)]
struct SpaceHandle(Rc<MeshableDomainSpace>);

#[derive(Clone)]
struct MeshHandle(Rc<RefCell<MeshIrBuilder>>);

fn dynamic_to_id(value: &Dynamic) -> Result<u64, String> {
    let id = value
        .as_int()
        .map_err(|_| "mesh ids must be integers".to_string())?;
    if id <= 0 {
        return Err("mesh ids must be positive".to_string());
    }
    Ok(id as u64)
}

fn array_to_ids(values: Array) -> Result<Vec<u64>, String> {
    values.iter().map(dynamic_to_id).collect()
}

fn int_to_id(id: i64) -> Result<u64, String> {
    if id <= 0 {
        return Err("mesh ids must be positive".to_string());
    }
    Ok(id as u64)
}

fn id_to_int(id: u64) -> Result<i64, Box<rhai::EvalAltResult>> {
    i64::try_from(id).map_err(|_| "mesh id exceeded Rhai integer range".to_string().into())
}

/// Run a mesher script against the document's declared Domains; returns the
/// built MeshIR. The exactness compile gate runs before any script code.
pub fn run_mesher_script(document: &SceneDocument, script: &str) -> Result<MeshIr, String> {
    let domains = meshable_domains_from_document(document).map_err(|error| error.to_string())?;
    let domains = DomainsHandle(Rc::new(domains));
    let mesh = MeshHandle(Rc::new(RefCell::new(MeshIrBuilder::new())));

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
        .register_fn(
            "sdf",
            |handle: &mut DomainHandle, x: f64, y: f64, z: f64| {
                handle.0.domain_sdf(&[vec3(x, y, z)])[0]
            },
        )
        .register_fn(
            "mesh_space",
            |handle: &mut DomainHandle| -> Result<SpaceHandle, Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .mesh_space()
                    .map(|space| SpaceHandle(Rc::new(space)))
                    .map_err(|error| error.to_string().into())
            },
        )
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

    engine
        .register_type_with_name::<SpaceHandle>("MeshSpace")
        .register_fn("bounds", |handle: &mut SpaceHandle| {
            handle
                .0
                .bounds()
                .iter()
                .copied()
                .map(Dynamic::from)
                .collect::<Array>()
        })
        .register_fn("point", |handle: &mut SpaceHandle, a: f64, b: f64| {
            let point = handle.0.point(a, b);
            [point.x, point.y, point.z]
                .into_iter()
                .map(Dynamic::from)
                .collect::<Array>()
        })
        .register_fn(
            "coords",
            |handle: &mut SpaceHandle, x: f64, y: f64, z: f64| {
                handle
                    .0
                    .coords(vec3(x, y, z))
                    .into_iter()
                    .map(Dynamic::from)
                    .collect::<Array>()
            },
        )
        .register_fn("sdf", |handle: &mut SpaceHandle, a: f64, b: f64| {
            handle.0.sdf(a, b)
        });

    engine
        .register_type_with_name::<MeshHandle>("Mesh")
        .register_fn("zone", |handle: &mut MeshHandle, name: &str, kind: &str| {
            id_to_int(handle.0.borrow_mut().zone(name, kind))
        })
        .register_fn("tag", |handle: &mut MeshHandle, name: &str, kind: &str| {
            id_to_int(handle.0.borrow_mut().tag(name, kind))
        })
        .register_fn(
            "point",
            |handle: &mut MeshHandle,
             x: f64,
             y: f64,
             z: f64|
             -> Result<i64, Box<rhai::EvalAltResult>> {
                let id = handle.0.borrow_mut().point(x, y, z)?;
                id_to_int(id)
            },
        )
        .register_fn(
            "face",
            |handle: &mut MeshHandle,
             type_name: &str,
             point_ids: Array|
             -> Result<i64, Box<rhai::EvalAltResult>> {
                let point_ids = array_to_ids(point_ids)?;
                let id = handle.0.borrow_mut().face(type_name, point_ids)?;
                id_to_int(id)
            },
        )
        .register_fn(
            "cell",
            |handle: &mut MeshHandle,
             type_name: &str,
             point_ids: Array,
             zone_id: i64|
             -> Result<i64, Box<rhai::EvalAltResult>> {
                let point_ids = array_to_ids(point_ids)?;
                let zone_id = int_to_id(zone_id)?;
                let id = handle.0.borrow_mut().cell(type_name, point_ids, zone_id)?;
                id_to_int(id)
            },
        )
        .register_fn(
            "cell_with_faces",
            |handle: &mut MeshHandle,
             type_name: &str,
             point_ids: Array,
             face_ids: Array,
             zone_id: i64|
             -> Result<i64, Box<rhai::EvalAltResult>> {
                let point_ids = array_to_ids(point_ids)?;
                let face_ids = array_to_ids(face_ids)?;
                let zone_id = int_to_id(zone_id)?;
                let id = handle
                    .0
                    .borrow_mut()
                    .cell_with_faces(type_name, point_ids, face_ids, zone_id)?;
                id_to_int(id)
            },
        )
        .register_fn(
            "tag_edge",
            |handle: &mut MeshHandle,
             point_ids: Array,
             tag_id: i64|
             -> Result<(), Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .borrow_mut()
                    .tag_edge(array_to_ids(point_ids)?, int_to_id(tag_id)?);
                Ok(())
            },
        )
        .register_fn(
            "tag_face",
            |handle: &mut MeshHandle,
             point_ids: Array,
             tag_id: i64|
             -> Result<(), Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .borrow_mut()
                    .tag_face(array_to_ids(point_ids)?, int_to_id(tag_id)?);
                Ok(())
            },
        )
        .register_fn(
            "attribute",
            |handle: &mut MeshHandle,
             target_kind: &str,
             target_id: i64,
             key: &str,
             value_json: &str|
             -> Result<(), Box<rhai::EvalAltResult>> {
                let value: serde_json::Value =
                    serde_json::from_str(value_json).map_err(|error| error.to_string())?;
                handle
                    .0
                    .borrow_mut()
                    .attribute(target_kind, int_to_id(target_id)?, key, value);
                Ok(())
            },
        );

    let mut scope = Scope::new();
    scope.push("domains", domains);
    scope.push("mesh", mesh.clone());
    engine
        .run_with_scope(&mut scope, script)
        .map_err(|error| error.to_string())?;
    drop(scope);
    drop(engine);

    Rc::try_unwrap(mesh.0)
        .map_err(|_| "mesh builder is still referenced".to_string())?
        .into_inner()
        .build()
}

/// The example script preloaded in the Meshing workspace.
pub const EXAMPLE_SCRIPT: &str = r#"// casoCAD MeshIR example script (Rhai).
let fluid = domains.get("fluid");
let zone = mesh.zone(fluid.name, fluid.kind);
let sample_edge = mesh.tag("sample_edge", "boundary");
let b = fluid.bounds();
let h = 0.15;
let x = b[0] + h;
let y = (b[2] + b[3]) * 0.5;
let z = (b[4] + b[5]) * 0.5;

let p00 = mesh.point(x,     y,     z);
let p10 = mesh.point(x + h, y,     z);
let p01 = mesh.point(x,     y + h, z);
let p11 = mesh.point(x + h, y + h, z);

if fluid.sdf(x + h * 0.5, y + h * 0.5, z) < 0.0 {
    mesh.cell("tri3", [p00, p10, p11], zone);
    mesh.cell("tri3", [p00, p11, p01], zone);
    mesh.tag_edge([p00, p10], sample_edge);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_script_builds_mesh_ir_slice_triangles() {
        let document = SceneDocument::default_scene().expect("default scene");
        let mesh = run_mesher_script(&document, EXAMPLE_SCRIPT).expect("script runs");
        assert_eq!(mesh.points.len(), 4);
        assert_eq!(mesh.cells.len(), 2);
        assert_eq!(mesh.edges.len(), 5);
        assert!(mesh.edges.iter().any(|edge| !edge.tag_ids.is_empty()));

        let domains = meshable_domains_from_document(&document).expect("domains");
        let fluid = domains.get("fluid").expect("fluid");
        for point in &mesh.points {
            assert!(
                fluid.domain_sdf(&[vec3(
                    point.position[0],
                    point.position[1],
                    point.position[2],
                )])[0]
                    < 0.0
            );
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
                mesh.point(0.0, 0.0, 0.5);
            }
            if !r.contains(4.5, 0.0, 0.5) {
                mesh.point(9.0, 9.0, 9.0);
            }
        "#;
        let mesh = run_mesher_script(&document, script).expect("script runs");
        assert_eq!(mesh.points.len(), 2);
    }

    #[test]
    fn script_errors_are_reported() {
        let document = SceneDocument::default_scene().expect("default scene");
        assert!(run_mesher_script(&document, "nonsense(").is_err());
        assert!(run_mesher_script(&document, r#"domains.get("nope");"#)
            .unwrap_err()
            .contains("unknown meshable domain"));
        assert!(
            run_mesher_script(&document, r#"mesh.cell("tri3", [1, 2, 3], 1);"#)
                .unwrap_err()
                .contains("does not exist")
        );
    }

    #[test]
    fn mesh_space_is_scriptable_for_2d_domains() {
        let mut document = SceneDocument::new();
        let section = document
            .add_primitive_from_drag("rectangle", vec3(0.0, 0.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
            .expect("rectangle");
        document
            .set_domain_root(section, caso_kernel::roles::DomainKind::Fluid)
            .expect("domain");
        let script = r#"
            let fluid = domains.get("fluid");
            let space = fluid.mesh_space();
            let zone = mesh.zone("fluid", "fluid");
            let b = space.bounds();
            let p0 = space.point(b[0], b[2]);
            let p1 = space.point(b[1], b[2]);
            let p2 = space.point(b[1], b[3]);
            if space.sdf(0.0, 0.0) < 0.0 {
                let a = mesh.point(p0[0], p0[1], p0[2]);
                let c = mesh.point(p1[0], p1[1], p1[2]);
                let d = mesh.point(p2[0], p2[1], p2[2]);
                mesh.cell("tri3", [a, c, d], zone);
            }
        "#;
        let mesh = run_mesher_script(&document, script).expect("script runs");
        assert_eq!(mesh.points.len(), 3);
        assert_eq!(mesh.cells.len(), 1);
    }
}
