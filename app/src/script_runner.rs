//! Rhai mesher-script runner. Scripts receive exact meshable `domains` plus
//! a topology-preserving `mesh` builder that emits MeshIR.

use std::cell::RefCell;
use std::rc::Rc;

use caso_kernel::meshing::{
    meshable_domains_from_document, MeshableBoundaryRegion, MeshableDomain, MeshableDomainSpace,
    MeshableDomains, MeshableInterface,
};
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::{vec3, Vec3};
use caso_meshing::toolkit::{boundary_loops, SizingBand, SizingField, SizingSpec};
use caso_meshing::{MeshIr, MeshIrBuilder};
use rhai::{Array, Dynamic, Engine, Map, Scope};

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

#[derive(Clone)]
struct InterfaceHandle(Rc<MeshableInterface>);

#[derive(Clone)]
struct SizingHandle(Rc<SizingField>);

fn vec3_array(value: Vec3) -> Array {
    [value.x, value.y, value.z]
        .into_iter()
        .map(Dynamic::from)
        .collect()
}

fn dynamic_to_f64(value: &Dynamic) -> Result<f64, Box<rhai::EvalAltResult>> {
    value
        .as_float()
        .or_else(|_| value.as_int().map(|int| int as f64))
        .map_err(|_| "expected a number".to_string().into())
}

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
    // Mesher scripts nest loops inside functions; the default per-function
    // expression depth (32) is too tight for ordinary connectivity code.
    engine.set_max_expr_depths(128, 128);

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
        )
        .register_fn(
            "normal",
            |handle: &mut RegionHandle, x: f64, y: f64, z: f64| {
                vec3_array(handle.0.normals(&[vec3(x, y, z)])[0])
            },
        )
        .register_fn(
            "project_to_owner",
            |handle: &mut RegionHandle,
             x: f64,
             y: f64,
             z: f64|
             -> Result<Array, Box<rhai::EvalAltResult>> {
                let projection = handle.0.project_to_owner(&[vec3(x, y, z)])[0];
                if !projection.converged {
                    return Err(format!(
                        "owner projection did not converge from ({x}, {y}, {z})"
                    )
                    .into());
                }
                Ok(vec3_array(projection.point))
            },
        );

    // Toolkit queries on domains: normals, interior projection, curvature,
    // total classification, exact 2D boundary loops.
    engine
        .register_fn(
            "normal",
            |handle: &mut DomainHandle, x: f64, y: f64, z: f64| {
                vec3_array(handle.0.normals(&[vec3(x, y, z)])[0])
            },
        )
        .register_fn(
            "project",
            |handle: &mut DomainHandle,
             x: f64,
             y: f64,
             z: f64|
             -> Result<Array, Box<rhai::EvalAltResult>> {
                let projection = handle
                    .0
                    .project_to_boundary(&[vec3(x, y, z)])
                    .map_err(|error| error.to_string())?[0];
                if !projection.converged {
                    return Err(format!(
                        "projection did not converge from ({x}, {y}, {z}); \
                         start from the domain interior"
                    )
                    .into());
                }
                Ok(vec3_array(projection.point))
            },
        )
        .register_fn(
            "curvature",
            |handle: &mut DomainHandle,
             x: f64,
             y: f64,
             z: f64|
             -> Result<f64, Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .curvature(&[vec3(x, y, z)])
                    .map(|values| values[0])
                    .map_err(|error| error.to_string().into())
            },
        )
        .register_fn(
            "classify",
            |handle: &mut DomainHandle,
             x: f64,
             y: f64,
             z: f64|
             -> Result<Map, Box<rhai::EvalAltResult>> {
                let class = handle
                    .0
                    .classify_boundary(&[vec3(x, y, z)], None)
                    .map_err(|error| error.to_string())?[0]
                    .clone();
                let mut map = Map::new();
                map.insert("on_boundary".into(), Dynamic::from(class.on_boundary));
                map.insert(
                    "owner".into(),
                    Dynamic::from(i64::from(class.owner_object_id)),
                );
                map.insert(
                    "region".into(),
                    match class.region_index {
                        Some(index) => {
                            Dynamic::from(handle.0.boundary_regions[index].name.clone())
                        }
                        None => Dynamic::UNIT,
                    },
                );
                Ok(map)
            },
        )
        .register_fn(
            "boundary_loops",
            |handle: &mut DomainHandle,
             resolution: i64|
             -> Result<Array, Box<rhai::EvalAltResult>> {
                let loops = boundary_loops(&handle.0, resolution.max(1) as usize)
                    .map_err(|error| error.to_string())?;
                Ok(loops
                    .into_iter()
                    .map(|chain| {
                        let mut map = Map::new();
                        map.insert("is_outer".into(), Dynamic::from(chain.is_outer));
                        map.insert("area".into(), Dynamic::from(chain.signed_area));
                        let spans: Array = chain
                            .spans
                            .into_iter()
                            .map(|span| {
                                let mut entry = Map::new();
                                entry.insert("patch".into(), Dynamic::from(span.patch_id));
                                entry.insert(
                                    "owner".into(),
                                    Dynamic::from(i64::from(span.owner_object_id)),
                                );
                                entry.insert(
                                    "region".into(),
                                    match span.region_name {
                                        Some(name) => Dynamic::from(name),
                                        None => Dynamic::UNIT,
                                    },
                                );
                                let points: Array = span
                                    .points
                                    .into_iter()
                                    .map(|point| Dynamic::from(vec3_array(point)))
                                    .collect();
                                entry.insert("points".into(), Dynamic::from(points));
                                Dynamic::from(entry)
                            })
                            .collect();
                        map.insert("spans".into(), Dynamic::from(spans));
                        Dynamic::from(map)
                    })
                    .collect())
            },
        );

    // Domain interfaces: keyed lookup first, iteration second.
    engine
        .register_type_with_name::<InterfaceHandle>("DomainsInterface")
        .register_get("domain_a", |handle: &mut InterfaceHandle| {
            handle.0.domain_a.clone()
        })
        .register_get("domain_b", |handle: &mut InterfaceHandle| {
            handle.0.domain_b.clone()
        })
        .register_get("owner", |handle: &mut InterfaceHandle| {
            i64::from(handle.0.owner_object_id)
        })
        .register_fn(
            "sdf",
            |handle: &mut InterfaceHandle, x: f64, y: f64, z: f64| {
                handle.0.surface_sdf(&[vec3(x, y, z)])[0]
            },
        )
        .register_fn(
            "contains",
            |handle: &mut InterfaceHandle, x: f64, y: f64, z: f64| {
                handle.0.contains(&[vec3(x, y, z)])[0]
            },
        )
        .register_fn(
            "project",
            |handle: &mut InterfaceHandle,
             x: f64,
             y: f64,
             z: f64|
             -> Result<Array, Box<rhai::EvalAltResult>> {
                let projection = handle.0.project(&[vec3(x, y, z)])[0];
                if !projection.converged {
                    return Err(format!(
                        "interface projection did not converge from ({x}, {y}, {z}); \
                         start inside one of the two domains"
                    )
                    .into());
                }
                Ok(vec3_array(projection.point))
            },
        )
        .register_fn(
            "interface",
            |handle: &mut DomainsHandle,
             a: &str,
             b: &str|
             -> Result<InterfaceHandle, Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .interface_between(a, b)
                    .map(|interface| InterfaceHandle(Rc::new(interface.clone())))
                    .map_err(|error| error.to_string().into())
            },
        )
        .register_fn(
            "interface",
            |handle: &mut DomainsHandle,
             a: DomainHandle,
             b: DomainHandle|
             -> Result<InterfaceHandle, Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .interface_between(&a.0.name, &b.0.name)
                    .map(|interface| InterfaceHandle(Rc::new(interface.clone())))
                    .map_err(|error| error.to_string().into())
            },
        )
        .register_fn("interfaces", |handle: &mut DomainsHandle| -> Array {
            handle
                .0
                .interfaces()
                .iter()
                .map(|interface| Dynamic::from(InterfaceHandle(Rc::new(interface.clone()))))
                .collect()
        });

    // Sizing field: spec map with explicit keys; unknown keys are errors.
    engine
        .register_type_with_name::<SizingHandle>("SizingField")
        .register_fn(
            "size_at",
            |handle: &mut SizingHandle, x: f64, y: f64, z: f64| handle.0.size_at(vec3(x, y, z)),
        )
        .register_fn(
            "sizing",
            |domain: DomainHandle, spec_map: Map| -> Result<SizingHandle, Box<rhai::EvalAltResult>> {
                let mut spec = SizingSpec::for_domain(&domain.0);
                for (key, value) in &spec_map {
                    match key.as_str() {
                        "background" => spec.background = dynamic_to_f64(value)?,
                        "min_size" => spec.min_size = dynamic_to_f64(value)?,
                        "gradation" => spec.gradation = dynamic_to_f64(value)?,
                        "curvature_factor" => {
                            spec.curvature_factor = Some(dynamic_to_f64(value)?)
                        }
                        "bands" => {
                            let bands: Array = value
                                .clone()
                                .try_cast()
                                .ok_or("sizing bands must be an array of maps")?;
                            for band in bands {
                                let entry: Map = band
                                    .try_cast()
                                    .ok_or("each sizing band must be a map")?;
                                let region: String = entry
                                    .get("region")
                                    .and_then(|name| name.clone().try_cast())
                                    .ok_or("sizing band needs a region name")?;
                                let distance = dynamic_to_f64(
                                    entry.get("distance").ok_or("sizing band needs a distance")?,
                                )?;
                                let size = dynamic_to_f64(
                                    entry.get("size").ok_or("sizing band needs a size")?,
                                )?;
                                spec.bands.push(SizingBand {
                                    region,
                                    distance,
                                    size,
                                });
                            }
                        }
                        other => {
                            return Err(format!(
                                "unknown sizing key {other:?}; known: background, min_size, \
                                 gradation, curvature_factor, bands"
                            )
                            .into())
                        }
                    }
                }
                SizingField::new((*domain.0).clone(), spec)
                    .map(|field| SizingHandle(Rc::new(field)))
                    .map_err(|error| error.to_string().into())
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

/// The example script preloaded in the Meshing workspace: a conforming
/// grid mesher over every domain in the document — uniform cells kept
/// where the domain is, near-wall vertices snapped exactly onto the
/// boundary (no gap), boundary faces/edges tagged by region. Fully
/// generic: no scene-specific knobs, and snap failures fall back to the
/// grid position instead of aborting.
pub const EXAMPLE_SCRIPT: &str = r#"// casoCAD example mesher (Rhai): a conforming grid mesh on every domain.
//
// One idea, both dimensions: lay a uniform grid over the domain, keep the
// cells that are fully inside, and snap every vertex that sits within one
// cell of the wall exactly onto it (projection of an interior point is
// exact). The mesh hugs the boundary with no gap; boundary faces (3D) and
// edges (2D) whose vertices all landed on the wall are tagged with the
// boundary region that owns them.

let cells = 24;         // grid cells along the longest axis

// A cached grid corner: [point id, snapped?, x, y, z]. Corners within one
// cell of the wall are pulled onto it; if the projection fails (a seam),
// the corner simply stays where it was - never an error.
fn corner(mesh, d, ids, key, x, y, z, band) {
    if key in ids { return ids[key]; }
    let snapped = false;
    if d.sdf(x, y, z) > -band {
        try {
            let p = d.project(x, y, z);
            x = p[0]; y = p[1]; z = p[2];
            snapped = true;
        } catch { }
    }
    let rec = [mesh.point(x, y, z), snapped, x, y, z];
    ids[key] = rec;
    rec
}

// Number of grid cells covering an extent at spacing h.
fn cell_count(extent, h) {
    let n = 1;
    while n * h < extent - 1e-12 { n += 1; }
    n
}

// Tag of a boundary sample: its winning region's tag, else the wall tag.
fn boundary_tag(d, tags, wall_tag, x, y, z) {
    let c = d.classify(x, y, z);
    if c.region != () && c.region in tags { tags[c.region] } else { wall_tag }
}

fn mesh_domain_3d(mesh, d, zone, tags, wall_tag, cells) {
    let b = d.bounds();
    let h = b[1] - b[0];
    if b[3] - b[2] > h { h = b[3] - b[2]; }
    if b[5] - b[4] > h { h = b[5] - b[4]; }
    h = h / cells;
    let band = 1.05 * h;
    let nx = cell_count(b[1] - b[0], h);
    let ny = cell_count(b[3] - b[2], h);
    let nz = cell_count(b[5] - b[4], h);
    let off = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
    let faces = [[0,3,7,4],[1,2,6,5],[0,1,5,4],[3,2,6,7],[0,1,2,3],[4,5,6,7]];
    let ids = #{};
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // keep the cell only if all corners and the centre are inside
                let inside = true;
                for o in off {
                    let x = b[0] + (i + o[0]) * h;
                    let y = b[2] + (j + o[1]) * h;
                    let z = b[4] + (k + o[2]) * h;
                    if d.sdf(x, y, z) >= 0.0 { inside = false; break; }
                }
                if !inside { continue; }
                let cx = b[0] + (i + 0.5) * h;
                let cy = b[2] + (j + 0.5) * h;
                let cz = b[4] + (k + 0.5) * h;
                if d.sdf(cx, cy, cz) >= 0.0 { continue; }

                let cs = [];
                for o in off {
                    let x = b[0] + (i + o[0]) * h;
                    let y = b[2] + (j + o[1]) * h;
                    let z = b[4] + (k + o[2]) * h;
                    let key = `${i + o[0]},${j + o[1]},${k + o[2]}`;
                    cs.push(corner(mesh, d, ids, key, x, y, z, band));
                }
                let pts = [cs[0][0], cs[1][0], cs[2][0], cs[3][0],
                           cs[4][0], cs[5][0], cs[6][0], cs[7][0]];
                mesh.cell("hex8", pts, zone);

                // a face whose four corners all landed on the wall IS wall
                for f in faces {
                    if cs[f[0]][1] && cs[f[1]][1] && cs[f[2]][1] && cs[f[3]][1] {
                        let mx = (cs[f[0]][2] + cs[f[1]][2] + cs[f[2]][2] + cs[f[3]][2]) / 4.0;
                        let my = (cs[f[0]][3] + cs[f[1]][3] + cs[f[2]][3] + cs[f[3]][3]) / 4.0;
                        let mz = (cs[f[0]][4] + cs[f[1]][4] + cs[f[2]][4] + cs[f[3]][4]) / 4.0;
                        let t = boundary_tag(d, tags, wall_tag, mx, my, mz);
                        mesh.tag_face([pts[f[0]], pts[f[1]], pts[f[2]], pts[f[3]]], t);
                    }
                }
            }
        }
    }
}

fn mesh_domain_2d(mesh, d, zone, tags, wall_tag, cells) {
    let space = d.mesh_space();
    let sb = space.bounds();
    let h = sb[1] - sb[0];
    if sb[3] - sb[2] > h { h = sb[3] - sb[2]; }
    h = h / cells;
    let band = 1.05 * h;
    let na = cell_count(sb[1] - sb[0], h);
    let nb = cell_count(sb[3] - sb[2], h);
    let off = [[0,0],[1,0],[1,1],[0,1]];
    let edges = [[0,1],[1,2],[2,3],[3,0]];
    let ids = #{};
    for i in 0..na {
        for j in 0..nb {
            let inside = true;
            for o in off {
                let w = space.point(sb[0] + (i + o[0]) * h, sb[2] + (j + o[1]) * h);
                if d.sdf(w[0], w[1], w[2]) >= 0.0 { inside = false; break; }
            }
            if !inside { continue; }
            let c = space.point(sb[0] + (i + 0.5) * h, sb[2] + (j + 0.5) * h);
            if d.sdf(c[0], c[1], c[2]) >= 0.0 { continue; }

            let cs = [];
            for o in off {
                let w = space.point(sb[0] + (i + o[0]) * h, sb[2] + (j + o[1]) * h);
                let key = `${i + o[0]},${j + o[1]}`;
                cs.push(corner(mesh, d, ids, key, w[0], w[1], w[2], band));
            }
            let pts = [cs[0][0], cs[1][0], cs[2][0], cs[3][0]];
            mesh.cell("quad4", pts, zone);

            for e in edges {
                if cs[e[0]][1] && cs[e[1]][1] {
                    let mx = (cs[e[0]][2] + cs[e[1]][2]) * 0.5;
                    let my = (cs[e[0]][3] + cs[e[1]][3]) * 0.5;
                    let mz = (cs[e[0]][4] + cs[e[1]][4]) * 0.5;
                    let t = boundary_tag(d, tags, wall_tag, mx, my, mz);
                    mesh.tag_edge([pts[e[0]], pts[e[1]]], t);
                }
            }
        }
    }
}

// ---------- every domain in the document ----------

let seen = [];
for key in domains.keys() {
    let d = domains.get(key);
    if d.name in seen { continue; }     // keys() also lists unique kinds
    seen.push(d.name);

    let zone = mesh.zone(d.name, d.kind);
    let tags = #{};                     // physics tags travel with the mesh
    for r in d.regions() { tags[r.name] = mesh.tag(r.name, r.tag); }
    let wall_tag = mesh.tag(d.name + "_wall", "wall");

    if d.dimension == 3 {
        mesh_domain_3d(mesh, d, zone, tags, wall_tag, cells);
    } else {
        mesh_domain_2d(mesh, d, zone, tags, wall_tag, cells);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The example at a coarser test grid (the default 24 is an
    /// interactive-quality setting; debug-build tests use 10).
    fn example_script_at_test_resolution() -> String {
        EXAMPLE_SCRIPT.replace("let cells = 24;", "let cells = 10;")
    }

    #[test]
    fn example_script_builds_a_conforming_3d_mesh() {
        let document = SceneDocument::default_scene().expect("default scene");
        let mesh =
            run_mesher_script(&document, &example_script_at_test_resolution()).expect("script runs");
        assert!(!mesh.cells.is_empty());
        assert!(mesh.cells.iter().all(|cell| cell.type_name == "hex8"));
        assert_eq!(mesh.zones.len(), 1);
        // inlet + outlet region tags plus the wall tag.
        assert_eq!(mesh.tags.len(), 3);

        let domains = meshable_domains_from_document(&document).expect("domains");
        let fluid = domains.get("fluid").expect("fluid");
        let mut snapped = 0;
        for point in &mesh.points {
            let value = fluid.domain_sdf(&[vec3(
                point.position[0],
                point.position[1],
                point.position[2],
            )])[0];
            // No point outside; snapped points sit exactly on the wall.
            assert!(value <= 1e-6, "point outside the fluid: {value}");
            if value.abs() <= 1e-6 {
                snapped += 1;
            }
        }
        assert!(snapped > 50, "the mesh hugs the boundary ({snapped} on-wall points)");
        // Fully-snapped boundary faces carry a tag.
        assert!(
            mesh.faces.iter().any(|face| !face.tag_ids.is_empty()),
            "boundary faces are tagged"
        );
    }

    /// The regression the previous example failed: adding a second domain
    /// must not break the default script.
    #[test]
    fn example_script_survives_an_added_domain() {
        use caso_kernel::roles::DomainKind;
        use caso_kernel::scene::ScenePayload;

        let mut document = SceneDocument::default_scene().expect("default scene");
        let ball = document.add_primitive("sphere", 1.0).expect("ball");
        if let ScenePayload::Sphere(sphere) =
            &mut document.object_mut(ball).expect("ball").payload
        {
            sphere.center = vec3(6.0, 0.0, 0.5);
            sphere.radius = 0.5;
        }
        document
            .set_domain_root(ball, DomainKind::Solid)
            .expect("solid domain");

        let mesh =
            run_mesher_script(&document, &example_script_at_test_resolution()).expect("script runs");
        assert_eq!(mesh.zones.len(), 2, "both domains meshed");
        let mut zones_with_cells: Vec<u64> = mesh
            .cells
            .iter()
            .filter_map(|cell| cell.zone_id)
            .collect();
        zones_with_cells.sort_unstable();
        zones_with_cells.dedup();
        assert_eq!(zones_with_cells.len(), 2, "cells in both zones");
    }

    #[test]
    fn example_script_meshes_2d_domains() {
        use caso_kernel::roles::DomainKind;

        let mut document = SceneDocument::new();
        let rect = document
            .add_primitive_from_drag(
                "rectangle",
                vec3(-2.0, -1.0, 0.0),
                vec3(2.0, 1.0, 0.0),
                1.0,
            )
            .expect("rectangle");
        let circle = document
            .add_primitive_from_drag("circle", vec3(0.2, -0.3, 0.0), vec3(0.8, 0.3, 0.0), 1.0)
            .expect("circle");
        let domain = document
            .combine(rect, circle, "difference")
            .expect("difference");
        document
            .set_domain_root(domain, DomainKind::Fluid)
            .expect("fluid domain");
        // A whole-surface region: every snapped boundary edge should carry
        // its tag instead of the wall fallback.
        let owner = {
            let domains = meshable_domains_from_document(&document).expect("domains");
            domains
                .get("fluid")
                .expect("fluid")
                .classify_boundary(&[vec3(-2.0, 0.5, 0.0)], None)
                .expect("classify")[0]
                .owner_object_id
        };
        let region_id = document
            .add_boundary_region(owner, None, None, None)
            .expect("region");
        document
            .boundary_regions
            .iter_mut()
            .find(|region| region.object_id == region_id)
            .expect("region")
            .name = "skin".to_string();

        let mesh =
            run_mesher_script(&document, &example_script_at_test_resolution()).expect("script runs");
        assert!(!mesh.cells.is_empty(), "2D cells were built");
        assert!(mesh.cells.iter().all(|cell| cell.type_name == "quad4"));
        assert_eq!(mesh.zones.len(), 1);
        let skin = mesh
            .tags
            .iter()
            .find(|tag| tag.name == "skin")
            .expect("skin tag");
        assert!(
            mesh.edges.iter().any(|edge| edge.tag_ids.contains(&skin.id)),
            "boundary edges carry the region tag"
        );

        let domains = meshable_domains_from_document(&document).expect("domains");
        let fluid = domains.get("fluid").expect("fluid");
        let mut snapped = 0;
        for point in &mesh.points {
            assert!(
                point.position[2].abs() < 1e-9,
                "2D mesh stays in the drawing plane"
            );
            let value = fluid.domain_sdf(&[vec3(
                point.position[0],
                point.position[1],
                point.position[2],
            )])[0];
            assert!(value <= 1e-6, "point outside the domain: {value}");
            if value.abs() <= 1e-6 {
                snapped += 1;
            }
        }
        assert!(snapped > 10, "the mesh hugs the outline ({snapped} on-wall points)");
    }

    #[test]
    fn example_script_meshes_a_nested_2d_solid() {
        use caso_kernel::roles::DomainKind;

        // Rectangle fluid minus a circle that is itself a solid domain: the
        // script must mesh BOTH zones, hole and pin, in the same plane.
        let mut document = SceneDocument::new();
        let rect = document
            .add_primitive_from_drag(
                "rectangle",
                vec3(-2.0, -1.0, 0.0),
                vec3(2.0, 1.0, 0.0),
                1.0,
            )
            .expect("rectangle");
        let circle = document
            .add_primitive_from_drag("circle", vec3(0.2, -0.3, 0.0), vec3(0.8, 0.3, 0.0), 1.0)
            .expect("circle");
        document.rename(circle, "pin").expect("rename");
        document
            .set_domain_root(circle, DomainKind::Solid)
            .expect("solid mark");
        let domain = document
            .combine(rect, circle, "difference")
            .expect("difference");
        document
            .set_domain_root(domain, DomainKind::Fluid)
            .expect("fluid domain");

        let mesh =
            run_mesher_script(&document, &example_script_at_test_resolution()).expect("script runs");
        assert_eq!(mesh.zones.len(), 2, "fluid and pin zones");
        let mut zones_with_cells: Vec<u64> = mesh
            .cells
            .iter()
            .filter_map(|cell| cell.zone_id)
            .collect();
        zones_with_cells.sort_unstable();
        zones_with_cells.dedup();
        assert_eq!(zones_with_cells.len(), 2, "cells in both zones");
        assert!(mesh.cells.iter().all(|cell| cell.type_name == "quad4"));
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
    fn toolkit_queries_are_scriptable() {
        use caso_kernel::roles::DomainKind;
        use caso_kernel::scene::ScenePayload;

        // Water box with a solid ball inside: one interface.
        let mut document = SceneDocument::new();
        let boxy = document.add_primitive("box", 4.0).expect("box");
        let ball = document.add_primitive("sphere", 1.0).expect("ball");
        if let ScenePayload::Sphere(sphere) =
            &mut document.object_mut(ball).expect("ball").payload
        {
            sphere.radius = 0.4;
        }
        let water = document.combine(boxy, ball, "difference").expect("water");
        document.rename(ball, "ball").expect("rename");
        document.rename(water, "water").expect("rename");
        document
            .set_domain_root(water, DomainKind::Fluid)
            .expect("water domain");
        document
            .set_domain_root(ball, DomainKind::Solid)
            .expect("ball domain");

        let script = r#"
            let water = domains.get("water");
            let ball = domains.get("ball");
            let itf = domains.interface(water, ball);
            if domains.interfaces().len() == 1 && itf.domain_b == "ball" {
                mesh.point(1.0, 0.0, 0.0);
            }
            // On the shared wall: contains is true, sdf is ~0.
            if itf.contains(0.4, 0.0, 0.0) && itf.sdf(0.4, 0.0, 0.0).abs() < 1e-9 {
                mesh.point(2.0, 0.0, 0.0);
            }
            // Interior-seeded projection lands on the wall.
            let p = itf.project(0.3, 0.0, 0.0);
            if itf.contains(p[0], p[1], p[2]) {
                mesh.point(3.0, 0.0, 0.0);
            }
            // Classification: the ball wall is untagged boundary with an owner.
            let c = water.classify(0.4, 0.0, 0.0);
            if c.on_boundary && c.region == () && c.owner > 0 {
                mesh.point(4.0, 0.0, 0.0);
            }
            // Sizing with defaults plus one override.
            let s = sizing(water, #{ background: 0.5 });
            if s.size_at(1.0, 1.0, 1.0) <= 0.5 {
                mesh.point(5.0, 0.0, 0.0);
            }
        "#;
        let mesh = run_mesher_script(&document, script).expect("script runs");
        assert_eq!(mesh.points.len(), 5, "every toolkit query answered");
    }

    #[test]
    fn boundary_loops_are_scriptable() {
        use caso_kernel::roles::DomainKind;

        let mut document = SceneDocument::new();
        let rect = document
            .add_primitive_from_drag(
                "rectangle",
                vec3(-2.0, -1.0, 0.0),
                vec3(2.0, 1.0, 0.0),
                1.0,
            )
            .expect("rectangle");
        let circle = document
            .add_primitive_from_drag("circle", vec3(0.2, -0.3, 0.0), vec3(0.8, 0.3, 0.0), 1.0)
            .expect("circle");
        let domain = document
            .combine(rect, circle, "difference")
            .expect("difference");
        document
            .set_domain_root(domain, DomainKind::Fluid)
            .expect("fluid domain");

        let script = r#"
            let fluid = domains.get("fluid");
            let loops = fluid.boundary_loops(32);
            if loops.len() == 2 {
                mesh.point(1.0, 0.0, 0.0);
            }
            for chain in loops {
                if chain.is_outer && chain.spans.len() == 4 {
                    mesh.point(2.0, 0.0, 0.0);
                }
                if !chain.is_outer && chain.area < 0.0 {
                    mesh.point(3.0, 0.0, 0.0);
                }
            }
        "#;
        let mesh = run_mesher_script(&document, script).expect("script runs");
        assert_eq!(mesh.points.len(), 3, "loops delivered to the script");
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
