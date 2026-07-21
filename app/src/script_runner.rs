//! Rhai mesher-script runner. Scripts receive exact meshable `domains` plus
//! a topology-preserving `mesh` builder that emits MeshIR.

use std::cell::RefCell;
use std::rc::Rc;

use caso_kernel::meshing::{
    meshable_domains_from_document, BoundaryBand, MeshableBoundaryRegion, MeshableDomain,
    MeshableDomainSpace, MeshableDomains, MeshableInterface,
};
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::{vec3, Vec3};
use caso_meshing::toolkit::{
    boundary_marching_sample, boundary_names, SizingBand, SizingField, SizingSpec,
};
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
        .register_fn("names", |handle: &mut DomainsHandle| {
            handle
                .0
                .names()
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
            let mut map = Map::new();
            map.insert("x_min".into(), Dynamic::from(bounds.x_min));
            map.insert("x_max".into(), Dynamic::from(bounds.x_max));
            map.insert("y_min".into(), Dynamic::from(bounds.y_min));
            map.insert("y_max".into(), Dynamic::from(bounds.y_max));
            map.insert("z_min".into(), Dynamic::from(bounds.z_min));
            map.insert("z_max".into(), Dynamic::from(bounds.z_max));
            map
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

    // Toolkit queries on domains: boundary normals, interior projection,
    // curvature, total classification, keyed boundary sampling.
    engine
        .register_fn(
            "boundary_normal",
            |handle: &mut DomainHandle, x: f64, y: f64, z: f64| {
                vec3_array(handle.0.normals(&[vec3(x, y, z)])[0])
            },
        )
        .register_fn(
            "boundary_projection",
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
            "boundary_curvature",
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
                    .classify_boundary(&[vec3(x, y, z)], BoundaryBand::UnprojectedSamples)
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
                    match class.region_name {
                        Some(name) => Dynamic::from(name),
                        None => Dynamic::UNIT,
                    },
                );
                Ok(map)
            },
        )
        .register_fn(
            "boundaries",
            |handle: &mut DomainHandle| -> Result<Array, Box<rhai::EvalAltResult>> {
                let names = boundary_names(&handle.0).map_err(|error| error.to_string())?;
                Ok(names.into_iter().map(Dynamic::from).collect())
            },
        )
        .register_fn(
            "boundary_marching_sample",
            |handle: &mut DomainHandle,
             name: &str,
             npoints: i64|
             -> Result<Array, Box<rhai::EvalAltResult>> {
                let points =
                    boundary_marching_sample(&handle.0, name, npoints.max(0) as usize)
                        .map_err(|error| error.to_string())?;
                Ok(points
                    .into_iter()
                    .map(|point| Dynamic::from(vec3_array(point)))
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
            let bounds = handle.0.bounds();
            let mut map = Map::new();
            map.insert("a_min".into(), Dynamic::from(bounds[0]));
            map.insert("a_max".into(), Dynamic::from(bounds[1]));
            map.insert("b_min".into(), Dynamic::from(bounds[2]));
            map.insert("b_max".into(), Dynamic::from(bounds[3]));
            map
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

/// The example script preloaded in the Meshing workspace: a general,
/// boundary-first structured-grid mesher for every 2D and 3D Domain.
/// Occupancy is established before topology, so every exterior edge/face is
/// projection-attempted and tagged rather than inferred from distance alone.
pub const EXAMPLE_SCRIPT: &str = r#"// casoCAD boundary-first example mesher (Rhai)
//
// This is deliberately readable mesher code, not a hidden black box:
//   1. classify a structured background grid,
//   2. find its topological exterior from missing neighbour cells,
//   3. project every shared exterior vertex onto the exact Domain boundary,
//   4. tag every exterior edge (2D) or face (3D).
//
// The retained cells have a closed, consistently tagged exterior on arbitrary
// declared Domains. Increase these values for a finer boundary approximation.
let cells_2d = 48;
let cells_3d = 20;
let minimum_cross_cells = 6; // preserves thin domains without filling empty space

fn axis_count(extent, target_h, minimum) {
    let n = minimum;
    while n * target_h < extent - 1e-12 { n += 1; }
    n
}

fn key2(i, j) { `${i},${j}` }
fn key3(i, j, k) { `${i},${j},${k}` }

// Projection starts only from retained (interior) cell corners. At a C0 seam
// Newton projection may honestly refuse; keeping the original on-grid point
// preserves connectivity and the exterior entity still receives a wall tag.
fn project_exterior(d, p) {
    if d.sdf(p[0], p[1], p[2]) <= 0.0 {
        try { return d.boundary_projection(p[0], p[1], p[2]); }
        catch { }
    }
    p
}

// Region classification uses projected vertices, never a curved chord's
// off-surface midpoint. A transition between regions falls back to the
// Domain wall tag instead of inventing ownership.
fn boundary_tag(d, tags, wall_tag, points) {
    let region = ();
    for p in points {
        let c = d.classify(p[1], p[2], p[3]);
        if !c.on_boundary || c.region == () { return wall_tag; }
        if region == () { region = c.region; }
        else if region != c.region { return wall_tag; }
    }
    if region in tags { tags[region] } else { wall_tag }
}

fn mesh_domain_2d(mesh, d, zone, tags, wall_tag, target_cells, minimum) {
    let space = d.mesh_space();
    let b = space.bounds();
    let da = b.a_max - b.a_min;
    let db = b.b_max - b.b_min;
    let longest = if da > db { da } else { db };
    let target_h = longest / target_cells;
    let na = axis_count(da, target_h, minimum);
    let nb = axis_count(db, target_h, minimum);
    let ha = da / na;
    let hb = db / nb;
    let offsets = [[0,0],[1,0],[1,1],[0,1]];
    let edges = [[0,1],[1,2],[2,3],[3,0]];
    let neighbours = [[0,-1],[1,0],[0,1],[-1,0]];
    let kept = #{};

    // Pass 1: retain only cells fully inside the exact Domain. Boundary-zero
    // corners are accepted, so grid-aligned walls do not lose a cell layer.
    for i in 0..na {
        for j in 0..nb {
            let inside = true;
            for o in offsets {
                let p = space.point(b.a_min + (i + o[0]) * ha,
                                    b.b_min + (j + o[1]) * hb);
                if d.sdf(p[0], p[1], p[2]) > 0.0 { inside = false; break; }
            }
            if inside {
                let p = space.point(b.a_min + (i + 0.5) * ha,
                                    b.b_min + (j + 0.5) * hb);
                if d.sdf(p[0], p[1], p[2]) <= 0.0 { kept[key2(i, j)] = true; }
            }
        }
    }

    // Pass 2: an edge is exterior exactly when its neighbour cell is absent.
    // Mark its vertices before creating any points so sharing remains exact.
    let exterior = #{};
    for i in 0..na {
        for j in 0..nb {
            if !(key2(i, j) in kept) { continue; }
            for e in 0..4 {
                let ni = i + neighbours[e][0];
                let nj = j + neighbours[e][1];
                if ni < 0 || ni >= na || nj < 0 || nj >= nb || !(key2(ni, nj) in kept) {
                    for c in edges[e] {
                        exterior[key2(i + offsets[c][0], j + offsets[c][1])] = true;
                    }
                }
            }
        }
    }

    // Pass 3: emit cells, then tag every topological exterior edge.
    // Keep cache mutation in this scope: Rhai map arguments are passed by
    // value, so a helper cannot persist newly assigned point ids for us.
    let ids = #{};
    for i in 0..na {
        for j in 0..nb {
            if !(key2(i, j) in kept) { continue; }
            let corners = [];
            for o in offsets {
                let gi = i + o[0];
                let gj = j + o[1];
                let key = key2(gi, gj);
                if key in ids {
                    corners.push(ids[key]);
                } else {
                    let p = space.point(b.a_min + gi * ha, b.b_min + gj * hb);
                    if key in exterior { p = project_exterior(d, p); }
                    let record = [mesh.point(p[0], p[1], p[2]), p[0], p[1], p[2]];
                    ids[key] = record;
                    corners.push(record);
                }
            }
            let points = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]];
            mesh.cell("quad4", points, zone);
            for e in 0..4 {
                let ni = i + neighbours[e][0];
                let nj = j + neighbours[e][1];
                if ni < 0 || ni >= na || nj < 0 || nj >= nb || !(key2(ni, nj) in kept) {
                    let pair = [corners[edges[e][0]], corners[edges[e][1]]];
                    mesh.tag_edge([pair[0][0], pair[1][0]],
                                  boundary_tag(d, tags, wall_tag, pair));
                }
            }
        }
    }
}

fn mesh_domain_3d(mesh, d, zone, tags, wall_tag, target_cells, minimum) {
    let b = d.bounds();
    let dx = b.x_max - b.x_min;
    let dy = b.y_max - b.y_min;
    let dz = b.z_max - b.z_min;
    let longest = dx;
    if dy > longest { longest = dy; }
    if dz > longest { longest = dz; }
    let target_h = longest / target_cells;
    let nx = axis_count(dx, target_h, minimum);
    let ny = axis_count(dy, target_h, minimum);
    let nz = axis_count(dz, target_h, minimum);
    let hx = dx / nx;
    let hy = dy / ny;
    let hz = dz / nz;
    let offsets = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]];
    let faces = [[0,3,7,4],[1,2,6,5],[0,1,5,4],
                 [3,2,6,7],[0,1,2,3],[4,5,6,7]];
    let neighbours = [[-1,0,0],[1,0,0],[0,-1,0],
                      [0,1,0],[0,0,-1],[0,0,1]];
    let kept = #{};

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let inside = true;
                for o in offsets {
                    let x = b.x_min + (i + o[0]) * hx;
                    let y = b.y_min + (j + o[1]) * hy;
                    let z = b.z_min + (k + o[2]) * hz;
                    if d.sdf(x, y, z) > 0.0 { inside = false; break; }
                }
                if inside {
                    let x = b.x_min + (i + 0.5) * hx;
                    let y = b.y_min + (j + 0.5) * hy;
                    let z = b.z_min + (k + 0.5) * hz;
                    if d.sdf(x, y, z) <= 0.0 { kept[key3(i, j, k)] = true; }
                }
            }
        }
    }

    let exterior = #{};
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if !(key3(i, j, k) in kept) { continue; }
                for f in 0..6 {
                    let ni = i + neighbours[f][0];
                    let nj = j + neighbours[f][1];
                    let nk = k + neighbours[f][2];
                    if ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz ||
                       !(key3(ni, nj, nk) in kept) {
                        for c in faces[f] {
                            exterior[key3(i + offsets[c][0], j + offsets[c][1],
                                          k + offsets[c][2])] = true;
                        }
                    }
                }
            }
        }
    }

    // The same in-scope cache makes adjacent hexes share actual MeshIR ids,
    // not merely equal coordinates.
    let ids = #{};
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if !(key3(i, j, k) in kept) { continue; }
                let corners = [];
                for o in offsets {
                    let gi = i + o[0];
                    let gj = j + o[1];
                    let gk = k + o[2];
                    let key = key3(gi, gj, gk);
                    if key in ids {
                        corners.push(ids[key]);
                    } else {
                        let p = [b.x_min + gi * hx, b.y_min + gj * hy,
                                 b.z_min + gk * hz];
                        if key in exterior { p = project_exterior(d, p); }
                        let record = [mesh.point(p[0], p[1], p[2]), p[0], p[1], p[2]];
                        ids[key] = record;
                        corners.push(record);
                    }
                }
                let points = [corners[0][0], corners[1][0], corners[2][0], corners[3][0],
                              corners[4][0], corners[5][0], corners[6][0], corners[7][0]];
                mesh.cell("hex8", points, zone);
                for f in 0..6 {
                    let ni = i + neighbours[f][0];
                    let nj = j + neighbours[f][1];
                    let nk = k + neighbours[f][2];
                    if ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz ||
                       !(key3(ni, nj, nk) in kept) {
                        let quad = [corners[faces[f][0]], corners[faces[f][1]],
                                    corners[faces[f][2]], corners[faces[f][3]]];
                        mesh.tag_face([quad[0][0], quad[1][0], quad[2][0], quad[3][0]],
                                      boundary_tag(d, tags, wall_tag, quad));
                    }
                }
            }
        }
    }
}

// Mesh every declared Domain independently; names and physical kinds are
// preserved in MeshIR zones and boundary-region metadata travels as tags.
for name in domains.names() {
    let d = domains.get(name);
    let zone = mesh.zone(d.name, d.kind);
    let tags = #{};
    for region in d.regions() {
        tags[region.name] = mesh.tag(region.name, region.tag);
    }
    let wall_tag = mesh.tag(d.name + "_wall", "wall");
    if d.dimension == 2 {
        mesh_domain_2d(mesh, d, zone, tags, wall_tag, cells_2d, minimum_cross_cells);
    } else if d.dimension == 3 {
        mesh_domain_3d(mesh, d, zone, tags, wall_tag, cells_3d, minimum_cross_cells);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The example at coarser grids; debug-build tests exercise the same
    /// algorithm without paying the interactive preview's full resolution.
    fn example_script_at_test_resolution() -> String {
        EXAMPLE_SCRIPT
            .replace("let cells_2d = 48;", "let cells_2d = 18;")
            .replace("let cells_3d = 20;", "let cells_3d = 10;")
    }

    fn assert_closed_tagged_2d_boundary(mesh: &MeshIr) {
        let boundary: Vec<_> = mesh
            .edges
            .iter()
            .filter(|edge| edge.owner_cell_id.is_some() && edge.neighbor_cell_id.is_none())
            .collect();
        assert!(!boundary.is_empty(), "mesh has a topological boundary");
        let untagged: Vec<_> = boundary
            .iter()
            .filter(|edge| edge.tag_ids.is_empty())
            .collect();
        assert!(
            untagged.is_empty(),
            "every exterior edge is tagged; first missing: {:?}",
            untagged.first()
        );
        assert!(
            mesh.edges
                .iter()
                .filter(|edge| !edge.tag_ids.is_empty())
                .all(|edge| edge.neighbor_cell_id.is_none()),
            "interior edges are not boundary-tagged"
        );

        let mut degree = std::collections::BTreeMap::<u64, usize>::new();
        for edge in boundary {
            for point_id in &edge.point_ids {
                *degree.entry(*point_id).or_default() += 1;
            }
        }
        assert!(
            degree.values().all(|count| *count == 2),
            "boundary edges form closed loops"
        );
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
        let fluid = domains.get("von_karman_fluid").expect("fluid");
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
        // Topological boundary faces carry a tag.
        assert!(
            mesh.faces.iter().any(|face| !face.tag_ids.is_empty()),
            "boundary faces are tagged"
        );
        let untagged: Vec<_> = mesh
            .faces
            .iter()
            .filter(|face| {
                face.owner_cell_id.is_some()
                    && face.neighbor_cell_id.is_none()
                    && face.tag_ids.is_empty()
            })
            .collect();
        assert!(
            untagged.is_empty(),
            "every exterior face is tagged; first missing: {:?}",
            untagged.first()
        );
        assert!(
            mesh.faces
                .iter()
                .filter(|face| !face.tag_ids.is_empty())
                .all(|face| face.neighbor_cell_id.is_none()),
            "interior faces are not boundary-tagged"
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
        let ellipse = document
            .add_primitive_from_drag("ellipse", vec3(0.1, -0.25, 0.0), vec3(0.9, 0.25, 0.0), 1.0)
            .expect("ellipse");
        let domain = document
            .combine(rect, ellipse, "difference")
            .expect("difference");
        document.rename(domain, "fluid").expect("rename");
        document
            .set_domain_root(domain, DomainKind::Fluid)
            .expect("fluid domain");
        // A whole-surface region: projected edges on this owner should carry
        // its tag instead of the wall fallback.
        let owner = {
            let domains = meshable_domains_from_document(&document).expect("domains");
            domains
                .get("fluid")
                .expect("fluid")
                .classify_boundary(&[vec3(-2.0, 0.5, 0.0)], BoundaryBand::UnprojectedSamples)
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
        assert_closed_tagged_2d_boundary(&mesh);

        let domains = meshable_domains_from_document(&document).expect("domains");
        let fluid = domains.get("fluid").expect("fluid");
        let boundary_points: std::collections::BTreeSet<u64> = mesh
            .edges
            .iter()
            .filter(|edge| edge.owner_cell_id.is_some() && edge.neighbor_cell_id.is_none())
            .flat_map(|edge| edge.point_ids.iter().copied())
            .collect();
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
            if boundary_points.contains(&point.id) {
                assert!(
                    value.abs() <= 1e-6,
                    "exterior point {} missed the exact boundary by {value}",
                    point.id
                );
            }
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
            let fluid = domains.get("von_karman_fluid");
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
    fn boundary_marching_samples_are_scriptable() {
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
        document.rename(circle, "pin").expect("rename");
        let domain = document
            .combine(rect, circle, "difference")
            .expect("difference");
        document.rename(domain, "fluid").expect("rename");
        document
            .set_domain_root(domain, DomainKind::Fluid)
            .expect("fluid domain");

        let script = r#"
            let fluid = domains.get("fluid");
            // The keyed listing names the outer loop and the hole's owner.
            let names = fluid.boundaries();
            if "outer" in names && "pin" in names {
                mesh.point(1.0, 0.0, 0.0);
            }
            // The outer rectangle, sampled by name: exact ordered points.
            let outer = fluid.boundary_marching_sample("outer", 24);
            if outer.len() == 24 {
                mesh.point(2.0, 0.0, 0.0);
            }
            // The hole, sampled by the subtracted object's name.
            let hole = fluid.boundary_marching_sample("pin", 16);
            let on_wall = hole.len() == 16;
            for p in hole {
                if fluid.sdf(p[0], p[1], p[2]).abs() > 1e-9 { on_wall = false; }
            }
            if on_wall {
                mesh.point(3.0, 0.0, 0.0);
            }
            // Unknown names error, listing what exists.
            try {
                fluid.boundary_marching_sample("nope", 8);
            } catch (error) {
                if error.contains("outer") && error.contains("pin") {
                    mesh.point(4.0, 0.0, 0.0);
                }
            }
        "#;
        let mesh = run_mesher_script(&document, script).expect("script runs");
        assert_eq!(mesh.points.len(), 4, "keyed sampling delivered to the script");
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
        document.rename(section, "fluid").expect("rename");
        document
            .set_domain_root(section, caso_kernel::roles::DomainKind::Fluid)
            .expect("domain");
        let script = r#"
            let fluid = domains.get("fluid");
            let space = fluid.mesh_space();
            let zone = mesh.zone("fluid", "fluid");
            let b = space.bounds();
            let p0 = space.point(b.a_min, b.b_min);
            let p1 = space.point(b.a_max, b.b_min);
            let p2 = space.point(b.a_max, b.b_max);
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
