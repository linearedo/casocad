//! Versioned scene.json read/write, ported from `core/serialization.py`.
//!
//! The format is `{"format": "casocad", "version": 1, "unit": "m", ...}` with
//! name-keyed object records referencing each other by key. Loading migrates
//! the legacy boundary-selector schema (volume selectors are inlined into cut
//! chains; interval selectors keep the legacy fields until 2D parity), and
//! saving always emits the new self-contained format.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use crate::error::{GeometryError, GeometryResult};
use crate::frame::Frame;
use crate::roles::DomainKind;
use crate::scene::{
    FluidDomainRecord, ObjectId, OperatorKind, SceneDocument, ScenePayload, TagRef,
};
use crate::sdf::curtain::NormalCurtain;
use crate::sdf::node::{Node, RotationAxis, Shape};
use crate::sdf::primitives_1d::{BooleanOp1D, Profile1D};
use crate::sdf::primitives_2d::{Point2, Profile2D};
use crate::sdf::primitives_3d::{
    Box3, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus,
};
use crate::sdf::solid_from_2d::{Extrude, RevolveAxis};
use crate::sdf::tubes::{CapStyle, PolylineTube, QuadraticBezierTube};
use crate::vec3::{vec3, Vec3};

pub const SCENE_FORMAT_VERSION: u64 = 1;
pub const FORMAT_NAME: &str = "casocad";

const DEFAULT_AXIS_U: Vec3 = vec3(1.0, 0.0, 0.0);
const DEFAULT_AXIS_V: Vec3 = vec3(0.0, 1.0, 0.0);

fn err(message: impl Into<String>) -> GeometryError {
    GeometryError::new(message)
}

// ---------------------------------------------------------------------------
// JSON helpers

fn vec3_json(value: Vec3) -> Value {
    json!([value.x, value.y, value.z])
}

fn point2_json(value: Point2) -> Value {
    json!([value[0], value[1]])
}

fn points3_json(points: &[Vec3]) -> Value {
    Value::Array(points.iter().map(|point| vec3_json(*point)).collect())
}

fn parse_f64(value: &Value) -> GeometryResult<f64> {
    value
        .as_f64()
        .ok_or_else(|| err("expected a number in scene file"))
}

fn parse_vec3(value: &Value) -> GeometryResult<Vec3> {
    let items = value
        .as_array()
        .ok_or_else(|| err("expected a 3D vector"))?;
    if items.len() != 3 {
        return Err(err("expected a 3D vector"));
    }
    Ok(vec3(
        parse_f64(&items[0])?,
        parse_f64(&items[1])?,
        parse_f64(&items[2])?,
    ))
}

fn parse_point2(value: &Value) -> GeometryResult<Point2> {
    let items = value
        .as_array()
        .ok_or_else(|| err("expected a 2D point"))?;
    if items.len() != 2 {
        return Err(err("expected a 2D point"));
    }
    Ok([parse_f64(&items[0])?, parse_f64(&items[1])?])
}

fn parse_points3(value: &Value) -> GeometryResult<Vec<Vec3>> {
    value
        .as_array()
        .ok_or_else(|| err("expected a point list"))?
        .iter()
        .map(parse_vec3)
        .collect()
}

fn parse_points2(value: &Value) -> GeometryResult<Vec<Point2>> {
    value
        .as_array()
        .ok_or_else(|| err("expected a point list"))?
        .iter()
        .map(parse_point2)
        .collect()
}

fn get_str<'a>(record: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    record.get(key).and_then(Value::as_str)
}

fn require_f64(record: &Map<String, Value>, key: &str) -> GeometryResult<f64> {
    parse_f64(
        record
            .get(key)
            .ok_or_else(|| err(format!("missing field {key:?}")))?,
    )
}

fn optional_f64(record: &Map<String, Value>, key: &str, default: f64) -> GeometryResult<f64> {
    match record.get(key) {
        Some(value) => parse_f64(value),
        None => Ok(default),
    }
}

fn optional_vec3(record: &Map<String, Value>, key: &str, default: Vec3) -> GeometryResult<Vec3> {
    match record.get(key) {
        Some(Value::Null) | None => Ok(default),
        Some(value) => parse_vec3(value),
    }
}

fn maybe_vec3(record: &Map<String, Value>, key: &str) -> GeometryResult<Option<Vec3>> {
    match record.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => Ok(Some(parse_vec3(value)?)),
    }
}

// ---------------------------------------------------------------------------
// Axes / frame

fn write_axes(record: &mut Map<String, Value>, frame: &Frame) {
    if frame.is_identity() {
        return;
    }
    record.insert(
        "axes".to_string(),
        json!({
            "u": [frame.u.x, frame.u.y, frame.u.z],
            "v": [frame.v.x, frame.v.y, frame.v.z],
            "w": [frame.w.x, frame.w.y, frame.w.z],
        }),
    );
}

fn read_axes(record: &Map<String, Value>) -> GeometryResult<Frame> {
    if let Some(Value::Object(axes)) = record.get("axes") {
        let u = optional_vec3(axes, "u", DEFAULT_AXIS_U)?;
        let v = optional_vec3(axes, "v", DEFAULT_AXIS_V)?;
        let w = optional_vec3(axes, "w", vec3(0.0, 0.0, 1.0))?;
        return Frame::orthonormal(u, v, w);
    }
    let u = optional_vec3(record, "axis_u", DEFAULT_AXIS_U)?;
    let v = optional_vec3(record, "axis_v", DEFAULT_AXIS_V)?;
    let w = optional_vec3(record, "axis_w", vec3(0.0, 0.0, 1.0))?;
    Frame::orthonormal(u, v, w)
}

// ---------------------------------------------------------------------------
// Profiles

fn profile_2d_to_json(profile: &Profile2D) -> Value {
    let mut record = Map::new();
    match profile {
        Profile2D::Polyline { points } => {
            record.insert("type".into(), "polyline".into());
            record.insert(
                "points".into(),
                Value::Array(points.iter().map(|point| point2_json(*point)).collect()),
            );
        }
        Profile2D::QuadraticBezierCurve { points } => {
            record.insert("type".into(), "quadratic_bezier_curve".into());
            record.insert(
                "points".into(),
                Value::Array(points.iter().map(|point| point2_json(*point)).collect()),
            );
        }
        Profile2D::QuadraticBezierSurface { points } => {
            record.insert("type".into(), "quadratic_bezier_surface".into());
            record.insert(
                "points".into(),
                Value::Array(points.iter().map(|point| point2_json(*point)).collect()),
            );
        }
        Profile2D::Polygon { points } => {
            record.insert("type".into(), "polygon".into());
            record.insert(
                "points".into(),
                Value::Array(points.iter().map(|point| point2_json(*point)).collect()),
            );
        }
        Profile2D::Circle { center, radius } => {
            record.insert("type".into(), "circle".into());
            record.insert("center".into(), point2_json(*center));
            record.insert("radius".into(), json!(radius));
        }
        Profile2D::Rectangle { center, half_size } => {
            record.insert("type".into(), "rectangle".into());
            record.insert("center".into(), point2_json(*center));
            record.insert("size".into(), json!([half_size[0] * 2.0, half_size[1] * 2.0]));
        }
        Profile2D::Square { center, half_size } => {
            record.insert("type".into(), "square".into());
            record.insert("center".into(), point2_json(*center));
            record.insert("half_size".into(), json!(half_size));
        }
        Profile2D::RoundedRectangle {
            center,
            half_size,
            corner_radius,
        } => {
            record.insert("type".into(), "rounded_rectangle".into());
            record.insert("center".into(), point2_json(*center));
            record.insert("size".into(), json!([half_size[0] * 2.0, half_size[1] * 2.0]));
            record.insert("corner_radius".into(), json!(corner_radius));
        }
        Profile2D::Ellipse { center, semi_axes } => {
            record.insert("type".into(), "ellipse".into());
            record.insert("center".into(), point2_json(*center));
            record.insert("semi_axes".into(), point2_json(*semi_axes));
        }
        Profile2D::RegularPolygon {
            center,
            radius,
            side_count,
            rotation,
        } => {
            record.insert("type".into(), "regular_polygon".into());
            record.insert("center".into(), point2_json(*center));
            record.insert("radius".into(), json!(radius));
            record.insert("side_count".into(), json!(side_count));
            record.insert("rotation".into(), json!(rotation));
        }
        Profile2D::Offset { child, offset } => {
            record.insert("type".into(), "offset".into());
            record.insert("child".into(), profile_2d_to_json(child));
            record.insert("offset".into(), point2_json(*offset));
        }
        Profile2D::DistanceOffset { child, offset } => {
            record.insert("type".into(), "distance_offset".into());
            record.insert("child".into(), profile_2d_to_json(child));
            record.insert("offset".into(), json!(offset));
        }
        Profile2D::Binary {
            left,
            right,
            operation,
            smoothing,
        } => {
            record.insert("type".into(), "binary".into());
            record.insert("left".into(), profile_2d_to_json(left));
            record.insert("right".into(), profile_2d_to_json(right));
            record.insert("operation".into(), operation.as_str().into());
            record.insert("smoothing".into(), json!(smoothing));
        }
    }
    Value::Object(record)
}

fn profile_2d_from_json(value: &Value) -> GeometryResult<Profile2D> {
    let record = value
        .as_object()
        .ok_or_else(|| err("2D profile must be a JSON object"))?;
    let profile_type = get_str(record, "type")
        .ok_or_else(|| err("2D profile requires a type"))?;
    match profile_type {
        "polyline" => Profile2D::polyline(parse_points2(
            record.get("points").ok_or_else(|| err("missing points"))?,
        )?),
        "quadratic_bezier_curve" => Profile2D::quadratic_bezier_curve(parse_points2(
            record.get("points").ok_or_else(|| err("missing points"))?,
        )?),
        "quadratic_bezier_surface" => Profile2D::quadratic_bezier_surface(parse_points2(
            record.get("points").ok_or_else(|| err("missing points"))?,
        )?),
        "polygon" => Profile2D::polygon(parse_points2(
            record.get("points").ok_or_else(|| err("missing points"))?,
        )?),
        "circle" => Profile2D::circle(
            parse_point2(record.get("center").ok_or_else(|| err("missing center"))?)?,
            require_f64(record, "radius")?,
        ),
        "rectangle" => Profile2D::rectangle(
            parse_point2(record.get("center").ok_or_else(|| err("missing center"))?)?,
            half_size_2d(record)?,
        ),
        "square" => Profile2D::square(
            parse_point2(record.get("center").ok_or_else(|| err("missing center"))?)?,
            require_f64(record, "half_size")?,
        ),
        "rounded_rectangle" => Profile2D::rounded_rectangle(
            parse_point2(record.get("center").ok_or_else(|| err("missing center"))?)?,
            half_size_2d(record)?,
            require_f64(record, "corner_radius")?,
        ),
        "ellipse" => Profile2D::ellipse(
            parse_point2(record.get("center").ok_or_else(|| err("missing center"))?)?,
            parse_point2(record.get("semi_axes").ok_or_else(|| err("missing semi_axes"))?)?,
        ),
        "regular_polygon" => Profile2D::regular_polygon(
            parse_point2(record.get("center").ok_or_else(|| err("missing center"))?)?,
            require_f64(record, "radius")?,
            require_f64(record, "side_count")? as u32,
            optional_f64(record, "rotation", 0.0)?,
        ),
        "offset" => Ok(Profile2D::Offset {
            child: Box::new(profile_2d_from_json(
                record.get("child").ok_or_else(|| err("missing child"))?,
            )?),
            offset: parse_point2(record.get("offset").ok_or_else(|| err("missing offset"))?)?,
        }),
        "distance_offset" => Profile2D::distance_offset(
            profile_2d_from_json(record.get("child").ok_or_else(|| err("missing child"))?)?,
            require_f64(record, "offset")?,
        ),
        "binary" => Ok(Profile2D::Binary {
            left: Box::new(profile_2d_from_json(
                record.get("left").ok_or_else(|| err("missing left"))?,
            )?),
            right: Box::new(profile_2d_from_json(
                record.get("right").ok_or_else(|| err("missing right"))?,
            )?),
            operation: BooleanOp1D::parse(
                get_str(record, "operation").unwrap_or("union"),
            )?,
            smoothing: optional_f64(record, "smoothing", 0.1)?,
        }),
        other => Err(err(format!("unknown 2D profile type: {other}"))),
    }
}

fn half_size_2d(record: &Map<String, Value>) -> GeometryResult<Point2> {
    if let Some(size) = record.get("size") {
        let size = parse_point2(size)?;
        return Ok([size[0] * 0.5, size[1] * 0.5]);
    }
    parse_point2(
        record
            .get("half_size")
            .ok_or_else(|| err("missing half_size"))?,
    )
}

fn profile_1d_to_json(profile: &Profile1D) -> Value {
    let mut record = Map::new();
    match profile {
        Profile1D::Segment {
            center,
            half_length,
        } => {
            record.insert("type".into(), "segment".into());
            record.insert("center".into(), json!(center));
            record.insert("length".into(), json!(half_length * 2.0));
        }
        Profile1D::Offset { child, offset } => {
            record.insert("type".into(), "offset".into());
            record.insert("child".into(), profile_1d_to_json(child));
            record.insert("offset".into(), json!(offset));
        }
        Profile1D::Binary {
            left,
            right,
            operation,
            smoothing,
        } => {
            record.insert("type".into(), "binary".into());
            record.insert("left".into(), profile_1d_to_json(left));
            record.insert("right".into(), profile_1d_to_json(right));
            record.insert("operation".into(), operation.as_str().into());
            record.insert("smoothing".into(), json!(smoothing));
        }
    }
    Value::Object(record)
}

fn profile_1d_from_json(value: &Value) -> GeometryResult<Profile1D> {
    let record = value
        .as_object()
        .ok_or_else(|| err("1D profile must be a JSON object"))?;
    let profile_type = get_str(record, "type")
        .ok_or_else(|| err("1D profile requires a type"))?;
    match profile_type {
        "segment" => {
            let half_length = if record.contains_key("length") {
                require_f64(record, "length")? * 0.5
            } else {
                require_f64(record, "half_length")?
            };
            Profile1D::segment(optional_f64(record, "center", 0.0)?, half_length)
        }
        "offset" => Ok(Profile1D::Offset {
            child: Box::new(profile_1d_from_json(
                record.get("child").ok_or_else(|| err("missing child"))?,
            )?),
            offset: require_f64(record, "offset")?,
        }),
        "binary" => Ok(Profile1D::Binary {
            left: Box::new(profile_1d_from_json(
                record.get("left").ok_or_else(|| err("missing left"))?,
            )?),
            right: Box::new(profile_1d_from_json(
                record.get("right").ok_or_else(|| err("missing right"))?,
            )?),
            operation: BooleanOp1D::parse(get_str(record, "operation").unwrap_or("union"))?,
            smoothing: optional_f64(record, "smoothing", 0.1)?,
        }),
        other => Err(err(format!("unknown 1D profile type: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// Leaf-shape records (shared by scene payloads and cut ghosts)

fn tube_record(
    record: &mut Map<String, Value>,
    points: &[Vec3],
    radius: f64,
    inner_radius: f64,
    caps: CapStyle,
) {
    record.insert("points".into(), points3_json(points));
    record.insert("radius".into(), json!(radius));
    record.insert("inner_radius".into(), json!(inner_radius));
    record.insert("caps".into(), caps.as_str().into());
}

/// Serialize a ghost (self-contained `Node`) — always carries "name".
/// Leaf shapes plus the recursive `extrude` composite record (casoWASM
/// format extension for one-sided stencil knives — see
/// design_docs/boundary_cutter_exactness.md); never scene references.
pub fn ghost_to_json(node: &Node) -> GeometryResult<Value> {
    let mut record = Map::new();
    match &node.shape {
        Shape::Extrude(extrude) => {
            record.insert("type".into(), "extrude".into());
            record.insert("section".into(), ghost_to_json(&extrude.section)?);
            record.insert("height".into(), json!(extrude.height));
            record.insert("center_offset".into(), json!(extrude.center_offset));
        }
        Shape::Sphere(shape) => {
            record.insert("type".into(), "sphere".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert("radius".into(), json!(shape.radius));
        }
        Shape::Box3(shape) => {
            record.insert("type".into(), "box".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert(
                "size".into(),
                json!([
                    shape.half_size.x * 2.0,
                    shape.half_size.y * 2.0,
                    shape.half_size.z * 2.0
                ]),
            );
            write_axes(&mut record, &shape.frame);
        }
        Shape::BoxFrame(shape) => {
            record.insert("type".into(), "box_frame".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert(
                "size".into(),
                json!([
                    shape.half_size.x * 2.0,
                    shape.half_size.y * 2.0,
                    shape.half_size.z * 2.0
                ]),
            );
            record.insert("thickness".into(), json!(shape.thickness));
            write_axes(&mut record, &shape.frame);
        }
        Shape::Cylinder(shape) => {
            record.insert("type".into(), "cylinder".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert("radius".into(), json!(shape.radius));
            record.insert("height".into(), json!(shape.half_height * 2.0));
            write_axes(&mut record, &shape.frame);
        }
        Shape::CappedCone(shape) => {
            record.insert("type".into(), "capped_cone".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert("radius_a".into(), json!(shape.radius_a));
            record.insert("radius_b".into(), json!(shape.radius_b));
            record.insert("height".into(), json!(shape.half_height * 2.0));
            write_axes(&mut record, &shape.frame);
        }
        Shape::Cone(shape) => {
            record.insert("type".into(), "cone".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert("radius".into(), json!(shape.radius));
            record.insert("height".into(), json!(shape.half_height * 2.0));
            write_axes(&mut record, &shape.frame);
        }
        Shape::Pyramid(shape) => {
            record.insert("type".into(), "pyramid".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert("base_size".into(), json!(shape.base_half_size * 2.0));
            record.insert("height".into(), json!(shape.half_height * 2.0));
            write_axes(&mut record, &shape.frame);
        }
        Shape::Torus(shape) => {
            record.insert("type".into(), "torus".into());
            record.insert("center".into(), vec3_json(shape.center));
            record.insert("major_radius".into(), json!(shape.major_radius));
            record.insert("minor_radius".into(), json!(shape.minor_radius));
            write_axes(&mut record, &shape.frame);
        }
        Shape::PolylineTube(tube) => {
            record.insert("type".into(), "polyline_tube".into());
            tube_record(&mut record, &tube.points, tube.radius, tube.inner_radius, tube.caps);
        }
        Shape::QuadraticBezierTube(tube) => {
            record.insert("type".into(), "quadratic_bezier_tube".into());
            tube_record(&mut record, &tube.points, tube.radius, tube.inner_radius, tube.caps);
        }
        Shape::NormalCurtain(curtain) => {
            record.insert("type".into(), "normal_curtain".into());
            record.insert("points".into(), points3_json(&curtain.points));
            record.insert("normals".into(), points3_json(&curtain.normals));
            record.insert("extent".into(), json!(curtain.extent));
        }
        Shape::PlacedSdf2D(placed) => {
            if !placed.sources.is_empty() {
                return Err(err("boundary cut ghosts must be self-contained leaf shapes"));
            }
            record.insert("type".into(), "placed_sdf_2d".into());
            record.insert("profile".into(), profile_2d_to_json(&placed.profile));
            record.insert("origin".into(), vec3_json(placed.origin));
            record.insert("axis_u".into(), vec3_json(placed.axis_u));
            record.insert("axis_v".into(), vec3_json(placed.axis_v));
        }
        Shape::PlacedPolyline1D(placed) => {
            record.insert("type".into(), "placed_polyline_1d".into());
            record.insert("profile".into(), profile_2d_to_json(&placed.profile));
            record.insert("origin".into(), vec3_json(placed.origin));
            record.insert("axis_u".into(), vec3_json(placed.axis_u));
            record.insert("axis_v".into(), vec3_json(placed.axis_v));
        }
        Shape::PlacedSdf1D(placed) => {
            if !placed.sources.is_empty() {
                return Err(err("boundary cut ghosts must be self-contained leaf shapes"));
            }
            record.insert("type".into(), "placed_sdf_1d".into());
            record.insert("profile".into(), profile_1d_to_json(&placed.profile));
            record.insert("origin".into(), vec3_json(placed.origin));
            record.insert("axis_u".into(), vec3_json(placed.axis_u));
        }
        _ => return Err(err("boundary cut ghosts must be self-contained leaf shapes")),
    }
    let name = if node.name.is_empty() {
        "ghost"
    } else {
        node.name.as_str()
    };
    record.insert("name".into(), name.into());
    Ok(Value::Object(record))
}

/// Parse a ghost record into a self-contained `Node` (object_id 0): a leaf
/// shape or the recursive `extrude` composite.
pub fn ghost_from_json(value: &Value) -> GeometryResult<Node> {
    let record = value
        .as_object()
        .ok_or_else(|| err("ghost record must be a JSON object"))?;
    let name = get_str(record, "name").unwrap_or("ghost").to_string();
    let node_type = get_str(record, "type").ok_or_else(|| err("ghost requires a type"))?;
    // The `extrude` ghost record nests a further ghost record — unlike the
    // scene payload "extrude", which references scene objects by id.
    let shape = match node_type {
        "extrude" => Shape::Extrude(Extrude::new(
            ghost_from_json(
                record
                    .get("section")
                    .ok_or_else(|| err("extrude ghost requires a section"))?,
            )?,
            require_f64(record, "height")?,
            optional_f64(record, "center_offset", 0.0)?,
        )?),
        _ => leaf_shape_from_record(node_type, record)?.ok_or_else(|| {
            err(format!(
                "boundary cut ghosts cannot reference scene objects: {node_type}"
            ))
        })?,
    };
    Ok(Node::new(name, shape))
}

/// Parse a leaf (reference-free) shape record; `Ok(None)` when the type is a
/// composite that needs scene references.
fn leaf_shape_from_record(
    node_type: &str,
    record: &Map<String, Value>,
) -> GeometryResult<Option<Shape>> {
    let center = optional_vec3(record, "center", Vec3::ZERO)?;
    let shape = match node_type {
        "sphere" => Some(Shape::Sphere(Sphere::new(center, require_f64(record, "radius")?)?)),
        "box" => Some(Shape::Box3(Box3::new(
            center,
            half_size_3d(record)?,
            read_axes(record)?,
        )?)),
        "box_frame" => Some(Shape::BoxFrame(BoxFrame::new(
            center,
            half_size_3d(record)?,
            require_f64(record, "thickness")?,
            read_axes(record)?,
        )?)),
        "cylinder" => Some(Shape::Cylinder(Cylinder::new(
            center,
            require_f64(record, "radius")?,
            half_height(record)?,
            read_axes(record)?,
        )?)),
        "capped_cone" => Some(Shape::CappedCone(CappedCone::new(
            center,
            require_f64(record, "radius_a")?,
            require_f64(record, "radius_b")?,
            half_height(record)?,
            read_axes(record)?,
        )?)),
        "cone" => Some(Shape::Cone(Cone::new(
            center,
            require_f64(record, "radius")?,
            half_height(record)?,
            read_axes(record)?,
        )?)),
        "pyramid" => {
            let base_half_size = if record.contains_key("base_size") {
                require_f64(record, "base_size")? * 0.5
            } else {
                require_f64(record, "base_half_size")?
            };
            Some(Shape::Pyramid(Pyramid::new(
                center,
                base_half_size,
                half_height(record)?,
                read_axes(record)?,
            )?))
        }
        "torus" => Some(Shape::Torus(Torus::new(
            center,
            require_f64(record, "major_radius")?,
            require_f64(record, "minor_radius")?,
            read_axes(record)?,
        )?)),
        "polyline_tube" => Some(Shape::PolylineTube(PolylineTube::new(
            parse_points3(record.get("points").ok_or_else(|| err("missing points"))?)?,
            require_f64(record, "radius")?,
            optional_f64(record, "inner_radius", 0.0)?,
            CapStyle::parse(get_str(record, "caps").unwrap_or("round"))?,
        )?)),
        "quadratic_bezier_tube" => Some(Shape::QuadraticBezierTube(QuadraticBezierTube::new(
            parse_points3(record.get("points").ok_or_else(|| err("missing points"))?)?,
            require_f64(record, "radius")?,
            optional_f64(record, "inner_radius", 0.0)?,
            CapStyle::parse(get_str(record, "caps").unwrap_or("round"))?,
        )?)),
        "normal_curtain" => Some(Shape::NormalCurtain(NormalCurtain::new(
            parse_points3(record.get("points").ok_or_else(|| err("missing points"))?)?,
            parse_points3(record.get("normals").ok_or_else(|| err("missing normals"))?)?,
            optional_f64(record, "extent", 4.0)?,
        )?)),
        "placed_sdf_2d" => Some(Shape::PlacedSdf2D(crate::sdf::placed::PlacedSdf2D::new(
            profile_2d_from_json(record.get("profile").ok_or_else(|| err("missing profile"))?)?,
            optional_vec3(record, "origin", Vec3::ZERO)?,
            optional_vec3(record, "axis_u", DEFAULT_AXIS_U)?,
            optional_vec3(record, "axis_v", DEFAULT_AXIS_V)?,
            Vec::new(),
        )?)),
        "placed_polyline_1d" => Some(Shape::PlacedPolyline1D(
            crate::sdf::placed::PlacedPolyline1D::new(
                profile_2d_from_json(record.get("profile").ok_or_else(|| err("missing profile"))?)?,
                optional_vec3(record, "origin", Vec3::ZERO)?,
                optional_vec3(record, "axis_u", DEFAULT_AXIS_U)?,
                optional_vec3(record, "axis_v", DEFAULT_AXIS_V)?,
            )?,
        )),
        "placed_sdf_1d" => Some(Shape::PlacedSdf1D(crate::sdf::placed::PlacedSdf1D::new(
            profile_1d_from_json(record.get("profile").ok_or_else(|| err("missing profile"))?)?,
            optional_vec3(record, "origin", Vec3::ZERO)?,
            optional_vec3(record, "axis_u", DEFAULT_AXIS_U)?,
            Vec::new(),
        )?)),
        _ => None,
    };
    Ok(shape)
}

fn half_size_3d(record: &Map<String, Value>) -> GeometryResult<Vec3> {
    if let Some(size) = record.get("size") {
        let size = parse_vec3(size)?;
        return Ok(size * 0.5);
    }
    parse_vec3(
        record
            .get("half_size")
            .ok_or_else(|| err("missing half_size"))?,
    )
}

fn half_height(record: &Map<String, Value>) -> GeometryResult<f64> {
    if record.contains_key("height") {
        return Ok(require_f64(record, "height")? * 0.5);
    }
    require_f64(record, "half_height")
}

// ---------------------------------------------------------------------------
// Save

struct SaveNames {
    /// object id -> unique file key, insertion-ordered by object id.
    node_keys: BTreeMap<ObjectId, String>,
    region_keys: BTreeMap<ObjectId, String>,
}

fn unique_key(name: &str, used: &mut Vec<String>) -> String {
    let base = {
        let text = name.trim();
        if text.is_empty() {
            "object".to_string()
        } else {
            text.to_string()
        }
    };
    let mut candidate = base.clone();
    let mut suffix = 2;
    while used.contains(&candidate) {
        candidate = format!("{base}_{suffix}");
        suffix += 1;
    }
    used.push(candidate.clone());
    candidate
}

fn scene_names(document: &SceneDocument) -> SaveNames {
    let mut used = Vec::new();
    let mut node_keys = BTreeMap::new();
    // Nodes sorted by object_id, like the Python `_scene_names`.
    let live = document.live_ids();
    let mut sorted = live;
    sorted.sort_unstable();
    for id in sorted {
        if let Ok(object) = document.object(id) {
            node_keys.insert(id, unique_key(&object.name, &mut used));
        }
    }
    let mut region_keys = BTreeMap::new();
    let mut regions: Vec<&BoundaryRegion> = document.boundary_regions.iter().collect();
    regions.sort_by_key(|region| region.object_id);
    for region in regions {
        region_keys.insert(region.object_id, unique_key(&region.name, &mut used));
    }
    SaveNames {
        node_keys,
        region_keys,
    }
}

fn payload_to_json(
    document: &SceneDocument,
    id: ObjectId,
    names: &SaveNames,
) -> GeometryResult<Value> {
    let object = document.object(id)?;
    let key = names
        .node_keys
        .get(&id)
        .ok_or_else(|| err(format!("unnamed object {id}")))?;
    let node_key = |child: &ObjectId| -> GeometryResult<Value> {
        names
            .node_keys
            .get(child)
            .map(|key| Value::String(key.clone()))
            .ok_or_else(|| err(format!("reference to unknown object {child}")))
    };
    let mut record = Map::new();
    match &object.payload {
        ScenePayload::Sphere(_)
        | ScenePayload::Box3(_)
        | ScenePayload::Cylinder(_)
        | ScenePayload::Cone(_)
        | ScenePayload::CappedCone(_)
        | ScenePayload::Pyramid(_)
        | ScenePayload::BoxFrame(_)
        | ScenePayload::Torus(_)
        | ScenePayload::PolylineTube(_)
        | ScenePayload::QuadraticBezierTube(_)
        | ScenePayload::NormalCurtain(_) => {
            // Reuse the ghost writer for the pure-leaf payloads, then strip
            // the always-on ghost "name".
            let shape = match &object.payload {
                ScenePayload::Sphere(shape) => Shape::Sphere(shape.clone()),
                ScenePayload::Box3(shape) => Shape::Box3(shape.clone()),
                ScenePayload::Cylinder(shape) => Shape::Cylinder(shape.clone()),
                ScenePayload::Cone(shape) => Shape::Cone(shape.clone()),
                ScenePayload::CappedCone(shape) => Shape::CappedCone(shape.clone()),
                ScenePayload::Pyramid(shape) => Shape::Pyramid(shape.clone()),
                ScenePayload::BoxFrame(shape) => Shape::BoxFrame(shape.clone()),
                ScenePayload::Torus(shape) => Shape::Torus(shape.clone()),
                ScenePayload::PolylineTube(shape) => Shape::PolylineTube(shape.clone()),
                ScenePayload::QuadraticBezierTube(shape) => {
                    Shape::QuadraticBezierTube(shape.clone())
                }
                ScenePayload::NormalCurtain(shape) => Shape::NormalCurtain(shape.clone()),
                _ => unreachable!(),
            };
            let ghost = ghost_to_json(&Node::new(object.name.clone(), shape))?;
            let Value::Object(mut ghost_record) = ghost else {
                unreachable!()
            };
            ghost_record.remove("name");
            record = ghost_record;
        }
        ScenePayload::Placed2D {
            profile,
            origin,
            axis_u,
            axis_v,
            sources,
        } => {
            record.insert("type".into(), "placed_sdf_2d".into());
            record.insert("profile".into(), profile_2d_to_json(profile));
            record.insert("origin".into(), vec3_json(*origin));
            record.insert("axis_u".into(), vec3_json(*axis_u));
            record.insert("axis_v".into(), vec3_json(*axis_v));
            if !sources.is_empty() {
                let keys: GeometryResult<Vec<Value>> = sources.iter().map(node_key).collect();
                record.insert("sources".into(), Value::Array(keys?));
            }
        }
        ScenePayload::PlacedPolyline1D {
            profile,
            origin,
            axis_u,
            axis_v,
        } => {
            record.insert("type".into(), "placed_polyline_1d".into());
            record.insert("profile".into(), profile_2d_to_json(profile));
            record.insert("origin".into(), vec3_json(*origin));
            record.insert("axis_u".into(), vec3_json(*axis_u));
            record.insert("axis_v".into(), vec3_json(*axis_v));
        }
        ScenePayload::Placed1D {
            profile,
            origin,
            axis_u,
            sources,
        } => {
            record.insert("type".into(), "placed_sdf_1d".into());
            record.insert("profile".into(), profile_1d_to_json(profile));
            record.insert("origin".into(), vec3_json(*origin));
            record.insert("axis_u".into(), vec3_json(*axis_u));
            if !sources.is_empty() {
                let keys: GeometryResult<Vec<Value>> = sources.iter().map(node_key).collect();
                record.insert("sources".into(), Value::Array(keys?));
            }
        }
        ScenePayload::Operator { kind, left, right } => {
            record.insert("type".into(), kind.as_str().into());
            record.insert("left".into(), node_key(left)?);
            record.insert("right".into(), node_key(right)?);
        }
        ScenePayload::Translate { child, offset } => {
            record.insert("type".into(), "translate".into());
            record.insert("object".into(), node_key(child)?);
            record.insert("offset".into(), vec3_json(*offset));
        }
        ScenePayload::Rotate {
            child,
            axis,
            angle_degrees,
        } => {
            record.insert("type".into(), "rotate".into());
            record.insert("object".into(), node_key(child)?);
            record.insert("axis".into(), axis.as_str().into());
            record.insert("angle_degrees".into(), json!(angle_degrees));
        }
        ScenePayload::Scale { child, factor } => {
            record.insert("type".into(), "scale".into());
            record.insert("object".into(), node_key(child)?);
            record.insert("factor".into(), json!(factor));
        }
        ScenePayload::Extrude {
            section,
            height,
            center_offset,
        } => {
            record.insert("type".into(), "extrude".into());
            record.insert("section".into(), node_key(section)?);
            record.insert("height".into(), json!(height));
            if *center_offset != 0.0 {
                record.insert("center_offset".into(), json!(center_offset));
            }
        }
        ScenePayload::Revolve {
            section,
            axis,
            axis_origin,
            axis_direction,
            radial_direction,
            angle_degrees,
        } => {
            record.insert("type".into(), "revolve".into());
            record.insert("section".into(), node_key(section)?);
            record.insert("axis".into(), axis.as_str().into());
            record.insert("angle_degrees".into(), json!(angle_degrees));
            if let Some(origin) = axis_origin {
                record.insert("axis_origin".into(), vec3_json(*origin));
            }
            if let Some(direction) = axis_direction {
                record.insert("axis_direction".into(), vec3_json(*direction));
            }
            if let Some(direction) = radial_direction {
                record.insert("radial_direction".into(), vec3_json(*direction));
            }
        }
    }
    if object.name != *key {
        record.insert("name".into(), object.name.clone().into());
    }
    Ok(Value::Object(record))
}

fn region_to_json(
    region: &BoundaryRegion,
    key: &str,
    names: &SaveNames,
) -> GeometryResult<Value> {
    let owner_key = names
        .node_keys
        .get(&region.owner_object_id)
        .ok_or_else(|| err(format!("region owner {} not in scene", region.owner_object_id)))?;
    let mut record = Map::new();
    record.insert("owner".into(), owner_key.clone().into());
    if region.name != key {
        record.insert("name".into(), region.name.clone().into());
    }
    if let Some(patch) = &region.patch_id {
        record.insert("patch".into(), patch.clone().into());
    }
    if let Some(patch_type) = &region.patch_type {
        record.insert("patch_type".into(), patch_type.clone().into());
    }
    if let Some(direction) = region.outside_direction {
        record.insert("outside_direction".into(), json!(direction));
    }
    if let Some(tag) = &region.tag {
        record.insert("tag".into(), tag.clone().into());
    }
    // Interval selectors (2D curve params) remain in the legacy fields until
    // 2D parity; volume selectors were already migrated into cuts at load.
    if let (Some(selector_id), Some(start)) = (&region.selector_id, region.selector_start) {
        record.insert(
            "selector".into(),
            selector_key(selector_id, names)?.into(),
        );
        if let Some(selector_type) = &region.selector_type {
            record.insert("selector_type".into(), selector_type.clone().into());
        }
        if region.selector_side != CutSide::Inside {
            record.insert("selector_side".into(), region.selector_side.as_str().into());
        }
        record.insert("selector_start".into(), json!(start));
        record.insert(
            "selector_end".into(),
            json!(region.selector_end.unwrap_or(start)),
        );
    }
    if !region.cuts.is_empty() {
        let cuts: GeometryResult<Vec<Value>> = region
            .cuts
            .iter()
            .map(|cut| {
                Ok(json!({
                    "side": cut.side.as_str(),
                    "ghost": ghost_to_json(&cut.ghost)?,
                }))
            })
            .collect();
        record.insert("cuts".into(), Value::Array(cuts?));
    }
    Ok(Value::Object(record))
}

fn selector_key(selector_id: &str, names: &SaveNames) -> GeometryResult<String> {
    let Some(rest) = selector_id.strip_prefix("selector:") else {
        return Ok(selector_id.to_string());
    };
    let object_id: ObjectId = rest
        .parse()
        .map_err(|_| err(format!("invalid selector id: {selector_id}")))?;
    names
        .node_keys
        .get(&object_id)
        .cloned()
        .ok_or_else(|| err(format!("unknown selector object id: {object_id}")))
}

/// Serialize the document to the scene.json v1 `Value` (semantically equal to
/// the Python writer's output).
pub fn scene_to_value(document: &SceneDocument) -> GeometryResult<Value> {
    let names = scene_names(document);
    let mut payload = Map::new();
    payload.insert("format".into(), FORMAT_NAME.into());
    payload.insert("version".into(), json!(SCENE_FORMAT_VERSION));
    payload.insert("unit".into(), "m".into());
    let root_keys: GeometryResult<Vec<Value>> = document
        .roots
        .iter()
        .map(|root| {
            names
                .node_keys
                .get(root)
                .map(|key| Value::String(key.clone()))
                .ok_or_else(|| err(format!("root object {root} not indexed")))
        })
        .collect();
    payload.insert("root_objects".into(), Value::Array(root_keys?));
    let mut objects = Map::new();
    for (id, key) in &names.node_keys {
        objects.insert(key.clone(), payload_to_json(document, *id, &names)?);
    }
    payload.insert("objects".into(), Value::Object(objects));
    if !document.boundary_regions.is_empty() {
        let mut regions = Map::new();
        let mut sorted: Vec<&BoundaryRegion> = document.boundary_regions.iter().collect();
        sorted.sort_by_key(|region| region.object_id);
        for region in sorted {
            let key = names
                .region_keys
                .get(&region.object_id)
                .ok_or_else(|| err("region not indexed"))?;
            regions.insert(key.clone(), region_to_json(region, key, &names)?);
        }
        payload.insert("boundary_regions".into(), Value::Object(regions));
    }
    // Domains: declared kinds plus the fluid root's implicit FLUID kind.
    let mut effective = document.domain_kinds.clone();
    if let Some(fluid) = &document.fluid_domain {
        effective.entry(fluid.root).or_insert(DomainKind::Fluid);
    }
    let mut domains = Map::new();
    for (object_id, kind) in &effective {
        let Some(root_key) = names.node_keys.get(object_id) else {
            continue;
        };
        let mut record = Map::new();
        record.insert("type".into(), kind.as_str().into());
        record.insert("root".into(), root_key.clone().into());
        if let Some(fluid) = &document.fluid_domain {
            if fluid.root == *object_id {
                let tags: GeometryResult<Vec<Value>> = fluid
                    .tags
                    .iter()
                    .map(|tag| match tag {
                        TagRef::Region(id) => names
                            .region_keys
                            .get(id)
                            .map(|key| Value::String(key.clone()))
                            .ok_or_else(|| err(format!("unknown tag region {id}"))),
                        TagRef::Node(id) => names
                            .node_keys
                            .get(id)
                            .map(|key| Value::String(key.clone()))
                            .ok_or_else(|| err(format!("unknown tag node {id}"))),
                    })
                    .collect();
                record.insert("tags".into(), Value::Array(tags?));
                let selectors: GeometryResult<Vec<Value>> = fluid
                    .selectors
                    .iter()
                    .map(|id| {
                        names
                            .node_keys
                            .get(id)
                            .map(|key| Value::String(key.clone()))
                            .ok_or_else(|| err(format!("unknown selector {id}")))
                    })
                    .collect();
                record.insert("selectors".into(), Value::Array(selectors?));
            }
        }
        domains.insert(root_key.clone(), Value::Object(record));
    }
    if !domains.is_empty() {
        payload.insert("domains".into(), Value::Object(domains));
    }
    Ok(Value::Object(payload))
}

pub fn save_scene_to_string(document: &SceneDocument) -> GeometryResult<String> {
    let value = scene_to_value(document)?;
    serde_json::to_string_pretty(&value)
        .map(|text| text + "\n")
        .map_err(|error| err(format!("scene serialization failed: {error}")))
}

// ---------------------------------------------------------------------------
// Load

struct Loader<'a> {
    raw_objects: &'a Map<String, Value>,
    built: BTreeMap<String, ObjectId>,
    building: Vec<String>,
    document: SceneDocument,
    next_object_id: ObjectId,
}

impl<'a> Loader<'a> {
    fn allocate(&mut self) -> GeometryResult<ObjectId> {
        let id = self.next_object_id;
        self.next_object_id += 1;
        if id > crate::scene::MAX_OBJECT_ID {
            return Err(err("maximum SDF object count exceeded"));
        }
        Ok(id)
    }

    fn build(&mut self, name: &str) -> GeometryResult<ObjectId> {
        if let Some(id) = self.built.get(name) {
            return Ok(*id);
        }
        if self.building.iter().any(|pending| pending == name) {
            return Err(err(format!("circular object reference: {name}")));
        }
        let record = self
            .raw_objects
            .get(name)
            .ok_or_else(|| err(format!("unknown object reference: {name}")))?
            .as_object()
            .ok_or_else(|| err(format!("object '{name}' must be a JSON object")))?
            .clone();
        self.building.push(name.to_string());
        // Ids are allocated at build entry, before children — matching the
        // Python loader's argument-evaluation order.
        let object_id = self.allocate()?;
        self.built.insert(name.to_string(), object_id);
        let display_name = get_str(&record, "name").unwrap_or(name).to_string();
        let payload = self.payload_from_record(&record)?;
        self.document
            .insert_object_with_id(display_name, object_id, payload)?;
        self.building.pop();
        Ok(object_id)
    }

    fn build_list(&mut self, record: &Map<String, Value>, key: &str) -> GeometryResult<Vec<ObjectId>> {
        let mut ids = Vec::new();
        if let Some(Value::Array(items)) = record.get(key) {
            for item in items {
                let name = item
                    .as_str()
                    .ok_or_else(|| err(format!("{key} entries must be names")))?;
                ids.push(self.build(name)?);
            }
        }
        Ok(ids)
    }

    fn payload_from_record(&mut self, record: &Map<String, Value>) -> GeometryResult<ScenePayload> {
        let node_type = get_str(record, "type")
            .ok_or_else(|| err("object requires a type"))?
            .to_string();
        if let Some(shape) = leaf_shape_from_record(&node_type, record)? {
            return Ok(match shape {
                Shape::Sphere(shape) => ScenePayload::Sphere(shape),
                Shape::Box3(shape) => ScenePayload::Box3(shape),
                Shape::Cylinder(shape) => ScenePayload::Cylinder(shape),
                Shape::Cone(shape) => ScenePayload::Cone(shape),
                Shape::CappedCone(shape) => ScenePayload::CappedCone(shape),
                Shape::Pyramid(shape) => ScenePayload::Pyramid(shape),
                Shape::BoxFrame(shape) => ScenePayload::BoxFrame(shape),
                Shape::Torus(shape) => ScenePayload::Torus(shape),
                Shape::PolylineTube(shape) => ScenePayload::PolylineTube(shape),
                Shape::QuadraticBezierTube(shape) => ScenePayload::QuadraticBezierTube(shape),
                Shape::NormalCurtain(shape) => ScenePayload::NormalCurtain(shape),
                Shape::PlacedSdf2D(placed) => ScenePayload::Placed2D {
                    profile: placed.profile,
                    origin: placed.origin,
                    axis_u: placed.axis_u,
                    axis_v: placed.axis_v,
                    sources: self.build_list(record, "sources")?,
                },
                Shape::PlacedPolyline1D(placed) => ScenePayload::PlacedPolyline1D {
                    profile: placed.profile,
                    origin: placed.origin,
                    axis_u: placed.axis_u,
                    axis_v: placed.axis_v,
                },
                Shape::PlacedSdf1D(placed) => ScenePayload::Placed1D {
                    profile: placed.profile,
                    origin: placed.origin,
                    axis_u: placed.axis_u,
                    sources: self.build_list(record, "sources")?,
                },
                _ => unreachable!("leaf_shape_from_record yields leaves only"),
            });
        }
        let child_ref = |loader: &mut Self, key: &str| -> GeometryResult<ObjectId> {
            let name = get_str(record, key)
                .ok_or_else(|| err(format!("missing reference {key:?}")))?
                .to_string();
            loader.build(&name)
        };
        match node_type.as_str() {
            "union" | "intersection" | "difference" | "xor" => Ok(ScenePayload::Operator {
                kind: OperatorKind::parse(&node_type)?,
                left: child_ref(self, "left")?,
                right: child_ref(self, "right")?,
            }),
            "translate" => Ok(ScenePayload::Translate {
                child: child_ref(self, "object")?,
                offset: parse_vec3(record.get("offset").ok_or_else(|| err("missing offset"))?)?,
            }),
            "rotate" => Ok(ScenePayload::Rotate {
                child: child_ref(self, "object")?,
                axis: RotationAxis::parse(get_str(record, "axis").unwrap_or("y"))?,
                angle_degrees: require_f64(record, "angle_degrees")?,
            }),
            "scale" => Ok(ScenePayload::Scale {
                child: child_ref(self, "object")?,
                factor: require_f64(record, "factor")?,
            }),
            "extrude" => Ok(ScenePayload::Extrude {
                section: child_ref(self, "section")?,
                height: require_f64(record, "height")?,
                center_offset: optional_f64(record, "center_offset", 0.0)?,
            }),
            "revolve" => Ok(ScenePayload::Revolve {
                section: child_ref(self, "section")?,
                axis: RevolveAxis::parse(get_str(record, "axis").unwrap_or("v"))?,
                axis_origin: maybe_vec3(record, "axis_origin")?,
                axis_direction: maybe_vec3(record, "axis_direction")?,
                radial_direction: maybe_vec3(record, "radial_direction")?,
                angle_degrees: optional_f64(record, "angle_degrees", 360.0)?,
            }),
            other => Err(err(format!("unknown SDF node type: {other}"))),
        }
    }
}

/// Load a scene.json v1 payload, migrating the legacy boundary-selector
/// schema exactly like the Python loader.
pub fn load_scene_from_str(text: &str) -> GeometryResult<SceneDocument> {
    let payload: Value = serde_json::from_str(text)
        .map_err(|error| err(format!("invalid scene JSON: {error}")))?;
    let payload = payload
        .as_object()
        .ok_or_else(|| err("scene file must be a JSON object"))?;
    if get_str(payload, "format") != Some(FORMAT_NAME) {
        return Err(err("not a casoCAD scene file"));
    }
    if payload.get("version").and_then(Value::as_u64) != Some(SCENE_FORMAT_VERSION) {
        return Err(err(format!(
            "unsupported scene version: {:?}",
            payload.get("version")
        )));
    }
    let empty = Map::new();
    let raw_objects = match payload.get("objects") {
        Some(Value::Object(objects)) => objects,
        None => &empty,
        Some(_) => return Err(err("casoCAD scene objects must be a name-keyed object")),
    };
    let mut loader = Loader {
        raw_objects,
        built: BTreeMap::new(),
        building: Vec::new(),
        document: SceneDocument::new(),
        next_object_id: 1,
    };
    let root_names = match payload.get("root_objects") {
        Some(Value::Array(items)) => items.clone(),
        None => Vec::new(),
        Some(_) => return Err(err("root_objects must be a list")),
    };
    let mut roots = Vec::new();
    for name in &root_names {
        let name = name
            .as_str()
            .ok_or_else(|| err("root_objects entries must be names"))?;
        roots.push(loader.build(name)?);
    }
    loader.document.roots = roots;

    // Boundary regions (with legacy selector migration).
    let mut regions_by_name: BTreeMap<String, usize> = BTreeMap::new();
    if let Some(raw_regions) = payload.get("boundary_regions") {
        let raw_regions = raw_regions
            .as_object()
            .ok_or_else(|| err("boundary_regions must be a name-keyed object"))?;
        for (name, raw_region) in raw_regions {
            let record = raw_region
                .as_object()
                .ok_or_else(|| err(format!("boundary region '{name}' must be a JSON object")))?;
            let object_id = loader.allocate()?;
            let region = region_from_record(name, record, &mut loader, object_id)?;
            regions_by_name.insert(name.clone(), loader.document.boundary_regions.len());
            loader.document.boundary_regions.push(region);
        }
    }

    // Domains.
    if let Some(raw_domains) = payload.get("domains") {
        let raw_domains = raw_domains
            .as_object()
            .ok_or_else(|| err("domains must be a name-keyed object"))?;
        for (domain_key, domain_record) in raw_domains {
            let record = domain_record
                .as_object()
                .ok_or_else(|| err(format!("domain '{domain_key}' must be a JSON object")))?;
            let root_name = get_str(record, "root")
                .ok_or_else(|| err(format!("domain '{domain_key}' requires a root")))?
                .to_string();
            let root = loader.build(&root_name)?;
            let kind = DomainKind::parse(get_str(record, "type").unwrap_or("fluid"))
                .map_err(|_| {
                    err(format!(
                        "domain '{domain_key}' has unknown type {:?}",
                        record.get("type")
                    ))
                })?;
            loader.document.domain_kinds.insert(root, kind);
            if kind != DomainKind::Fluid {
                continue;
            }
            let mut tags = Vec::new();
            if let Some(Value::Array(raw_tags)) = record.get("tags") {
                for tag_name in raw_tags {
                    let key = tag_name
                        .as_str()
                        .ok_or_else(|| err("fluid tags must be names"))?;
                    if let Some(index) = regions_by_name.get(key) {
                        tags.push(TagRef::Region(
                            loader.document.boundary_regions[*index].object_id,
                        ));
                    } else {
                        let tag_id = loader.build(key)?;
                        let payload = &loader.document.object(tag_id)?.payload;
                        if !matches!(
                            payload,
                            ScenePayload::Placed1D { .. }
                                | ScenePayload::PlacedPolyline1D { .. }
                                | ScenePayload::Placed2D { .. }
                        ) {
                            return Err(err(format!(
                                "fluid tag '{key}' has an unsupported dimension"
                            )));
                        }
                        tags.push(TagRef::Node(tag_id));
                    }
                }
            }
            let mut selectors = Vec::new();
            if let Some(Value::Array(raw_selectors)) = record.get("selectors") {
                for selector_name in raw_selectors {
                    let key = selector_name
                        .as_str()
                        .ok_or_else(|| err("fluid selectors must be names"))?;
                    selectors.push(loader.build(key)?);
                }
            }
            if loader.document.fluid_domain.is_none() {
                loader.document.fluid_domain = Some(FluidDomainRecord {
                    root,
                    tags,
                    selectors,
                });
            }
        }
    }

    let mut document = loader.document;
    document.bump_next_object_id(loader.next_object_id.saturating_sub(1));
    drop_orphaned_internal_selectors(&mut document);
    Ok(document)
}

fn region_from_record(
    key: &str,
    record: &Map<String, Value>,
    loader: &mut Loader<'_>,
    object_id: ObjectId,
) -> GeometryResult<BoundaryRegion> {
    let owner_name = get_str(record, "owner")
        .ok_or_else(|| err(format!("boundary region '{key}' requires an owner")))?
        .to_string();
    let owner = loader.build(&owner_name)?;
    let mut cuts = Vec::new();
    let mut selector_id = None;
    let is_interval = record
        .get("selector_start")
        .is_some_and(|value| !value.is_null());
    if let Some(selector_name) = get_str(record, "selector") {
        let selector_name = selector_name.to_string();
        let selector = loader.build(&selector_name)?;
        if is_interval {
            // 2D interval selectors stay legacy until 2D parity (v2 §9).
            selector_id = Some(format!("selector:{selector}"));
        } else {
            // One-way migration: a legacy volume selector becomes the first
            // entry of the cut chain; the hidden node is dropped afterwards.
            let mut knife = loader.document.build_node(selector)?;
            knife.object_id = 0;
            let side = CutSide::parse(get_str(record, "selector_side").unwrap_or("inside"))?;
            cuts.push(BoundaryCut { side, ghost: knife });
        }
    }
    if let Some(raw_cuts) = record.get("cuts") {
        let raw_cuts = raw_cuts
            .as_array()
            .ok_or_else(|| err(format!("boundary region '{key}' cuts must be a list")))?;
        for raw_cut in raw_cuts {
            let cut = raw_cut
                .as_object()
                .filter(|cut| cut.get("ghost").is_some_and(Value::is_object))
                .ok_or_else(|| {
                    err(format!(
                        "boundary region '{key}' cuts must be objects with a ghost record"
                    ))
                })?;
            cuts.push(BoundaryCut {
                side: CutSide::parse(get_str(cut, "side").unwrap_or("inside"))?,
                ghost: ghost_from_json(cut.get("ghost").expect("checked above"))?,
            });
        }
    }
    let region = BoundaryRegion {
        name: get_str(record, "name").unwrap_or(key).to_string(),
        object_id,
        owner_object_id: owner,
        outside_direction: record
            .get("outside_direction")
            .and_then(Value::as_u64)
            .map(|value| value as u8),
        patch_id: get_str(record, "patch").map(str::to_string),
        patch_type: get_str(record, "patch_type").map(str::to_string),
        selector_id,
        selector_type: if is_interval {
            get_str(record, "selector_type").map(str::to_string)
        } else {
            None
        },
        selector_side: CutSide::parse(get_str(record, "selector_side").unwrap_or("inside"))?,
        selector_start: match record.get("selector_start") {
            Some(Value::Null) | None => None,
            Some(value) => Some(parse_f64(value)?),
        },
        selector_end: match record.get("selector_end") {
            Some(Value::Null) | None => None,
            Some(value) => Some(parse_f64(value)?),
        },
        cuts,
        tag: get_str(record, "tag").map(str::to_string),
    };
    region.validate()?;
    Ok(region)
}

/// Remove hidden `__boundary_selector_*` nodes once no region references
/// them (Python `_drop_orphaned_internal_selectors`).
fn drop_orphaned_internal_selectors(document: &mut SceneDocument) {
    let mut referenced: Vec<ObjectId> = Vec::new();
    for region in &document.boundary_regions {
        if let Some(selector_id) = &region.selector_id {
            if let Some(rest) = selector_id.strip_prefix("selector:") {
                if let Ok(id) = rest.parse::<ObjectId>() {
                    referenced.push(id);
                }
            }
        }
    }
    let doomed: Vec<ObjectId> = document
        .roots
        .iter()
        .copied()
        .filter(|id| {
            document
                .object(*id)
                .map(|object| {
                    SceneDocument::is_internal_scene_node(&object.name)
                        && !referenced.contains(id)
                })
                .unwrap_or(false)
        })
        .collect();
    document.roots.retain(|id| !doomed.contains(id));
    if let Some(fluid) = document.fluid_domain.as_mut() {
        fluid
            .selectors
            .retain(|id| referenced.contains(id) || !doomed.contains(id));
    }
    document.refresh_fluid_domain();
}
