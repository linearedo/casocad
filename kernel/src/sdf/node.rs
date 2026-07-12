//! The SDF scene-graph node: one named, identified object wrapping a `Shape`.
//! Mirrors `core/sdf/base.py` (SDFNode), `core/sdf/operators.py`, and
//! `core/sdf/transforms.py`.

use crate::bbox::BoundingBox3D;
use crate::error::{GeometryError, GeometryResult};
use crate::sdf::curtain::NormalCurtain;
use crate::sdf::placed::{PlacedPolyline1D, PlacedSdf1D, PlacedSdf2D};
use crate::sdf::primitives_3d::{
    Box3, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus,
};
use crate::sdf::solid_from_2d::{Extrude, Revolve};
use crate::sdf::tubes::{PolylineTube, QuadraticBezierTube};
use crate::vec3::{vec3, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}

impl RotationAxis {
    pub fn parse(axis: &str) -> GeometryResult<Self> {
        match axis {
            "x" => Ok(Self::X),
            "y" => Ok(Self::Y),
            "z" => Ok(Self::Z),
            _ => Err(GeometryError::new("rotation axis must be x, y, or z")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
        }
    }
}

/// The two operands of an SDF operator (same dimension, checked at build).
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOperands {
    pub left: Box<Node>,
    pub right: Box<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    // 3D primitives
    Sphere(Sphere),
    Box3(Box3),
    Cylinder(Cylinder),
    Cone(Cone),
    CappedCone(CappedCone),
    Pyramid(Pyramid),
    BoxFrame(BoxFrame),
    Torus(Torus),
    PolylineTube(PolylineTube),
    QuadraticBezierTube(QuadraticBezierTube),
    // Exact generators
    Extrude(Extrude),
    Revolve(Revolve),
    // Classification-only ghost geometry
    NormalCurtain(NormalCurtain),
    // Placed lower-dimensional objects
    PlacedSdf2D(PlacedSdf2D),
    PlacedPolyline1D(PlacedPolyline1D),
    PlacedSdf1D(PlacedSdf1D),
    // SDF operators
    Union(BinaryOperands),
    Intersection(BinaryOperands),
    Difference(BinaryOperands),
    Xor(BinaryOperands),
    // Exact transforms
    Translate { child: Box<Node>, offset: Vec3 },
    Scale { child: Box<Node>, factor: f64 },
    Rotate {
        child: Box<Node>,
        axis: RotationAxis,
        angle_degrees: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub name: String,
    pub object_id: u32,
    pub shape: Shape,
}

fn binary_operands(left: Node, right: Node) -> GeometryResult<BinaryOperands> {
    if left.dimension() != right.dimension() {
        return Err(GeometryError::new(
            "boolean operands must have the same dimension",
        ));
    }
    Ok(BinaryOperands {
        left: Box::new(left),
        right: Box::new(right),
    })
}

impl Shape {
    pub fn union(left: Node, right: Node) -> GeometryResult<Shape> {
        Ok(Shape::Union(binary_operands(left, right)?))
    }

    pub fn intersection(left: Node, right: Node) -> GeometryResult<Shape> {
        Ok(Shape::Intersection(binary_operands(left, right)?))
    }

    pub fn difference(left: Node, right: Node) -> GeometryResult<Shape> {
        Ok(Shape::Difference(binary_operands(left, right)?))
    }

    pub fn xor(left: Node, right: Node) -> GeometryResult<Shape> {
        Ok(Shape::Xor(binary_operands(left, right)?))
    }

    pub fn scale(child: Node, factor: f64) -> GeometryResult<Shape> {
        if factor <= 0.0 {
            return Err(GeometryError::new("scale factor must be positive"));
        }
        Ok(Shape::Scale {
            child: Box::new(child),
            factor,
        })
    }
}

impl Node {
    pub fn new(name: impl Into<String>, shape: Shape) -> Node {
        Node {
            name: name.into(),
            object_id: 0,
            shape,
        }
    }

    pub fn with_id(name: impl Into<String>, object_id: u32, shape: Shape) -> Node {
        Node {
            name: name.into(),
            object_id,
            shape,
        }
    }

    /// Coordinate dimension of this visible SDF object (1, 2, or 3).
    pub fn dimension(&self) -> u8 {
        match &self.shape {
            Shape::Sphere(_)
            | Shape::Box3(_)
            | Shape::Cylinder(_)
            | Shape::Cone(_)
            | Shape::CappedCone(_)
            | Shape::Pyramid(_)
            | Shape::BoxFrame(_)
            | Shape::Torus(_)
            | Shape::PolylineTube(_)
            | Shape::QuadraticBezierTube(_)
            | Shape::Extrude(_)
            | Shape::Revolve(_)
            | Shape::NormalCurtain(_) => 3,
            Shape::PlacedSdf2D(_) => 2,
            Shape::PlacedPolyline1D(_) | Shape::PlacedSdf1D(_) => 1,
            Shape::Union(op) | Shape::Intersection(op) | Shape::Difference(op) | Shape::Xor(op) => {
                op.left.dimension()
            }
            Shape::Translate { child, .. }
            | Shape::Scale { child, .. }
            | Shape::Rotate { child, .. } => child.dimension(),
        }
    }

    /// Kind string, matching the Python `SDFNode.kind` values exactly.
    pub fn kind(&self) -> &'static str {
        match &self.shape {
            Shape::Sphere(_) => "sphere",
            Shape::Box3(_) => "box",
            Shape::Cylinder(_) => "cylinder",
            Shape::Cone(_) => "cone",
            Shape::CappedCone(_) => "cappedcone",
            Shape::Pyramid(_) => "pyramid",
            Shape::BoxFrame(_) => "boxframe",
            Shape::Torus(_) => "torus",
            Shape::PolylineTube(_) => "polylinetube",
            Shape::QuadraticBezierTube(tube) => tube.kind(),
            Shape::Extrude(_) => "extrude",
            Shape::Revolve(_) => "revolve",
            Shape::NormalCurtain(_) => "normalcurtain",
            Shape::PlacedSdf2D(_) => "placed_sdf_2d",
            Shape::PlacedPolyline1D(placed) => placed.kind(),
            Shape::PlacedSdf1D(_) => "placed_sdf_1d",
            Shape::Union(_) => "union",
            Shape::Intersection(_) => "intersection",
            Shape::Difference(_) => "difference",
            Shape::Xor(_) => "xor",
            Shape::Translate { .. } => "translate",
            Shape::Scale { .. } => "scale",
            Shape::Rotate { .. } => "rotate",
        }
    }

    pub fn children(&self) -> Vec<&Node> {
        match &self.shape {
            Shape::Union(op) | Shape::Intersection(op) | Shape::Difference(op) | Shape::Xor(op) => {
                vec![&op.left, &op.right]
            }
            Shape::Translate { child, .. }
            | Shape::Scale { child, .. }
            | Shape::Rotate { child, .. } => vec![child],
            Shape::Extrude(extrude) => vec![&extrude.section],
            Shape::Revolve(revolve) => vec![&revolve.section],
            Shape::PlacedSdf2D(placed) => placed.sources.iter().collect(),
            Shape::PlacedSdf1D(placed) => placed.sources.iter().collect(),
            _ => Vec::new(),
        }
    }

    pub fn children_mut(&mut self) -> Vec<&mut Node> {
        match &mut self.shape {
            Shape::Union(op) | Shape::Intersection(op) | Shape::Difference(op) | Shape::Xor(op) => {
                vec![&mut op.left, &mut op.right]
            }
            Shape::Translate { child, .. }
            | Shape::Scale { child, .. }
            | Shape::Rotate { child, .. } => vec![child],
            Shape::Extrude(extrude) => vec![&mut extrude.section],
            Shape::Revolve(revolve) => vec![&mut revolve.section],
            Shape::PlacedSdf2D(placed) => placed.sources.iter_mut().collect(),
            Shape::PlacedSdf1D(placed) => placed.sources.iter_mut().collect(),
            _ => Vec::new(),
        }
    }

    /// Leaf nodes of the subtree (nodes without children), in walk order.
    pub fn leaves(&self) -> Vec<&Node> {
        let children = self.children();
        if children.is_empty() {
            return vec![self];
        }
        children.into_iter().flat_map(|child| child.leaves()).collect()
    }

    /// This node followed by all descendants, depth-first.
    pub fn walk(&self) -> Vec<&Node> {
        let mut nodes = vec![self];
        for child in self.children() {
            nodes.extend(child.walk());
        }
        nodes
    }

    /// Signed distance at one world point. The authoritative evaluation path
    /// (the `to_numpy()` analog); all math in f64.
    pub fn eval_point(&self, p: Vec3) -> f64 {
        match &self.shape {
            Shape::Sphere(shape) => shape.eval_point(p),
            Shape::Box3(shape) => shape.eval_point(p),
            Shape::Cylinder(shape) => shape.eval_point(p),
            Shape::Cone(shape) => shape.eval_point(p),
            Shape::CappedCone(shape) => shape.eval_point(p),
            Shape::Pyramid(shape) => shape.eval_point(p),
            Shape::BoxFrame(shape) => shape.eval_point(p),
            Shape::Torus(shape) => shape.eval_point(p),
            Shape::PolylineTube(shape) => shape.eval_point(p),
            Shape::QuadraticBezierTube(shape) => shape.eval_point(p),
            Shape::Extrude(shape) => shape.eval_point(p),
            Shape::Revolve(shape) => shape.eval_point(p),
            Shape::NormalCurtain(shape) => shape.eval_point(p),
            Shape::PlacedSdf2D(shape) => shape.eval_point(p),
            Shape::PlacedPolyline1D(shape) => shape.eval_point(p),
            Shape::PlacedSdf1D(shape) => shape.eval_point(p),
            Shape::Union(op) => op.left.eval_point(p).min(op.right.eval_point(p)),
            Shape::Intersection(op) => op.left.eval_point(p).max(op.right.eval_point(p)),
            Shape::Difference(op) => op.left.eval_point(p).max(-op.right.eval_point(p)),
            Shape::Xor(op) => {
                let left = op.left.eval_point(p);
                let right = op.right.eval_point(p);
                left.min(right).max(-left.max(right))
            }
            Shape::Translate { child, offset } => child.eval_point(p - *offset),
            Shape::Scale { child, factor } => child.eval_point(p / *factor) * *factor,
            Shape::Rotate {
                child,
                axis,
                angle_degrees,
            } => {
                let c = angle_degrees.to_radians().cos();
                let s = angle_degrees.to_radians().sin();
                let local = match axis {
                    RotationAxis::X => vec3(p.x, c * p.y + s * p.z, -s * p.y + c * p.z),
                    RotationAxis::Y => vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z),
                    RotationAxis::Z => vec3(c * p.x + s * p.y, -s * p.x + c * p.y, p.z),
                };
                child.eval_point(local)
            }
        }
    }

    /// Batched evaluation over many points.
    pub fn eval(&self, points: &[Vec3]) -> Vec<f64> {
        points.iter().map(|p| self.eval_point(*p)).collect()
    }

    /// Internal provisional traversal bound, not part of SDF semantics.
    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        match &self.shape {
            Shape::Sphere(shape) => shape.bounding_box(),
            Shape::Box3(shape) => shape.bounding_box(),
            Shape::Cylinder(shape) => shape.bounding_box(),
            Shape::Cone(shape) => shape.bounding_box(),
            Shape::CappedCone(shape) => shape.bounding_box(),
            Shape::Pyramid(shape) => shape.bounding_box(),
            Shape::BoxFrame(shape) => shape.bounding_box(),
            Shape::Torus(shape) => shape.bounding_box(),
            Shape::PolylineTube(shape) => shape.bounding_box(),
            Shape::QuadraticBezierTube(shape) => shape.bounding_box(),
            Shape::Extrude(shape) => shape.bounding_box(),
            Shape::Revolve(shape) => shape.bounding_box(),
            Shape::NormalCurtain(shape) => shape.bounding_box(),
            Shape::PlacedSdf2D(shape) => padded_bounds(&shape.workplane_corners(), 0.002),
            Shape::PlacedPolyline1D(shape) => padded_bounds(&shape.workplane_corners(), 0.004),
            Shape::PlacedSdf1D(shape) => {
                let (minimum, maximum) = shape.profile.bounds();
                let endpoints = [
                    shape.origin + shape.axis_u * minimum,
                    shape.origin + shape.axis_u * maximum,
                ];
                padded_bounds(&endpoints, 0.004)
            }
            Shape::Union(op) | Shape::Xor(op) => Ok(op
                .left
                .bounding_box()?
                .union(&op.right.bounding_box()?)),
            Shape::Intersection(op) => {
                let left = op.left.bounding_box()?;
                let right = op.right.bounding_box()?;
                Ok(left.intersection(&right).unwrap_or_else(|_| left.union(&right)))
            }
            Shape::Difference(op) => op.left.bounding_box(),
            Shape::Translate { child, offset } => {
                let bounds = child.bounding_box()?;
                BoundingBox3D::new(
                    bounds.x_min + offset.x,
                    bounds.x_max + offset.x,
                    bounds.y_min + offset.y,
                    bounds.y_max + offset.y,
                    bounds.z_min + offset.z,
                    bounds.z_max + offset.z,
                )
            }
            Shape::Scale { child, factor } => {
                let bounds = child.bounding_box()?;
                BoundingBox3D::new(
                    bounds.x_min * factor,
                    bounds.x_max * factor,
                    bounds.y_min * factor,
                    bounds.y_max * factor,
                    bounds.z_min * factor,
                    bounds.z_max * factor,
                )
            }
            Shape::Rotate {
                child,
                axis,
                angle_degrees,
            } => {
                let bounds = child.bounding_box()?;
                let c = angle_degrees.to_radians().cos();
                let s = angle_degrees.to_radians().sin();
                let rotate = |p: Vec3| -> Vec3 {
                    match axis {
                        RotationAxis::X => vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z),
                        RotationAxis::Y => vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z),
                        RotationAxis::Z => vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z),
                    }
                };
                let mut corners = Vec::with_capacity(8);
                for x in [bounds.x_min, bounds.x_max] {
                    for y in [bounds.y_min, bounds.y_max] {
                        for z in [bounds.z_min, bounds.z_max] {
                            corners.push(rotate(vec3(x, y, z)));
                        }
                    }
                }
                BoundingBox3D::from_points(corners)
            }
        }
    }
}

fn padded_bounds(points: &[Vec3], padding: f64) -> GeometryResult<BoundingBox3D> {
    let pad = vec3(padding, padding, padding);
    BoundingBox3D::from_points(points.iter().flat_map(|p| [*p - pad, *p + pad]))
}
