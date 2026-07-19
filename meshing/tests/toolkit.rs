//! Toolkit gates (`design_docs/meshing_toolkit.md` §7-§8): exact tagged 2D
//! boundary loops and the analytic sizing field.

use caso_meshing::toolkit::{
    boundary_loops, boundary_marching_sample, boundary_names, SizingBand, SizingField, SizingSpec,
};

use caso_kernel::boundary_paths::point_knife;
use caso_kernel::meshing::{meshable_domains_from_document, BoundaryBand, MeshableDomain};
use caso_kernel::roles::DomainKind;
use caso_kernel::scene::SceneDocument;
use caso_kernel::sdf::node::RotationAxis;
use caso_kernel::vec3::{vec3, Vec3};

/// Rectangle (4 x 2, centred) minus a circle (r 0.3 at (0.5, 0)), fluid.
fn rectangle_with_hole() -> SceneDocument {
    let mut document = SceneDocument::new();
    let rect = document
        .add_primitive_from_drag("rectangle", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    let circle = document
        .add_primitive_from_drag("circle", vec3(0.2, -0.3, 0.0), vec3(0.8, 0.3, 0.0), 1.0)
        .expect("circle");
    let domain = document
        .combine(rect, circle, "difference")
        .expect("difference");
    document.rename(domain, "fluid").expect("rename");
    document
        .set_domain_root(domain, DomainKind::Fluid)
        .expect("fluid domain");
    document
}

fn fluid(document: &SceneDocument) -> MeshableDomain {
    meshable_domains_from_document(document)
        .expect("domains")
        .get("fluid")
        .expect("fluid")
        .clone()
}

fn loop_length(chain: &caso_meshing::toolkit::BoundaryLoop) -> f64 {
    chain
        .spans
        .iter()
        .flat_map(|span| span.points.windows(2))
        .map(|pair| (pair[1] - pair[0]).length())
        .sum()
}

#[test]
fn loops_of_rectangle_with_hole() {
    let document = rectangle_with_hole();
    let domain = fluid(&document);
    let loops = boundary_loops(&domain, 64).expect("loops");
    assert_eq!(loops.len(), 2, "outer rectangle + hole");

    let outer = loops.iter().find(|chain| chain.is_outer).expect("outer");
    let hole = loops.iter().find(|chain| !chain.is_outer).expect("hole");

    // Outer loop: the exact rectangle, corners included, CCW, area 8.
    assert!((outer.signed_area - 8.0).abs() < 1e-9);
    let outer_points: Vec<Vec3> = outer
        .spans
        .iter()
        .flat_map(|span| span.points.iter().copied())
        .collect();
    for corner in [
        vec3(-2.0, -1.0, 0.0),
        vec3(2.0, -1.0, 0.0),
        vec3(2.0, 1.0, 0.0),
        vec3(-2.0, 1.0, 0.0),
    ] {
        assert!(
            outer_points
                .iter()
                .any(|point| (*point - corner).length() < 1e-12),
            "exact corner {corner:?}"
        );
    }

    // Hole: clockwise (material on the left), area close to -pi r^2 from
    // the sampled polygon (inscribed, so slightly smaller in magnitude).
    // Drag semantics: the circle radius is the drag half-diagonal.
    let radius = (vec3(0.8, 0.3, 0.0) - vec3(0.5, 0.0, 0.0)).length();
    let circle_area = std::f64::consts::PI * radius * radius;
    assert!(hole.signed_area < 0.0, "hole runs clockwise");
    assert!(hole.signed_area.abs() < circle_area);
    assert!(hole.signed_area.abs() > 0.98 * circle_area);

    // Chains are closed and welded: every span tail is the next span's head.
    for chain in &loops {
        let count = chain.spans.len();
        for index in 0..count {
            let tail = *chain.spans[index]
                .points
                .last()
                .expect("span has points");
            let head = chain.spans[(index + 1) % count].points[0];
            assert_eq!(tail, head, "welded junction is bitwise-shared");
        }
    }
}

#[test]
fn all_loop_points_lie_on_the_domain_boundary() {
    let document = rectangle_with_hole();
    let domain = fluid(&document);
    let root = domain.region_node();
    let band = 1e-8 * domain.bounds.diagonal();
    for chain in boundary_loops(&domain, 64).expect("loops") {
        for span in &chain.spans {
            for point in &span.points {
                assert!(
                    root.eval_point(*point).abs() <= band,
                    "loop point off the boundary: {point:?}"
                );
            }
        }
    }
}

#[test]
fn spans_split_exactly_at_region_junctions() {
    let mut document = rectangle_with_hole();
    let fluid_root = document.fluid_domain.as_ref().expect("fluid").root;
    let root = document.build_node(fluid_root).expect("root");
    // Tag the left edge as a region, split it with a point knife at y=0.25.
    let owner = {
        let domains = meshable_domains_from_document(&document).expect("domains");
        let domain = domains.get("fluid").expect("fluid");
        domain.classify_boundary(&[vec3(-2.0, 0.5, 0.0)], BoundaryBand::UnprojectedSamples).expect("classify")[0]
            .owner_object_id
    };
    let region_id = document
        .add_boundary_region(owner, None, None, None)
        .expect("region");
    let click = vec3(-2.0, 0.25, 0.0);
    let ghost = point_knife(&root, click).expect("point knife");
    document
        .split_boundary_region(region_id, &ghost, None)
        .expect("split");

    let domain = fluid(&document);
    let loops = boundary_loops(&domain, 64).expect("loops");
    // Some span boundary must now sit on the knife's zero set (the label
    // flip happens within the tight classification band of it).
    let knife_band = 1e-8 * domain.bounds.diagonal();
    let junction_found = loops.iter().flat_map(|chain| &chain.spans).any(|span| {
        [span.points[0], *span.points.last().expect("points")]
            .iter()
            .any(|endpoint| ghost.eval_point(*endpoint).abs() <= knife_band)
    });
    assert!(junction_found, "a span endpoint lies on the knife zero set");
    // And the two sides of the junction carry different regions.
    let names: std::collections::BTreeSet<Option<&str>> = loops
        .iter()
        .flat_map(|chain| &chain.spans)
        .map(|span| span.region_name.as_deref())
        .collect();
    assert!(names.len() >= 2, "regions differ across the junction: {names:?}");
}

#[test]
fn rotated_domain_loops_are_frame_invariant() {
    let document = rectangle_with_hole();
    let straight = fluid(&document);

    let mut rotated_document = rectangle_with_hole();
    let fluid_root = rotated_document
        .fluid_domain
        .as_ref()
        .expect("fluid")
        .root;
    rotated_document
        .rotate_object(fluid_root, RotationAxis::Z, 90.0, Some(Vec3::ZERO))
        .expect("rotate");
    let rotated = fluid(&rotated_document);

    let mut straight_loops = boundary_loops(&straight, 64).expect("loops");
    let mut rotated_loops = boundary_loops(&rotated, 64).expect("loops");
    let key = |chain: &caso_meshing::toolkit::BoundaryLoop| chain.is_outer;
    straight_loops.sort_by_key(|chain| key(chain));
    rotated_loops.sort_by_key(|chain| key(chain));
    assert_eq!(straight_loops.len(), rotated_loops.len());
    for (a, b) in straight_loops.iter().zip(&rotated_loops) {
        assert!((loop_length(a) - loop_length(b)).abs() < 1e-9);
        assert!((a.signed_area - b.signed_area).abs() < 1e-9);
    }
}

/// The rectangle-minus-circle fixture with the circle ALSO marked as a
/// solid domain: the fluid region becomes a scene-level difference of two
/// placed 2D nodes (additive base minus the exact circle primitive).
fn rectangle_with_marked_hole() -> (SceneDocument, u32) {
    let mut document = SceneDocument::new();
    let rect = document
        .add_primitive_from_drag("rectangle", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
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
    document.rename(domain, "fluid").expect("rename");
    document
        .set_domain_root(domain, DomainKind::Fluid)
        .expect("fluid domain");
    (document, circle)
}

#[test]
fn loops_with_a_marked_solid_hole() {
    let (document, circle_id) = rectangle_with_marked_hole();
    let domains = meshable_domains_from_document(&document).expect("domains");

    // Fluid: the same two loops as the unmarked fixture — each patch
    // contributes exactly once (no duplicate arcs from the double
    // representation of the subtracted circle).
    let fluid_domain = domains.get("fluid").expect("fluid").clone();
    let loops = boundary_loops(&fluid_domain, 64).expect("loops");
    assert_eq!(loops.len(), 2, "outer rectangle + hole");
    let outer = loops.iter().find(|chain| chain.is_outer).expect("outer");
    let hole = loops.iter().find(|chain| !chain.is_outer).expect("hole");
    assert!((outer.signed_area - 8.0).abs() < 1e-9);
    let radius = (vec3(0.8, 0.3, 0.0) - vec3(0.5, 0.0, 0.0)).length();
    let circle_area = std::f64::consts::PI * radius * radius;
    assert!(hole.signed_area < 0.0, "hole runs clockwise");
    assert!(hole.signed_area.abs() > 0.98 * circle_area);
    assert!(hole.signed_area.abs() < circle_area);
    // The hole boundary is owned by the subtracted solid's leaf.
    for span in &hole.spans {
        assert_eq!(span.owner_object_id, circle_id);
    }

    // The solid pin domain has its own single outer loop.
    let pin = domains.get("pin").expect("pin").clone();
    let pin_loops = boundary_loops(&pin, 64).expect("pin loops");
    assert_eq!(pin_loops.len(), 1);
    assert!(pin_loops[0].is_outer, "solid outline runs CCW");
    assert!(pin_loops[0].signed_area > 0.98 * circle_area);
    assert!(pin_loops[0].signed_area < circle_area);
}

#[test]
fn boundary_names_list_outer_regions_and_owners() {
    let (document, _circle_id) = rectangle_with_marked_hole();
    let domains = meshable_domains_from_document(&document).expect("domains");

    let fluid_names = boundary_names(domains.get("fluid").expect("fluid")).expect("names");
    assert_eq!(fluid_names[0], "outer", "the outer loop is always first");
    assert!(fluid_names.iter().any(|name| name == "pin"), "{fluid_names:?}");

    let pin_names = boundary_names(domains.get("pin").expect("pin")).expect("names");
    assert_eq!(pin_names, vec!["outer".to_string(), "pin".to_string()]);
}

#[test]
fn marching_sample_outer_is_exact_ordered_and_even() {
    let (document, _circle_id) = rectangle_with_marked_hole();
    let domains = meshable_domains_from_document(&document).expect("domains");
    let pin = domains.get("pin").expect("pin");

    let sample = boundary_marching_sample(pin, "outer", 16).expect("sample");
    assert_eq!(sample.len(), 16, "closed loop: npoints, head not repeated");
    let band = 1e-8 * pin.bounds.diagonal();
    let root = pin.region_node();
    for point in &sample {
        assert!(root.eval_point(*point).abs() <= band, "exact vertex {point:?}");
    }
    // CCW (material on the left of an outer loop) and roughly even spacing.
    let mut area = 0.0;
    let mut shortest = f64::INFINITY;
    let mut longest = 0.0f64;
    for index in 0..sample.len() {
        let a = sample[index];
        let b = sample[(index + 1) % sample.len()];
        area += a.x * b.y - b.x * a.y;
        let chord = (b - a).length();
        shortest = shortest.min(chord);
        longest = longest.max(chord);
    }
    assert!(area > 0.0, "solid outline runs CCW");
    assert!(longest / shortest < 3.0, "even spacing: {shortest}..{longest}");
}

#[test]
fn marching_sample_reaches_the_hole_by_owner_name() {
    let (document, _circle_id) = rectangle_with_marked_hole();
    let domains = meshable_domains_from_document(&document).expect("domains");
    let fluid_domain = domains.get("fluid").expect("fluid");

    let sample = boundary_marching_sample(fluid_domain, "pin", 16).expect("sample");
    assert_eq!(sample.len(), 16);
    let radius = (vec3(0.8, 0.3, 0.0) - vec3(0.5, 0.0, 0.0)).length();
    for point in &sample {
        let radial = (*point - vec3(0.5, 0.0, 0.0)).length();
        assert!((radial - radius).abs() < 1e-8, "on the circle: {point:?}");
    }
    // The hole runs CW: material (the fluid) on the left.
    let mut area = 0.0;
    for index in 0..sample.len() {
        let a = sample[index];
        let b = sample[(index + 1) % sample.len()];
        area += a.x * b.y - b.x * a.y;
    }
    assert!(area < 0.0, "hole sample runs CW");
}

#[test]
fn marching_sample_region_is_open_with_exact_endpoints() {
    // A plain rectangle: with a hole present, a direction-scoped region
    // also claims the hole arc facing the same way (correctly reported as
    // "not a single connected curve"), so the clean open piece needs a
    // hole-free domain.
    let mut document = SceneDocument::new();
    let rect = document
        .add_primitive_from_drag("rectangle", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    document.rename(rect, "fluid").expect("rename");
    document
        .set_domain_root(rect, DomainKind::Fluid)
        .expect("fluid domain");
    // Tag the left edge (-axis_u direction) of the rectangle as "west".
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
        .add_boundary_region(owner, Some(0), None, None)
        .expect("west region");
    document
        .boundary_regions
        .iter_mut()
        .find(|region| region.object_id == region_id)
        .expect("region")
        .name = "west".to_string();

    let domain = fluid(&document);
    let names = boundary_names(&domain).expect("names");
    assert!(names.iter().any(|name| name == "west"), "{names:?}");

    let sample = boundary_marching_sample(&domain, "west", 5).expect("sample");
    assert_eq!(sample.len(), 5, "open piece: both endpoints included");
    // Endpoints land at the region's true extent: the classification
    // transition sits within the tight band (~1e-9·diag) of each corner
    // (the exact corner has no single outward direction, so it is not part
    // of a direction-scoped region).
    for corner in [vec3(-2.0, 1.0, 0.0), vec3(-2.0, -1.0, 0.0)] {
        assert!(
            [sample[0], sample[4]]
                .iter()
                .any(|point| (*point - corner).length() < 1e-6),
            "endpoint at corner {corner:?}"
        );
    }
    for point in &sample {
        assert!((point.x + 2.0).abs() < 1e-12, "on the left edge: {point:?}");
    }

    // On the holed fixture the same direction region ALSO claims the arc of
    // the hole facing -x: honestly refused as two pieces.
    let holed = fluid(&{
        let mut document = rectangle_with_hole();
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
            .add_boundary_region(owner, Some(0), None, None)
            .expect("west region");
        document
            .boundary_regions
            .iter_mut()
            .find(|region| region.object_id == region_id)
            .expect("region")
            .name = "west".to_string();
        document
    });
    let split = boundary_marching_sample(&holed, "west", 5).expect_err("two pieces");
    assert!(split.to_string().contains("not a single connected curve"));
}

#[test]
fn marching_sample_errors_are_keyed_and_guarded() {
    let (document, _circle_id) = rectangle_with_marked_hole();
    let domains = meshable_domains_from_document(&document).expect("domains");
    let fluid_domain = domains.get("fluid").expect("fluid");

    let unknown = boundary_marching_sample(fluid_domain, "nope", 8).expect_err("unknown");
    let message = unknown.to_string();
    assert!(message.contains("outer"), "lists names: {message}");
    assert!(message.contains("pin"), "lists names: {message}");

    let too_few = boundary_marching_sample(fluid_domain, "outer", 1).expect_err("npoints");
    assert!(too_few.to_string().contains("npoints"));

    let volume = von_karman_fluid();
    assert!(boundary_names(&volume).is_err(), "3D domains have no loops");
}

fn von_karman_fluid() -> MeshableDomain {
    let document = SceneDocument::default_scene().expect("default scene");
    meshable_domains_from_document(&document)
        .expect("domains")
        .get("von_karman_fluid")
        .expect("fluid")
        .clone()
}

#[test]
fn sizing_band_and_background() {
    let domain = von_karman_fluid();
    let mut spec = SizingSpec::for_domain(&domain);
    let background = spec.background;
    spec.bands.push(SizingBand {
        region: "inlet".to_string(),
        distance: 0.2,
        size: 0.01,
    });
    let field = SizingField::new(domain, spec).expect("field");

    // Inside the band (0.05 from the box wall): the band size, exactly.
    assert!((field.size_at(vec3(0.05, 0.0, 0.5)) - 0.01).abs() < 1e-12);
    // Far from every wall of the box owner: graded band vs background.
    let far = field.size_at(vec3(2.25, 0.0, 0.5)); // 0.5 from the z walls
    let expected = (0.01_f64 + 0.3 * (0.5 - 0.2)).min(background);
    assert!((far - expected).abs() < 1e-12);
}

#[test]
fn sizing_is_gradation_lipschitz() {
    let domain = von_karman_fluid();
    let mut spec = SizingSpec::for_domain(&domain);
    spec.bands.push(SizingBand {
        region: "inlet".to_string(),
        distance: 0.1,
        size: 0.02,
    });
    let gradation = spec.gradation;
    let field = SizingField::new(domain, spec).expect("field");

    // Deterministic interior sample pairs across the domain.
    let mut points = Vec::new();
    for i in 0..6 {
        for j in 0..4 {
            points.push(vec3(
                0.3 + 3.9 * (i as f64) / 5.0,
                -1.2 + 2.4 * (j as f64) / 3.0,
                0.5,
            ));
        }
    }
    for a in &points {
        for b in &points {
            let bound = gradation * (*a - *b).length() + 1e-12;
            let difference = (field.size_at(*a) - field.size_at(*b)).abs();
            assert!(
                difference <= bound,
                "gradation bound violated between {a:?} and {b:?}"
            );
        }
    }
}

#[test]
fn sizing_curvature_clamps_at_the_cylinder_wall() {
    let domain = von_karman_fluid();
    let mut spec = SizingSpec::for_domain(&domain);
    spec.curvature_factor = Some(0.5);
    let gradation = spec.gradation;
    let field = SizingField::new(domain, spec).expect("field");

    // Interior point 0.05 from the cylinder wall (radius 0.15): the fluid
    // boundary's mean curvature there is 1/(2r), so the clamp is
    // factor/(1/(2r)) + gradation * depth.
    let near_wall = vec3(1.6, 0.0, 0.5);
    let expected = 0.5 * (2.0 * 0.15) + gradation * 0.05;
    let size = field.size_at(near_wall);
    assert!(
        (size - expected).abs() < 1e-4,
        "curvature clamp: got {size}, expected {expected}"
    );
}

#[test]
fn unknown_band_region_is_reported() {
    let domain = von_karman_fluid();
    let mut spec = SizingSpec::for_domain(&domain);
    spec.bands.push(SizingBand {
        region: "no_such_region".to_string(),
        distance: 0.1,
        size: 0.01,
    });
    let error = SizingField::new(domain, spec).expect_err("unknown region");
    let message = error.to_string();
    assert!(message.contains("no_such_region"));
    assert!(message.contains("inlet"), "error lists available regions");
}
