use std::cell::RefCell;
use std::rc::Rc;

use caso_kernel::meshing::MeshableDomains;
use caso_kernel::vec3::{vec3, Vec3};
use caso_meshing::{ControlRegion, ControlSet};
use rhai::{Array, Dynamic, Engine, Map, Scope};

#[derive(Clone)]
struct ControlsHandle(Rc<RefCell<ControlSet>>);

#[derive(Clone)]
struct ControlRegionHandle(ControlRegion);

#[derive(Clone, Copy)]
struct ControlRegionApi;

pub fn compile_control_script(
    domains: &MeshableDomains,
    script: &str,
) -> Result<ControlSet, String> {
    let controls = ControlsHandle(Rc::new(RefCell::new(ControlSet::default())));
    let mut engine = Engine::new();
    engine.set_max_operations(2_000_000);
    engine.set_max_expr_depths(64, 64);
    engine
        .register_type_with_name::<ControlsHandle>("ControlSet")
        .register_fn(
            "target_size",
            |handle: &mut ControlsHandle, size: f64| -> Result<(), Box<rhai::EvalAltResult>> {
                handle.0.borrow_mut().target_size(size).map_err(Into::into)
            },
        )
        .register_fn(
            "target_size",
            |handle: &mut ControlsHandle, size: i64| -> Result<(), Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .borrow_mut()
                    .target_size(size as f64)
                    .map_err(Into::into)
            },
        )
        .register_fn(
            "boundary_layer",
            |handle: &mut ControlsHandle,
             domain: &str,
             region: &str,
             spec: Map|
             -> Result<(), Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .borrow_mut()
                    .boundary_layer(
                        domain,
                        region,
                        map_number(&spec, "hwall_n")?,
                        map_number(&spec, "hwall_t")?,
                        map_number(&spec, "ratio")?,
                        map_number(&spec, "thickness")?,
                    )
                    .map_err(Into::into)
            },
        )
        .register_fn(
            "refinement",
            |handle: &mut ControlsHandle,
             domain: &str,
             region: ControlRegionHandle,
             spec: Map|
             -> Result<(), Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .borrow_mut()
                    .refinement(
                        domain,
                        region.0,
                        map_number(&spec, "size")?,
                        map_number_or(&spec, "gradation", 0.2)?,
                    )
                    .map_err(Into::into)
            },
        )
        .register_fn(
            "refinement_box",
            |handle: &mut ControlsHandle,
             domain: &str,
             bounds: Map,
             spec: Map|
             -> Result<(), Box<rhai::EvalAltResult>> {
                handle
                    .0
                    .borrow_mut()
                    .refinement_box(
                        domain,
                        map_vec3(&bounds, "min")?,
                        map_vec3(&bounds, "max")?,
                        map_number(&spec, "size")?,
                        map_number_or(&spec, "gradation", 0.2)?,
                    )
                    .map_err(Into::into)
            },
        );
    engine
        .register_type_with_name::<ControlRegionApi>("ControlRegionApi")
        .register_type_with_name::<ControlRegionHandle>("ControlRegion")
        .register_fn(
            "box",
            |_api: &mut ControlRegionApi,
             bounds: Map|
             -> Result<ControlRegionHandle, Box<rhai::EvalAltResult>> {
                ControlRegion::box_region(map_vec3(&bounds, "min")?, map_vec3(&bounds, "max")?)
                    .map(ControlRegionHandle)
                    .map_err(Into::into)
            },
        )
        .register_fn(
            "sphere",
            |_api: &mut ControlRegionApi,
             center: Array,
             radius: f64|
             -> Result<ControlRegionHandle, Box<rhai::EvalAltResult>> {
                ControlRegion::sphere(dynamic_vec3(&Dynamic::from(center))?, radius)
                    .map(ControlRegionHandle)
                    .map_err(Into::into)
            },
        )
        .register_fn(
            "cylinder",
            |_api: &mut ControlRegionApi,
             a: Array,
             b: Array,
             radius: f64|
             -> Result<ControlRegionHandle, Box<rhai::EvalAltResult>> {
                ControlRegion::cylinder(
                    dynamic_vec3(&Dynamic::from(a))?,
                    dynamic_vec3(&Dynamic::from(b))?,
                    radius,
                )
                .map(ControlRegionHandle)
                .map_err(Into::into)
            },
        )
        .register_fn(
            "polyline_tube",
            |_api: &mut ControlRegionApi,
             points: Array,
             radius: f64|
             -> Result<ControlRegionHandle, Box<rhai::EvalAltResult>> {
                ControlRegion::polyline_tube(
                    points
                        .iter()
                        .map(dynamic_vec3)
                        .collect::<Result<Vec<_>, _>>()?,
                    radius,
                )
                .map(ControlRegionHandle)
                .map_err(Into::into)
            },
        )
        .register_fn(
            "union",
            |left: &mut ControlRegionHandle, right: ControlRegionHandle| {
                ControlRegionHandle(left.0.clone().union(right.0))
            },
        )
        .register_fn(
            "intersection",
            |left: &mut ControlRegionHandle, right: ControlRegionHandle| {
                ControlRegionHandle(left.0.clone().intersection(right.0))
            },
        )
        .register_fn(
            "difference",
            |left: &mut ControlRegionHandle, right: ControlRegionHandle| {
                ControlRegionHandle(left.0.clone().difference(right.0))
            },
        );
    let mut scope = Scope::new();
    scope.push("controls", controls.clone());
    scope.push("control_region", ControlRegionApi);
    engine
        .run_with_scope(&mut scope, script)
        .map_err(|error| error.to_string())?;
    drop(scope);
    drop(engine);
    let controls = Rc::try_unwrap(controls.0)
        .map_err(|_| "control set is still referenced".to_string())?
        .into_inner();
    controls.validate(domains)?;
    Ok(controls)
}

fn dynamic_to_f64(value: &Dynamic) -> Result<f64, Box<rhai::EvalAltResult>> {
    value
        .as_float()
        .or_else(|_| value.as_int().map(|value| value as f64))
        .map_err(|_| "expected a number".to_string().into())
}

fn map_number(map: &Map, key: &str) -> Result<f64, Box<rhai::EvalAltResult>> {
    map.get(key)
        .ok_or_else(|| format!("missing control parameter {key:?}").into())
        .and_then(dynamic_to_f64)
}

fn map_number_or(map: &Map, key: &str, default: f64) -> Result<f64, Box<rhai::EvalAltResult>> {
    map.get(key).map(dynamic_to_f64).unwrap_or(Ok(default))
}

fn dynamic_vec3(value: &Dynamic) -> Result<Vec3, Box<rhai::EvalAltResult>> {
    let values = value
        .clone()
        .try_cast::<Array>()
        .ok_or_else(|| "expected a three-number point".to_string())?;
    if values.len() != 3 {
        return Err("expected a three-number point".to_string().into());
    }
    Ok(vec3(
        dynamic_to_f64(&values[0])?,
        dynamic_to_f64(&values[1])?,
        dynamic_to_f64(&values[2])?,
    ))
}

fn map_vec3(map: &Map, key: &str) -> Result<Vec3, Box<rhai::EvalAltResult>> {
    dynamic_vec3(
        map.get(key)
            .ok_or_else(|| format!("missing control parameter {key:?}"))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use caso_kernel::meshing::meshable_domains_from_document;
    use caso_kernel::roles::DomainKind;
    use caso_kernel::scene::SceneDocument;

    #[test]
    fn refinement_script_produces_typed_controls() {
        let mut document = SceneDocument::new();
        let rectangle = document
            .add_primitive_from_drag("rectangle", vec3(0.0, 0.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
            .unwrap();
        document.rename(rectangle, "sea").unwrap();
        document
            .set_domain_root(rectangle, DomainKind::Fluid)
            .unwrap();
        let domains = meshable_domains_from_document(&document).unwrap();
        let controls = compile_control_script(
            &domains,
            r#"controls.target_size(0.2);
controls.refinement_box("sea", #{min:[0,0,-1],max:[1,1,1]}, #{size:0.1});
let region = control_region.sphere([1, 0.5, 0], 0.25);
controls.refinement("sea", region, #{size:0.08, gradation:0.3});"#,
        )
        .unwrap();
        assert_eq!(controls.target_size, Some(0.2));
        assert_eq!(controls.refinements.len(), 2);
    }

    #[test]
    fn boundary_layer_script_uses_gmsh_style_sizes_and_maximum_thickness() {
        let mut document = SceneDocument::new();
        let rectangle = document
            .add_primitive_from_drag("rectangle", vec3(0.0, 0.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
            .unwrap();
        document.rename(rectangle, "sea").unwrap();
        document
            .set_domain_root(rectangle, DomainKind::Fluid)
            .unwrap();
        document
            .add_boundary_region(rectangle, None, None, Some("wall"))
            .unwrap();
        let domains = meshable_domains_from_document(&document).unwrap();
        let region = domains
            .iter()
            .next()
            .unwrap()
            .boundary_regions
            .first()
            .unwrap()
            .name
            .clone();
        let controls = compile_control_script(
            &domains,
            &format!(
                r#"controls.target_size(0.2);
controls.boundary_layer("sea", "{region}", #{{hwall_n:0.01, hwall_t:0.05, ratio:1.2, thickness:0.05}});"#
            ),
        )
        .unwrap();
        let layer = &controls.boundary_layers[0];
        assert_eq!(layer.layers, 3);
        assert_eq!(layer.hwall_t, 0.05);
        assert!(layer.total_height() <= layer.thickness);
    }

    #[test]
    fn target_size_is_required_once_and_must_be_positive_and_finite() {
        let document = SceneDocument::new();
        let domains = meshable_domains_from_document(&document).unwrap();
        for script in [
            "",
            "controls.target_size(0.0);",
            "controls.target_size(-1.0);",
            "controls.target_size(1.0 / 0.0);",
            "controls.target_size(0.1); controls.target_size(0.2);",
        ] {
            assert!(
                compile_control_script(&domains, script).is_err(),
                "{script}"
            );
        }

        let mut controls = ControlSet {
            target_size: Some(f64::NAN),
            ..ControlSet::default()
        };
        assert!(controls.require_target_size().is_err());
        controls.target_size = Some(f64::INFINITY);
        assert!(controls.require_target_size().is_err());
    }
}
