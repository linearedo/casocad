//! Transactional Rhai runner for the Console Draw workspace.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use caso_kernel::scene::{ObjectId, SceneDocument};
use caso_kernel::sdf::node::RotationAxis;
use caso_kernel::sdf::solid_from_2d::RevolveAxis;
use caso_kernel::vec3::{vec3, Vec3};
use rhai::{Array, Dynamic, Engine, EvalAltResult, Scope};

use crate::state::AppState;

const MAX_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_LINES: usize = 500;
const MAX_MUTATIONS: usize = 1_000;
const MAX_ALLOCATIONS: usize = 1_000;

pub const EXAMPLE_SCRIPT: &str = r#"// Console Draw edits the current scene atomically.
let block = cad.draw(
    "box",
    [0.0, 0.0, 0.0],
    [mm(80), mm(50), mm(10)],
    mm(1)
);
let bore = cad.add("cylinder", mm(20));
let bore = cad.move(bore, [mm(40), mm(25), mm(5)]);
let part = cad.boolean(block, bore, "difference");
cad.rename(part, "Console Part");
print(`Created Console Part (object ${part})`);
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsoleDrawResult {
    pub output: String,
    pub mutated: bool,
    pub selection: Vec<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsoleDrawError {
    pub output: String,
    pub message: String,
}

impl ConsoleDrawError {
    pub fn display_output(&self) -> String {
        if self.output.is_empty() {
            self.message.clone()
        } else {
            format!("{}\n{}", self.output, self.message)
        }
    }
}

impl fmt::Display for ConsoleDrawError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.display_output())
    }
}

impl std::error::Error for ConsoleDrawError {}

#[derive(Clone)]
struct CadHandle(Rc<RefCell<RunContext>>);

struct RunContext {
    document: SceneDocument,
    initial_selection: Vec<ObjectId>,
    explicit_selection: Option<Vec<ObjectId>>,
    last_result: Option<ObjectId>,
    mutations: usize,
    allocations: usize,
}

impl RunContext {
    fn mutate<T>(
        &mut self,
        operation: impl FnOnce(&mut SceneDocument) -> caso_kernel::GeometryResult<T>,
    ) -> Result<T, Box<EvalAltResult>> {
        self.mutations += 1;
        if self.mutations > MAX_MUTATIONS {
            return Err(error("more than 1,000 mutating CAD calls"));
        }
        let before = self.document.objects.len();
        let result = operation(&mut self.document).map_err(|failure| error(failure.to_string()))?;
        self.allocations += self.document.objects.len().saturating_sub(before);
        if self.allocations > MAX_ALLOCATIONS {
            return Err(error("more than 1,000 new scene objects"));
        }
        Ok(result)
    }

    fn valid_id(&self, value: i64) -> Result<ObjectId, Box<EvalAltResult>> {
        let id =
            ObjectId::try_from(value).map_err(|_| error("object IDs must be positive integers"))?;
        if id == 0 {
            return Err(error("object IDs must be positive integers"));
        }
        self.document
            .object(id)
            .map_err(|failure| error(failure.to_string()))?;
        Ok(id)
    }

    fn finish_selection(&self) -> Vec<ObjectId> {
        let mut selection = self
            .explicit_selection
            .clone()
            .or_else(|| self.last_result.map(|id| vec![id]))
            .unwrap_or_else(|| self.initial_selection.clone());
        selection.retain(|id| self.document.objects.contains_key(id));
        selection
    }

    fn unique_name(&self, id: ObjectId, requested: &str) -> String {
        let is_used = |candidate: &str| {
            self.document
                .objects
                .values()
                .any(|object| object.id != id && object.name == candidate)
        };
        if !is_used(requested) {
            return requested.to_string();
        }
        let mut index = 2;
        loop {
            let candidate = format!("{requested}_{index}");
            if !is_used(&candidate) {
                return candidate;
            }
            index += 1;
        }
    }
}

#[derive(Default)]
struct OutputCapture {
    text: String,
    exceeded: bool,
}

impl OutputCapture {
    fn push_line(&mut self, line: &str) {
        let separator = usize::from(!self.text.is_empty());
        let available = MAX_OUTPUT_BYTES.saturating_sub(self.text.len());
        if separator + line.len() > available {
            self.exceeded = true;
            if available > separator {
                if separator == 1 {
                    self.text.push('\n');
                }
                let mut end = available - separator;
                while !line.is_char_boundary(end) {
                    end -= 1;
                }
                self.text.push_str(&line[..end]);
            }
            return;
        }
        if separator == 1 {
            self.text.push('\n');
        }
        self.text.push_str(line);
    }
}

fn error(message: impl Into<String>) -> Box<EvalAltResult> {
    message.into().into()
}

fn number(value: &Dynamic) -> Result<f64, Box<EvalAltResult>> {
    let number = value
        .as_float()
        .or_else(|_| value.as_int().map(|integer| integer as f64))
        .map_err(|_| error("expected a number"))?;
    if !number.is_finite() {
        return Err(error("numbers must be finite"));
    }
    Ok(number)
}

fn vector(values: Array) -> Result<Vec3, Box<EvalAltResult>> {
    if values.len() != 3 {
        return Err(error("vectors must contain exactly three numbers"));
    }
    Ok(vec3(
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ))
}

fn points(values: Array) -> Result<Vec<Vec3>, Box<EvalAltResult>> {
    values
        .into_iter()
        .map(|value| {
            value
                .try_cast::<Array>()
                .ok_or_else(|| error("points must be an array of three-number vectors"))
                .and_then(vector)
        })
        .collect()
}

fn ids(handle: &CadHandle, values: Array) -> Result<Vec<ObjectId>, Box<EvalAltResult>> {
    values
        .into_iter()
        .map(|value| {
            let integer = value
                .as_int()
                .map_err(|_| error("object IDs must be integers"))?;
            handle.0.borrow().valid_id(integer)
        })
        .collect()
}

fn id_result(id: ObjectId) -> Result<i64, Box<EvalAltResult>> {
    Ok(i64::from(id))
}

fn capped_lines(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.len() <= MAX_OUTPUT_LINES {
        text.to_string()
    } else {
        format!(
            "{}\n… output truncated after {MAX_OUTPUT_LINES} lines …",
            lines[..MAX_OUTPUT_LINES].join("\n")
        )
    }
}

fn draw_points(
    handle: &mut CadHandle,
    kind: &str,
    raw_points: Array,
) -> Result<i64, Box<EvalAltResult>> {
    let points = points(raw_points)?;
    let plane = infer_plane(&points)?;
    let id = handle
        .0
        .borrow_mut()
        .mutate(|document| document.add_point_shape_from_world_points(kind, &points, plane))?;
    handle.0.borrow_mut().last_result = Some(id);
    id_result(id)
}

fn infer_plane(points: &[Vec3]) -> Result<&'static str, Box<EvalAltResult>> {
    let range = |coordinate: fn(&Vec3) -> f64| {
        points.iter().map(coordinate).fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        )
    };
    let (x_min, x_max) = range(|point| point.x);
    let (y_min, y_max) = range(|point| point.y);
    let (z_min, z_max) = range(|point| point.z);
    let scale = [x_min, x_max, y_min, y_max, z_min, z_max]
        .into_iter()
        .filter(|value| value.is_finite())
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    let tolerance = scale * 1.0e-9;
    if z_max - z_min <= tolerance {
        Ok("xy")
    } else if y_max - y_min <= tolerance {
        Ok("xz")
    } else if x_max - x_min <= tolerance {
        Ok("yz")
    } else {
        Err(error(
            "points must lie on one axis-aligned plane (xy, xz, or yz)",
        ))
    }
}

fn draw_regular_polygon(
    handle: &mut CadHandle,
    kind: &str,
    raw_points: Array,
    sides: i64,
) -> Result<i64, Box<EvalAltResult>> {
    if kind != "regular_polygon" {
        return Err(error(
            "the sides argument is only valid for regular_polygon",
        ));
    }
    let points = points(raw_points)?;
    let plane = infer_plane(&points)?;
    let sides =
        u32::try_from(sides).map_err(|_| error("polygon side count must be a positive integer"))?;
    let id = handle
        .0
        .borrow_mut()
        .mutate(|document| document.add_regular_polygon_from_world_points(&points, sides, plane))?;
    handle.0.borrow_mut().last_result = Some(id);
    id_result(id)
}

fn register_cad_api(engine: &mut Engine) {
    engine
        .register_type_with_name::<CadHandle>("Cad")
        .register_fn(
            "add",
            |handle: &mut CadHandle,
             kind: &str,
             scale: Dynamic|
             -> Result<i64, Box<EvalAltResult>> {
                let scale = number(&scale)?;
                let id = handle
                    .0
                    .borrow_mut()
                    .mutate(|document| document.add_primitive(kind, scale))?;
                handle.0.borrow_mut().last_result = Some(id);
                id_result(id)
            },
        )
        .register_fn(
            "draw",
            |handle: &mut CadHandle,
             kind: &str,
             start: Array,
             end: Array,
             scale: Dynamic|
             -> Result<i64, Box<EvalAltResult>> {
                let (start, end, scale) = (vector(start)?, vector(end)?, number(&scale)?);
                let id = handle
                    .0
                    .borrow_mut()
                    .mutate(|document| document.add_primitive_from_drag(kind, start, end, scale))?;
                handle.0.borrow_mut().last_result = Some(id);
                id_result(id)
            },
        )
        .register_fn("draw", draw_points)
        .register_fn("draw", draw_regular_polygon)
        .register_fn(
            "boolean",
            |handle: &mut CadHandle,
             first: i64,
             second: i64,
             operation: &str|
             -> Result<i64, Box<EvalAltResult>> {
                if !matches!(operation, "union" | "intersection" | "difference") {
                    return Err(error(
                        "boolean operation must be union, intersection, or difference",
                    ));
                }
                let (first, second) = {
                    let context = handle.0.borrow();
                    (context.valid_id(first)?, context.valid_id(second)?)
                };
                let id = handle
                    .0
                    .borrow_mut()
                    .mutate(|document| document.combine(first, second, operation))?;
                handle.0.borrow_mut().last_result = Some(id);
                id_result(id)
            },
        )
        .register_fn(
            "move",
            |handle: &mut CadHandle,
             raw_id: i64,
             delta: Array|
             -> Result<i64, Box<EvalAltResult>> {
                let id = handle.0.borrow().valid_id(raw_id)?;
                let delta = vector(delta)?;
                let result = handle
                    .0
                    .borrow_mut()
                    .mutate(|document| document.move_object(id, delta))?;
                id_result(result)
            },
        )
        .register_fn(
            "rotate",
            |handle: &mut CadHandle,
             raw_id: i64,
             axis: &str,
             degrees: Dynamic|
             -> Result<i64, Box<EvalAltResult>> {
                rotate(handle, raw_id, axis, degrees, None)
            },
        )
        .register_fn(
            "rotate_about",
            |handle: &mut CadHandle,
             raw_id: i64,
             axis: &str,
             degrees: Dynamic,
             pivot: Array|
             -> Result<i64, Box<EvalAltResult>> {
                rotate(handle, raw_id, axis, degrees, Some(vector(pivot)?))
            },
        )
        .register_fn(
            "extrude",
            |handle: &mut CadHandle,
             raw_id: i64,
             height: Dynamic|
             -> Result<i64, Box<EvalAltResult>> {
                let id = handle.0.borrow().valid_id(raw_id)?;
                let height = number(&height)?;
                let result = handle.0.borrow_mut().mutate(|document| {
                    document.solid_from_2d(
                        id,
                        "extrude",
                        Some(height),
                        RevolveAxis::V,
                        None,
                        None,
                        None,
                        360.0,
                    )
                })?;
                handle.0.borrow_mut().last_result = Some(result);
                id_result(result)
            },
        )
        .register_fn(
            "revolve",
            |handle: &mut CadHandle,
             raw_id: i64,
             axis: &str,
             degrees: Dynamic|
             -> Result<i64, Box<EvalAltResult>> {
                let id = handle.0.borrow().valid_id(raw_id)?;
                let axis =
                    RevolveAxis::parse(axis).map_err(|failure| error(failure.to_string()))?;
                let degrees = number(&degrees)?;
                let result = handle.0.borrow_mut().mutate(|document| {
                    document.solid_from_2d(id, "revolve", None, axis, None, None, None, degrees)
                })?;
                handle.0.borrow_mut().last_result = Some(result);
                id_result(result)
            },
        )
        .register_fn(
            "rename",
            |handle: &mut CadHandle, raw_id: i64, name: &str| -> Result<(), Box<EvalAltResult>> {
                let (id, name) = {
                    let context = handle.0.borrow();
                    let id = context.valid_id(raw_id)?;
                    (id, context.unique_name(id, name))
                };
                handle
                    .0
                    .borrow_mut()
                    .mutate(|document| document.rename(id, name))
            },
        )
        .register_fn(
            "delete",
            |handle: &mut CadHandle, raw_id: i64| -> Result<(), Box<EvalAltResult>> {
                let id = handle.0.borrow().valid_id(raw_id)?;
                handle.0.borrow_mut().mutate(|document| {
                    document.delete(id);
                    Ok(())
                })
            },
        )
        .register_fn(
            "find",
            |handle: &mut CadHandle, name: &str| -> Result<i64, Box<EvalAltResult>> {
                let matches: Vec<ObjectId> = handle
                    .0
                    .borrow()
                    .document
                    .objects
                    .values()
                    .filter(|object| object.name == name)
                    .map(|object| object.id)
                    .collect();
                match matches.as_slice() {
                    [id] => id_result(*id),
                    [] => Err(error(format!("no scene object named {name:?}"))),
                    _ => Err(error(format!("multiple scene objects named {name:?}"))),
                }
            },
        )
        .register_fn("selection", |handle: &mut CadHandle| -> Array {
            let context = handle.0.borrow();
            context
                .explicit_selection
                .as_ref()
                .unwrap_or(&context.initial_selection)
                .iter()
                .map(|id| Dynamic::from(i64::from(*id)))
                .collect()
        })
        .register_fn(
            "select",
            |handle: &mut CadHandle, raw_id: i64| -> Result<(), Box<EvalAltResult>> {
                let id = handle.0.borrow().valid_id(raw_id)?;
                handle.0.borrow_mut().explicit_selection = Some(vec![id]);
                Ok(())
            },
        )
        .register_fn(
            "select_many",
            |handle: &mut CadHandle, values: Array| -> Result<(), Box<EvalAltResult>> {
                let selection = ids(handle, values)?;
                handle.0.borrow_mut().explicit_selection = Some(selection);
                Ok(())
            },
        );

    for (name, factor) in [("mm", 0.001), ("cm", 0.01), ("km", 1_000.0)] {
        engine.register_fn(
            name,
            move |value: Dynamic| -> Result<f64, Box<EvalAltResult>> {
                Ok(number(&value)? * factor)
            },
        );
    }
}

fn rotate(
    handle: &mut CadHandle,
    raw_id: i64,
    axis: &str,
    degrees: Dynamic,
    pivot: Option<Vec3>,
) -> Result<i64, Box<EvalAltResult>> {
    let id = handle.0.borrow().valid_id(raw_id)?;
    let axis = RotationAxis::parse(axis).map_err(|failure| error(failure.to_string()))?;
    let degrees = number(&degrees)?;
    handle
        .0
        .borrow_mut()
        .mutate(|document| document.rotate_object(id, axis, degrees, pivot))?;
    id_result(id)
}

/// Run `script` against a cloned document and commit only a complete success.
pub fn run_console_draw(
    state: &mut AppState,
    script: &str,
) -> Result<ConsoleDrawResult, ConsoleDrawError> {
    let original = state.document.snapshot();
    let context = Rc::new(RefCell::new(RunContext {
        document: original.snapshot(),
        initial_selection: state.selection.clone(),
        explicit_selection: None,
        last_result: None,
        mutations: 0,
        allocations: 0,
    }));
    let output = Rc::new(RefCell::new(OutputCapture::default()));

    let mut engine = Engine::new();
    engine
        .set_max_operations(1_000_000)
        .set_max_call_levels(32)
        .set_max_expr_depths(64, 64)
        .set_max_string_size(MAX_OUTPUT_BYTES)
        .set_max_array_size(10_000)
        .set_max_map_size(1_000);
    let output_writer = output.clone();
    engine.on_print(move |line| output_writer.borrow_mut().push_line(line));
    register_cad_api(&mut engine);

    let mut scope = Scope::new();
    scope.push("cad", CadHandle(context.clone()));
    let run_result = engine.run_with_scope(&mut scope, script);
    let captured = capped_lines(&output.borrow().text);
    if let Err(failure) = run_result {
        return Err(ConsoleDrawError {
            output: captured,
            message: failure.to_string(),
        });
    }
    if output.borrow().exceeded {
        return Err(ConsoleDrawError {
            output: captured,
            message: "output exceeded the 64 KiB limit".to_string(),
        });
    }

    let context = context.borrow();
    let mutated = context.document != original;
    let selection = context.finish_selection();
    if mutated {
        state.push_undo();
        state.install_document(context.document.snapshot());
    }
    state.selection = selection.clone();
    state.retain_live_selection();

    Ok(ConsoleDrawResult {
        output: captured,
        mutated,
        selection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use caso_kernel::scene::ScenePayload;

    fn empty_state() -> AppState {
        AppState::new(SceneDocument::new())
    }

    #[test]
    fn default_example_runs_and_selects_result() {
        let mut state = empty_state();
        let result = run_console_draw(&mut state, EXAMPLE_SCRIPT).unwrap();
        assert!(result.mutated);
        assert_eq!(state.selection.len(), 1);
        assert_eq!(
            state.document.object(state.selection[0]).unwrap().name,
            "Console Part"
        );
    }

    #[test]
    fn repeated_console_names_are_enumerated() {
        let mut state = empty_state();
        let script = r#"
            let object = cad.add("box", 1);
            cad.rename(object, "Console Part");
        "#;
        run_console_draw(&mut state, script).unwrap();
        run_console_draw(&mut state, script).unwrap();
        run_console_draw(&mut state, script).unwrap();
        let names: Vec<&str> = state
            .document
            .roots
            .iter()
            .map(|id| state.document.object(*id).unwrap().name.as_str())
            .collect();
        assert_eq!(names, ["Console Part", "Console Part_2", "Console Part_3"]);
    }

    #[test]
    fn multi_command_run_is_one_history_step() {
        let mut state = empty_state();
        run_console_draw(
            &mut state,
            r#"
                let a = cad.draw("box", [0, 0, 0], [2, 2, 2], 1);
                let b = cad.add("sphere", 1);
                let b = cad.move(b, [1, 1, 1]);
                let result = cad.boolean(a, b, "difference");
                cad.rename(result, "part");
            "#,
        )
        .unwrap();
        assert_eq!(state.document.roots.len(), 1);
        state.undo();
        assert!(state.document.roots.is_empty());
        state.redo();
        assert_eq!(state.document.roots.len(), 1);
        assert_eq!(
            state.document.object(state.document.roots[0]).unwrap().name,
            "part"
        );
    }

    #[test]
    fn add_draw_move_and_all_boolean_operations_build_expected_tree() {
        for operation in ["union", "intersection", "difference"] {
            let mut state = empty_state();
            let script = format!(
                r#"
                    let a = cad.draw("box", [0, 0, 0], [2, 2, 2], 1);
                    let b = cad.add("sphere", 1);
                    let b = cad.move(b, [1, 1, 1]);
                    let result = cad.boolean(a, b, "{operation}");
                    cad.rename(result, "result");
                "#
            );
            run_console_draw(&mut state, &script).unwrap();
            let root = state.document.object(state.document.roots[0]).unwrap();
            assert_eq!(root.name, "result");
            assert!(matches!(root.payload, ScenePayload::Operator { .. }));
            assert_eq!(state.selection, vec![root.id]);
        }
    }

    #[test]
    fn failure_discards_document_selection_and_history() {
        let mut state = empty_state();
        let existing = state.document.add_primitive("box", 1.0).unwrap();
        state.selection = vec![existing];
        state.push_undo();
        state.document.rename(existing, "before").unwrap();
        state.undo();
        assert!(state.can_redo());
        let document = state.document.snapshot();
        let selection = state.selection.clone();
        let error = run_console_draw(
            &mut state,
            r#"print("before failure"); cad.add("sphere", 1); throw "stop";"#,
        )
        .unwrap_err();
        assert!(error.display_output().contains("before failure"));
        assert_eq!(state.document, document);
        assert_eq!(state.selection, selection);
        assert!(!state.can_undo());
        assert!(state.can_redo());
    }

    #[test]
    fn selection_and_find_are_transactional_without_history() {
        let mut state = empty_state();
        let first = state.document.add_primitive("box", 1.0).unwrap();
        let second = state.document.add_primitive("sphere", 1.0).unwrap();
        state.document.rename(second, "target").unwrap();
        state.selection = vec![first];
        run_console_draw(
            &mut state,
            r#"
                let before = cad.selection();
                if before[0] != 1 { throw "initial selection missing"; }
                cad.select(cad.find("target"));
            "#,
        )
        .unwrap();
        assert_eq!(state.selection, vec![second]);
        assert!(!state.can_undo());
        assert!(run_console_draw(&mut state, r#"cad.find("missing");"#).is_err());
        state.document.rename(first, "target").unwrap();
        assert!(run_console_draw(&mut state, r#"cad.find("target");"#).is_err());
    }

    #[test]
    fn selection_only_run_preserves_redo_history() {
        let mut state = empty_state();
        let first = state.document.add_primitive("box", 1.0).unwrap();
        state.push_undo();
        state.document.add_primitive("sphere", 1.0).unwrap();
        state.undo();
        assert!(state.can_redo());
        let result = run_console_draw(&mut state, &format!("cad.select({first});")).unwrap();
        assert!(!result.mutated);
        assert_eq!(state.selection, vec![first]);
        assert!(state.can_redo());
    }

    #[test]
    fn point_builders_and_generators_use_kernel_paths() {
        let mut state = empty_state();
        run_console_draw(
            &mut state,
            r#"
                let polygon = cad.draw("polygon", [[0,0,0], [2,0,0], [1,1,0]]);
                let solid = cad.extrude(polygon, 2);
                let regular = cad.draw("regular_polygon", [[4,0,0], [5,0,0]], 6);
                let revolved = cad.revolve(regular, "v", 180);
                cad.select_many([solid, revolved]);
            "#,
        )
        .unwrap();
        assert!(matches!(
            state.document.object(state.selection[0]).unwrap().payload,
            ScenePayload::Extrude { .. }
        ));
        assert!(matches!(
            state.document.object(state.selection[1]).unwrap().payload,
            ScenePayload::Revolve { .. }
        ));
    }

    #[test]
    fn validation_and_limits_fail_cleanly() {
        for script in [
            r#"cad.move(999, [0, 0, 0]);"#,
            r#"cad.add("box", 0.0 / 0.0);"#,
            r#"cad.move(cad.add("box", 1), [0, 1]);"#,
            r#"cad.rotate(cad.add("box", 1), "q", 20);"#,
            r#"cad.boolean(cad.add("box", 1), cad.add("sphere", 1), "xor");"#,
            r#"cad.boolean(cad.add("box", 1), cad.add("circle", 1), "union");"#,
            r#"cad.extrude(cad.add("box", 1), 2);"#,
            r#"cad.draw("polygon", [[0,0,0], [1,0,1], [0,1,1]]);"#,
            r#"let a = []; while a.len < 10001 { a.push(0); }"#,
            r#"while true {}"#,
        ] {
            let mut state = empty_state();
            let before = state.document.snapshot();
            assert!(run_console_draw(&mut state, script).is_err(), "{script}");
            assert_eq!(state.document, before, "{script}");
        }
    }

    #[test]
    fn output_is_line_capped() {
        let mut state = empty_state();
        let result = run_console_draw(&mut state, "for n in 0..600 { print(n); }").unwrap();
        assert!(result.output.contains("truncated after 500 lines"));
        assert!(result.output.lines().count() <= 501);
    }

    #[test]
    fn mutation_creation_and_output_limits_discard_the_run() {
        let cases = [
            "cad.add(\"box\", 1);\n".repeat(1_001),
            "print(\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\");\n".repeat(700),
            format!("let text = \"{}\";", "x".repeat(MAX_OUTPUT_BYTES + 1)),
        ];
        for script in cases {
            let mut state = empty_state();
            assert!(run_console_draw(&mut state, &script).is_err());
            assert!(state.document.roots.is_empty());
            assert!(!state.can_undo());
        }
    }
}
