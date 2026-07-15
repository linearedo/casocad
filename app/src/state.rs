//! Application state: the scene document plus selection, undo/redo history,
//! working unit, and the status line. All edits flow through here so every
//! mutating action gets a history snapshot (Python keeps 50, we match it).

use caso_kernel::scene::{ObjectId, SceneDocument};

pub const UNDO_LIMIT: usize = 50;

/// A user-selectable working unit (display + entry); the model is always
/// meters, matching `app/dimensions.py`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthUnit {
    pub key: &'static str,
    pub label: &'static str,
    /// Meters per unit.
    pub factor: f64,
}

pub const LENGTH_UNITS: [LengthUnit; 4] = [
    LengthUnit { key: "km", label: "Kilometers", factor: 1000.0 },
    LengthUnit { key: "m", label: "Meters", factor: 1.0 },
    LengthUnit { key: "cm", label: "Centimeters", factor: 0.01 },
    LengthUnit { key: "mm", label: "Millimeters", factor: 0.001 },
];

pub const DEFAULT_LENGTH_UNIT: LengthUnit = LENGTH_UNITS[1];

pub struct AppState {
    pub document: SceneDocument,
    pub selection: Vec<ObjectId>,
    /// Selected BoundaryRegion (regions are not scene objects).
    pub selected_region: Option<u32>,
    pub unit: LengthUnit,
    pub status: String,
    undo_stack: Vec<SceneDocument>,
    redo_stack: Vec<SceneDocument>,
}

impl AppState {
    pub fn new(document: SceneDocument) -> Self {
        Self {
            document,
            selection: Vec::new(),
            selected_region: None,
            unit: DEFAULT_LENGTH_UNIT,
            status: String::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Call before a mutating edit: snapshots the current document.
    pub fn push_undo(&mut self) {
        self.undo_stack.push(self.document.snapshot());
        if self.undo_stack.len() > UNDO_LIMIT {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self) {
        if let Some(previous) = self.undo_stack.pop() {
            self.redo_stack.push(self.document.snapshot());
            self.document = previous;
            self.document.mark_changed();
            self.retain_live_selection();
            self.status = "Undo".to_string();
        }
    }

    /// Revert to the last undo snapshot WITHOUT creating a redo entry — for
    /// aborting an in-flight gesture (or rolling back a refused commit)
    /// whose `push_undo` already ran. That `push_undo` cleared the redo
    /// stack, so popping without pushing restores the exact pre-gesture
    /// history state.
    pub fn abort_to_last_snapshot(&mut self) {
        if let Some(previous) = self.undo_stack.pop() {
            self.document = previous;
            self.document.mark_changed();
            self.retain_live_selection();
        }
    }

    pub fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(self.document.snapshot());
            self.document = next;
            self.document.mark_changed();
            self.retain_live_selection();
            self.status = "Redo".to_string();
        }
    }

    pub fn selected_single(&self) -> Option<ObjectId> {
        match self.selection.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

    pub fn select_only(&mut self, id: ObjectId) {
        self.selection = vec![id];
    }

    pub fn toggle_select(&mut self, id: ObjectId) {
        if let Some(position) = self.selection.iter().position(|other| *other == id) {
            self.selection.remove(position);
        } else {
            self.selection.push(id);
        }
    }

    pub fn retain_live_selection(&mut self) {
        let live = self.document.live_ids();
        self.selection.retain(|id| live.contains(id));
        if let Some(region_id) = self.selected_region {
            if !self
                .document
                .boundary_regions
                .iter()
                .any(|region| region.object_id == region_id)
            {
                self.selected_region = None;
            }
        }
    }

    /// Report a `GeometryResult` outcome on the status line.
    pub fn report<T>(
        &mut self,
        result: caso_kernel::GeometryResult<T>,
        success: &str,
    ) -> Option<T> {
        match result {
            Ok(value) => {
                self.status = success.to_string();
                Some(value)
            }
            Err(error) => {
                self.status = error.to_string();
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The toolbar workflow: add two primitives, subtract, then walk the
    /// history back and forward again.
    #[test]
    fn undo_redo_roundtrip_over_toolbar_ops() {
        let mut state = AppState::new(SceneDocument::new());
        state.push_undo();
        let a = state.document.add_primitive("box", 1.0).unwrap();
        state.push_undo();
        let b = state.document.add_primitive("cylinder", 1.0).unwrap();
        state.push_undo();
        let op = state.document.combine(a, b, "difference").unwrap();
        state.select_only(op);
        assert_eq!(state.document.roots, vec![op]);

        state.undo();
        assert_eq!(state.document.roots, vec![a, b]);
        assert!(state.selection.is_empty(), "dead selection dropped");
        state.undo();
        assert_eq!(state.document.roots, vec![a]);
        state.redo();
        state.redo();
        assert_eq!(state.document.roots.len(), 1);
        assert!(!state.can_redo());
    }

    #[test]
    fn undo_history_is_bounded() {
        let mut state = AppState::new(SceneDocument::new());
        for _ in 0..(UNDO_LIMIT + 10) {
            state.push_undo();
            state.document.mark_changed();
        }
        let mut undone = 0;
        while state.can_undo() {
            state.undo();
            undone += 1;
        }
        assert_eq!(undone, UNDO_LIMIT);
    }

    /// Aborting a gesture reverts the document without leaving the
    /// half-applied edit reachable through Ctrl+Y.
    #[test]
    fn abort_to_last_snapshot_leaves_no_redo() {
        let mut state = AppState::new(SceneDocument::new());
        state.push_undo();
        let a = state.document.add_primitive("box", 1.0).unwrap();
        // The gesture: snapshot, then a mutation the user aborts.
        state.push_undo();
        state.document.add_primitive("cylinder", 1.0).unwrap();
        state.abort_to_last_snapshot();
        assert_eq!(state.document.roots, vec![a]);
        assert!(!state.can_redo(), "aborted gesture must not be redoable");
        // The pre-gesture history is intact: one more undo removes the box.
        state.undo();
        assert!(state.document.roots.is_empty());
    }

    /// Deleting selected nodes (the Delete-key path) leaves no dangling ids.
    #[test]
    fn delete_many_clears_selection() {
        let mut state = AppState::new(SceneDocument::default_scene().unwrap());
        let roots = state.document.roots.clone();
        state.selection = roots.clone();
        state.push_undo();
        state.document.delete_many(&roots);
        state.retain_live_selection();
        assert!(state.selection.is_empty());
        state.undo();
        assert_eq!(state.document.roots, roots);
    }
}
