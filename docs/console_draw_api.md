# Console Draw API

Console Draw runs [Rhai](https://rhai.rs) scripts against the scene that is
currently open. Choose **Console Draw** in the toolbar, edit the script, and
use **Run** or **Ctrl+Enter**.

A run is transactional. casoCAD executes the entire script against a cloned
document and commits a successful edit as one undo step. A parse error,
runtime error, geometry error, or resource-limit error discards every change,
including selection changes. A selection-only run creates no undo entry.
Scripts are external `.rhai` files and are never embedded in `scene.json`.
Running a script again edits the then-current scene, so creation is additive.

## Example

```rhai
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
```

Coordinates and lengths are metres. `mm(x)`, `cm(x)`, and `km(x)` convert a
number to metres. Coordinates may be integers or floating-point values.
Vectors are three-number arrays such as `[x, y, z]`.

## `cad` methods

All object arguments and return values are integer scene object IDs.

```text
cad.add(kind, scale) -> id
cad.draw(kind, start, end, scale) -> id
cad.draw(kind, points) -> id
cad.draw("regular_polygon", [center, vertex], sides) -> id
cad.boolean(first, second, operation) -> id
cad.move(id, delta) -> id
cad.rotate(id, axis, degrees) -> id
cad.rotate_about(id, axis, degrees, pivot) -> id
cad.extrude(section, signed_height) -> id
cad.revolve(section, local_axis, angle_degrees) -> id
cad.rename(id, name)
cad.delete(id)
cad.find(name) -> id
cad.selection() -> [id, ...]
cad.select(id)
cad.select_many([id, ...])
```

`cad.add` accepts every kind in the Add menu, including a default polygon.
`cad.draw` is overloaded: pass `start, end, scale` for drag-defined shapes or
pass `points` for point-defined shapes such as `polygon`, `segment`,
`polyline`, `quadratic_bezier_curve`,
`quadratic_bezier_polycurve`, `polyline_tube`, `quadratic_bezier_tube`,
and `quadratic_bezier_surface`. For example:

```rhai
let naca = cad.draw("polygon", naca_points);
```

The plane is inferred from the three-dimensional points: constant Z means XY,
constant Y means XZ, and constant X means YZ. Non-axis-aligned input is
rejected.

Rotation axes are `"x"`, `"y"`, or `"z"`; revolve axes are the section-local
`"u"` or `"v"`. Boolean operations are `"union"`, `"intersection"`, or
`"difference"` (first minus second).

`cad.move` can create and return a new transform-wrapper ID. Always keep its
return value, as in `let moved = cad.move(id, delta)`. `cad.find` is exact and
case-sensitive; it fails when no object or more than one object has the name.
`cad.rename` keeps the requested name when it is available and otherwise adds
the first available suffix (`Console Part_2`, `Console Part_3`, and so on).

An explicit `cad.select` or `cad.select_many` determines the selection after a
successful run. Otherwise the last add, draw, point placement, boolean,
extrude, or revolve result is selected. If none was created, the previous
selection is preserved. `print(value)` writes to the Console output pane.

## Validation and limits

Malformed vectors, non-finite numbers, invalid IDs, unsupported kinds, axes,
or operations, and invalid geometry stop and roll back the run.
Execution is limited to:

- 1,000,000 Rhai operations and 32 call levels;
- expression depth 64;
- 64 KiB strings and captured output;
- 10,000 array elements and 1,000 map entries;
- 1,000 mutating CAD calls and 1,000 newly allocated scene objects.

The output pane shows at most 500 captured lines, followed by a truncation
marker. Script files must be UTF-8 and no larger than 1 MiB. Rhai scripts have
no shell, filesystem, network, or Python access.
