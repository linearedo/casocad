# Mesh quality measures

The Meshing panel's Inspector evaluates the quality of the cells in a MeshIR
mesh. It provides five measures:

- Scaled Jacobian
- Skewness
- Aspect Ratio
- Compactness
- Orthogonality

Every displayed score follows the same convention:

| score | meaning |
| --- | --- |
| `1.0` | ideal element for the selected measure |
| near `0.0` | poor, nearly degenerate, or collapsed element |
| `0.0` | worst representable quality; invalid or degenerate in measures that detect it |
| `N/A` | unsupported element type, missing topology, or malformed references |

Scores are dimensionless, invariant under translation, rotation, and uniform
scaling, and clamped to the range `[0, 1]`. A value above `1` produced by a
raw formula is displayed as `1`; a negative value is displayed as `0`.

The measures are complementary. A cell can score well under one measure and
poorly under another. For example, a rhombus can have four equal edges and
therefore a perfect Aspect Ratio score while still having poor angles and a
low Skewness score.

---

## Scaled Jacobian

Scaled Jacobian measures the quality and orientation of the geometry at each
corner of an element. The cell receives the score of its worst corner.

For a three-dimensional corner with edge vectors `e1`, `e2`, and `e3`, the
basic quantity is the normalized determinant

```text
                 det(e1, e2, e3)
J = -----------------------------------------
              |e1| |e2| |e3|
```

This quantity describes both the independence and orientation of the three
edge directions. Family-specific normalization factors make the repository's
ideal reference elements score `1`.

For two-dimensional cells, the equivalent calculation is the signed sine of
the angle between the two edges at each corner. It is normalized so that an
equilateral triangle and a square both score `1`.

Interpretation:

- `1`: ideal corner geometry;
- near `0`: edges are nearly collinear or coplanar and the corner is close to
  collapse;
- negative raw determinant: inverted orientation; the displayed score is
  clamped to `0`.

Scaled Jacobian is the most useful measure in the Inspector for finding
inverted and locally collapsed cells. Because it reports the minimum corner
value, one bad corner is enough to give the entire cell a poor score.

It supports triangle, quadrilateral, tetrahedron, hexahedron, prism, and
pyramid families. Generic polygons and polyhedra return `N/A` for this
measure.

Implementation: [`scaled_jacobian`](../meshing/src/quality.rs#L135).

---

## Skewness

Skewness measures how far the angles of an element depart from those of a
regular element.

For a polygon with `n` sides, the ideal internal angle is

```text
ideal angle = pi (n - 2) / n
```

The implementation finds the minimum and maximum corner angles, then computes
the largest normalized departure from the ideal angle:

```text
              / max angle - ideal angle    ideal angle - min angle \
skew = max   |  ------------------------,  -----------------------  |
              \       pi - ideal angle             ideal angle     /

quality = 1 - skew
```

Consequently:

- a regular polygon scores `1`;
- increasingly acute or obtuse corners reduce the score;
- a degenerate face scores `0`.

The label requires some care. Many mesh tools report conventional skewness,
where `0` is ideal and larger values are worse. CasoCAD displays the inverted
quality form, `1 - skewness`, so **larger is better**.

For a three-dimensional cell, the calculation is performed on every face and
the cell receives the score of its worst face. It therefore measures the
quality of the cell's face angles; it does not directly measure volume or
three-dimensional solid angles.

Implementation: [`skewness`](../meshing/src/quality.rs#L226).

---

## Aspect Ratio

Aspect Ratio compares the shortest and longest topological edges of a cell:

```text
          shortest edge length
quality = --------------------
           longest edge length
```

Although the Inspector calls it Aspect Ratio, the displayed value is the
reciprocal of the common `longest / shortest` definition. This keeps the
Inspector's common convention that larger scores are better.

Examples:

- all edge lengths equal: `1`;
- longest edge ten times the shortest: `0.1`;
- a zero-length edge: `0`.

Aspect Ratio is useful for detecting stretched elements, but it only examines
edge lengths. It does not examine angles, orientation, area, or volume. A
skewed element with equal-length edges can still score `1`, and a nearly flat
three-dimensional element can retain a moderately good Aspect Ratio score.

Generic polyhedra return `N/A` because the implementation has no fixed edge
map for them.

Implementation: [`aspect_ratio`](../meshing/src/quality.rs#L270).

---

## Compactness

Compactness measures how much area or volume an element encloses relative to
the size of its boundary. It answers the question: *does this boundary enclose
a healthy element, or only a thin sliver?*

Compactness is not packing density, material volume fraction, or the fraction
of a domain filled by cells. It is an isoperimetric-style shape measure for
one cell.

### Two-dimensional compactness

For a polygon with `n` sides, area `A`, and perimeter `P`, the score is

```text
quality = 4 n tan(pi / n) A / P^2
```

The factor `4 n tan(pi / n)` normalizes the result for the number of sides.
A regular polygon therefore scores `1`:

- an equilateral triangle scores `1`;
- a square scores `1`;
- any other regular `n`-gon scores `1`.

For a fixed perimeter, a compact shape encloses a large area. A long, narrow,
or collapsed shape encloses little area and its score approaches `0`.

### Three-dimensional compactness

For a cell with volume `V` and surface area `S`, the basic dimensionless ratio
is

```text
V / S^(3/2)
```

The implementation divides this value by the same value for an ideal
reference element of the corresponding family:

```text
              V / S^(3/2)
quality = -------------------------
          V_ref / S_ref^(3/2)
```

The reference shapes are:

- a regular tetrahedron;
- a cube;
- a unit-edge right equilateral triangular prism;
- a unit-edge regular square pyramid.

An ideal reference element scores `1`. Flattening or stretching a cell while
retaining a large surface relative to its volume reduces its score toward
`0`. A raw score greater than the chosen reference is clamped to `1`.

The volume calculation uses absolute tetrahedral contributions from the cell
center to its faces. Compactness can therefore give a plausible positive
value to an inverted cell. It should be used together with Scaled Jacobian,
not as an inversion check.

Three-dimensional Compactness also requires usable face topology. Generic
polyhedra are explicitly unsupported and return `N/A`.

Implementation: [`compactness`](../meshing/src/quality.rs#L282).

---

## Orthogonality

Orthogonality measures whether the line connecting neighboring cell centers
is perpendicular to their shared face. This is particularly relevant to
finite-volume discretizations.

For every face of a 3D cell, or every edge of a 2D cell, the implementation
compares:

- the unit face-normal direction `n`;
- the unit direction `d` from the current cell center to the neighboring cell
  center.

The local score is

```text
quality = |n dot d|
```

For a boundary face with no neighboring cell, `d` points from the cell center
to the face center instead. The cell receives the minimum value over all its
faces or edges.

Interpretation:

- `1`: the center-to-center direction is perpendicular to the face;
- values between `0` and `1`: increasingly non-orthogonal connection;
- `0`: the direction lies along the face instead of through it.

The absolute value makes the score independent of face-normal orientation.
Orthogonality depends on complete edge or face topology and correct
owner/neighbor relationships. Missing topology can produce `N/A`.

Implementation: [`orthogonality`](../meshing/src/quality.rs#L332).

---

## Supported element families

The following table summarizes the current implementation.

| element family | Scaled Jacobian | Skewness | Aspect Ratio | Compactness | Orthogonality |
| --- | --- | --- | --- | --- | --- |
| triangles and quads | yes | yes | yes | yes | yes, with edge topology |
| generic polygons | no | yes | yes | yes | yes, with edge topology |
| tetrahedra, hexahedra, prisms, pyramids | yes | yes, with faces | yes | yes, with faces | yes, with faces |
| generic polyhedra | no | yes, with supported faces | no | no | yes, with faces |
| points and one-dimensional cells | no | no | no | no | no |

Both linear and higher-order named families are recognized. Most calculations
use only corner nodes, even for higher-order cells. They do not evaluate the
Jacobian throughout a curved element and generally do not measure distortion
introduced only by mid-edge or mid-face nodes. Orthogonality uses more of the
stored edge/face geometry in some paths, but it is still not a complete
curved-element quality analysis.

---

## Which cells are analyzed?

Only cells in the mesh's highest dimension are analyzed. If a mesh contains
3D volume cells and 2D boundary cells, the Inspector scores only the 3D
cells. If the highest-dimensional cells are 2D, it scores those cells.

This prevents boundary entities from being mixed into the statistics for a
volume mesh.

Implementation: [`analyze`](../meshing/src/quality.rs#L60).

## Inspector display and statistics

The viewport uses 32 color bands:

- red: low quality;
- yellow: approximately middle quality;
- green: high quality;
- gray: `N/A`.

The Inspector reports:

- **Visible** — the number of analyzed cells that remain after viewport,
  Z-range, boundary-distance, and boundary-tag filtering;
- **Min** — the worst visible numeric score;
- **Mean** — the arithmetic mean of visible numeric scores;
- **Max** — the best visible numeric score;
- **Worst ID** — the cell ID with the lowest visible numeric score;
- **N/A** — visible cells for which the selected measure could not be
  calculated.

`N/A` cells are excluded from Min, Mean, Max, and Worst ID. The application
does not define solver-specific thresholds for acceptable or unacceptable
quality; the colors visualize the continuous `[0, 1]` score.

Inspector implementation: [`inspector_ui`](../app/src/meshing_panel.rs#L186)
and [`quality_color`](../app/src/meshing_panel.rs#L1142).

---

## Practical use

A useful inspection order is:

1. **Scaled Jacobian:** find inverted, collapsed, or locally invalid cells.
2. **Skewness:** find poor face angles.
3. **Aspect Ratio:** find excessively stretched cells.
4. **Compactness:** find slivers with little area or volume relative to their
   boundary.
5. **Orthogonality:** find unfavorable connections between neighboring cell
   centers, especially for finite-volume solvers.

No single measure is a complete validity test. In particular, Aspect Ratio
does not see angles, Compactness does not reliably see inversion, and
Skewness in 3D assesses face angles rather than cell volume. Scaled Jacobian
should be the first validity check, followed by the measures relevant to the
target numerical method.
