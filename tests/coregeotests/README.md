Run the core geometry timing suite with live `INFO` logs:

```bash
.venv/bin/pytest -q tests/coregeotests -o log_cli=true --log-cli-level=INFO
```

The suite benchmarks:

- default startup scene render-prep
- all 3D primitive creation, move, rotate, boolean, translate, scale
- all 2D and 1D primitive/profile creation, move, rotate, boolean
- drag-based primitive creation for supported tools
- custom polyline, quadratic Bezier, polygon, and world-point shape workflows
- copy, paste, delete, and transform wrapper workflows
- fluid-domain root, tag, and boundary-region workflows
- extrude and revolve from 2D profiles
- path-based solids via `polyline_tube` and `quadratic_bezier_tube`

Each step logs:

- mutation time on `SceneDocument`
- `visual_snapshot()` time
- render artifact build time
- viewport surface generation time
- generated viewport vertex and triangle counts
