from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.meshing import load_meshable_domains
from core.scene import SceneDocument
from core.serialization import load_scene, save_scene
from core.sdf import CircleProfile, DistanceOffsetProfile, PlacedSDF2D


def test_save_scene_writes_name_based_json(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"

    save_scene(SceneDocument.default(), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "casocad"
    assert payload["version"] == 1
    assert payload["root_objects"] == ["von_karman_fluid"]
    assert payload["objects"]["von_karman_fluid"] == {
        "type": "difference",
        "left": "flow_volume",
        "right": "cylinder_obstacle",
    }
    assert "object_id" not in payload["objects"]["flow_volume"]
    assert payload["boundary_regions"]["inlet"]["owner"] == "flow_volume"
    assert payload["domains"]["fluid"]["root"] == "von_karman_fluid"
    assert payload["domains"]["fluid"]["tags"] == ["inlet", "outlet"]


def test_load_scene_reads_hand_authored_name_based_json(tmp_path: Path) -> None:
    path = tmp_path / "channel.json"
    path.write_text(
        json.dumps(
            {
                "format": "casocad",
                "version": 1,
                "unit": "m",
                "root_objects": ["fluid"],
                "objects": {
                    "channel": {
                        "type": "box",
                        "center": [0.0, 0.0, 0.0],
                        "size": [3.2, 1.4, 0.9],
                    },
                    "obstacle": {
                        "type": "cylinder",
                        "center": [0.0, 0.0, 0.0],
                        "radius": 0.24,
                        "height": 1.1,
                    },
                    "fluid": {
                        "type": "difference",
                        "left": "channel",
                        "right": "obstacle",
                    },
                },
                "boundary_regions": {
                    "inlet": {"owner": "channel", "patch": "-X"},
                    "outlet": {"owner": "channel", "patch": "+X"},
                },
                "domains": {
                    "fluid": {
                        "type": "fluid",
                        "root": "fluid",
                        "tags": ["inlet", "outlet"],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    document = load_scene(path)

    assert document.fluid_domain is not None
    assert document.fluid_domain.root.name == "fluid"
    assert [tag.name for tag in document.fluid_domain.tag_objects] == [
        "inlet",
        "outlet",
    ]
    values = document.fluid_domain.root.to_numpy(
        np.asarray([0.0, 1.4], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
    )
    assert values[0] > 0.0
    assert values[1] < 0.0


def test_mesh_api_loads_new_scene_json() -> None:
    domains = load_meshable_domains("scene.json")

    assert domains[0].name == "von_karman_fluid"


def test_save_load_preserves_distance_offset_profile(tmp_path: Path) -> None:
    section = PlacedSDF2D(
        name="offset_circle",
        object_id=1,
        profile=DistanceOffsetProfile(CircleProfile(radius=0.5), offset=0.1),
    )
    path = tmp_path / "offset.json"

    save_scene(SceneDocument([section]), path)
    loaded = load_scene(path)

    assert isinstance(loaded.objects[0], PlacedSDF2D)
    assert isinstance(loaded.objects[0].profile, DistanceOffsetProfile)
