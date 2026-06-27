#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("perf_logs")
DEFAULT_LOG_GLOB = "ultimate_frame_test_*.jsonl"


def _stats(values: list[float]) -> str:
    if not values:
        return "count=0"
    values = sorted(values)
    p95 = values[min(len(values) - 1, int(len(values) * 0.95))]
    return (
        f"count={len(values)} median={statistics.median(values):.2f} "
        f"p95={p95:.2f} max={values[-1]:.2f}"
    )


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _latest_log(log_dir: Path) -> Path:
    logs = sorted(
        log_dir.glob(DEFAULT_LOG_GLOB),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        logs = sorted(
            log_dir.glob("*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not logs:
        raise SystemExit(
            f"No JSONL files found in {log_dir}. "
            "Run tools/ultimate_frame_test.py first."
        )
    return logs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl",
        nargs="?",
        help="JSONL log to analyze. Default: newest perf_logs/ultimate_frame_test_*.jsonl",
    )
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    path = Path(args.jsonl) if args.jsonl else _latest_log(Path(args.log_dir))
    events = _load(path)
    artifacts = [event for event in events if event.get("kind") == "artifact"]
    slow = [event for event in events if event.get("kind") == "slow_frame"]
    inits = [event for event in events if event.get("kind") == "qrhi_init"]
    large_scene = [event for event in events if event.get("kind") == "large_scene"]
    render_wait = [
        float(event["wait_ms"])
        for event in events
        if event.get("kind") in {"render_wait_done", "render_wait_timeout"}
    ]
    render_wait_timeouts = [
        event for event in events if event.get("kind") == "render_wait_timeout"
    ]
    summaries = [event for event in events if event.get("kind") == "summary"]
    assertion_failures = [
        event for event in events if event.get("kind") == "assertion_failed"
    ]

    print(f"log={path}")
    print(f"events={len(events)} artifacts={len(artifacts)} slow_frames={len(slow)}")
    if inits:
        init = inits[-1]
        print(
            "backend     "
            f"{init.get('backend')} fb_y_up={init.get('fb_y_up')} "
            f"clip_y_sign={init.get('clip_y_sign')}"
        )
    print(f"artifact_ms {_stats([float(event['total_ms']) for event in artifacts])}")
    print(f"surface_ms {_stats([float(event['surface_ms']) for event in artifacts])}")
    print(f"render_wait_ms {_stats(render_wait)} timeouts={len(render_wait_timeouts)}")
    print(f"large_scene count={len(large_scene)}")
    print(f"assertion_failures count={len(assertion_failures)}")
    if large_scene:
        last_large = large_scene[-1]
        print(
            "large_scene last "
            f"exact={last_large.get('exact')} total={last_large.get('total')} "
            f"no_blur={last_large.get('no_blur')} "
            f"reason={last_large.get('reason')}"
        )

    if artifacts:
        print("\nheaviest artifacts:")
        for event in sorted(
            artifacts,
            key=lambda item: float(item.get("surface_ms", 0.0)),
            reverse=True,
        )[:args.top]:
            print(
                f"  t={event.get('t_ms'):>9} total={float(event.get('total_ms', 0.0)):7.2f}ms "
                f"surface={float(event.get('surface_ms', 0.0)):7.2f}ms "
                f"verts={event.get('surface_vertices')} tris={event.get('surface_triangles')} "
                f"objects={event.get('objects')} large={event.get('large_scene')}"
            )

    if slow:
        print("\nslowest frames:")
        for event in sorted(
            slow,
            key=lambda item: float(item["interval_ms"]),
            reverse=True,
        )[:args.top]:
            print(
                f"  t={event.get('t_ms'):>9} interval={float(event['interval_ms']):8.2f}ms "
                f"render={float(event.get('render_ms', 0.0)):7.2f}ms "
                f"action={event.get('action_index')} loop={event.get('loop')}"
            )

    if summaries:
        summary = summaries[-1]
        print("\nsummary:")
        for key in (
            "run_id",
            "fps",
            "frames",
            "frame_interval_ms",
            "render_call_ms",
            "artifact_wait_ms",
            "surface_ms",
            "artifact_ms",
            "backend",
            "large_scene_events",
            "render_wait_timeouts",
        ):
            print(f"  {key}: {summary.get(key)}")
    elif events:
        last = events[-1]
        print("\npartial:")
        print(f"  last_event: kind={last.get('kind')} t_ms={last.get('t_ms')}")


if __name__ == "__main__":
    main()
