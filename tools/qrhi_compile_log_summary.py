#!/usr/bin/env python3
from __future__ import annotations

"""Summarize QRhi shader/pipeline compile telemetry from casoCAD logs."""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


_SOURCE_RE = re.compile(
    r"qrhi: (?P<event>codegen baked variant|async bake done|prewarm bake start)"
    r"(?: [^ ]+=[^ ]+)* (?P<label>kinds=\[.*?\].*?) source_bytes=(?P<src>\d+)"
)
_QSB_RE = re.compile(r"qsb=(?P<qsb>[0-9.]+) ms")
_PIPE_RE = re.compile(
    r"qrhi: (?P<event>prewarm pipeline driver-compiled|pipeline driver-compiled)"
    r" (?P<label>kinds=\[.*?\].*?) backend=(?P<backend>[^ ]+) "
    r"source_bytes=(?P<src>\d+) in (?P<seconds>[0-9.]+)s"
)
_QSB_LABEL_RE = re.compile(r" qsb=[0-9.]+ ms")


@dataclass(frozen=True)
class CompileEvent:
    event: str
    label: str
    source_bytes: int
    qsb_ms: float | None = None
    pipeline_s: float | None = None
    backend: str = ""


@dataclass(frozen=True)
class SignatureSummary:
    label: str
    source_bytes: int
    backends: tuple[str, ...]
    max_qsb_ms: float | None
    max_pipeline_s: float | None
    event_count: int


def parse_events(text: str) -> list[CompileEvent]:
    events: list[CompileEvent] = []
    for line in text.splitlines():
        pipe = _PIPE_RE.search(line)
        if pipe:
            events.append(
                CompileEvent(
                    event=pipe.group("event"),
                    label=pipe.group("label"),
                    source_bytes=int(pipe.group("src")),
                    pipeline_s=float(pipe.group("seconds")),
                    backend=pipe.group("backend"),
                )
            )
            continue
        source = _SOURCE_RE.search(line)
        if source:
            qsb = _QSB_RE.search(line)
            label = _QSB_LABEL_RE.sub("", source.group("label"))
            events.append(
                CompileEvent(
                    event=source.group("event"),
                    label=label,
                    source_bytes=int(source.group("src")),
                    qsb_ms=float(qsb.group("qsb")) if qsb else None,
                )
            )
    return events


def _duration_key(event: CompileEvent) -> float:
    if event.pipeline_s is not None:
        return event.pipeline_s * 1000.0
    if event.qsb_ms is not None:
        return event.qsb_ms
    return 0.0


def summarize_by_signature(events: list[CompileEvent]) -> list[SignatureSummary]:
    grouped: dict[str, list[CompileEvent]] = {}
    for event in events:
        grouped.setdefault(event.label, []).append(event)
    rows = []
    for label, items in grouped.items():
        qsb_values = [item.qsb_ms for item in items if item.qsb_ms is not None]
        pipe_values = [item.pipeline_s for item in items if item.pipeline_s is not None]
        backends = sorted({item.backend for item in items if item.backend})
        rows.append(
            SignatureSummary(
                label=label,
                source_bytes=max(item.source_bytes for item in items),
                backends=tuple(backends),
                max_qsb_ms=max(qsb_values) if qsb_values else None,
                max_pipeline_s=max(pipe_values) if pipe_values else None,
                event_count=len(items),
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -1.0 if item.max_pipeline_s is None else item.max_pipeline_s,
            -1.0 if item.max_qsb_ms is None else item.max_qsb_ms / 1000.0,
            item.source_bytes,
        ),
        reverse=True,
    )


def format_summary(events: list[CompileEvent]) -> str:
    if not events:
        return "No QRhi compile telemetry found."
    rows = sorted(events, key=_duration_key, reverse=True)
    lines = [
        "event                              backend source_bytes qsb_ms  pipeline_s label",
        "---------------------------------- ------- ------------ ------- ---------- -----",
    ]
    for item in rows:
        qsb = "" if item.qsb_ms is None else f"{item.qsb_ms:.1f}"
        pipe = "" if item.pipeline_s is None else f"{item.pipeline_s:.2f}"
        lines.append(
            f"{item.event[:34]:34s} {item.backend[:7]:7s} "
            f"{item.source_bytes:12d} {qsb:7s} {pipe:10s} {item.label}"
        )
    signature_rows = summarize_by_signature(events)
    lines.extend([
        "",
        "signature summary",
        "backends source_bytes max_qsb_ms max_pipeline_s events label",
        "-------- ------------ ---------- -------------- ------ -----",
    ])
    for item in signature_rows:
        backends = ",".join(item.backends)
        qsb = "" if item.max_qsb_ms is None else f"{item.max_qsb_ms:.1f}"
        pipe = "" if item.max_pipeline_s is None else f"{item.max_pipeline_s:.2f}"
        lines.append(
            f"{backends[:8]:8s} {item.source_bytes:12d} {qsb:10s} "
            f"{pipe:14s} {item.event_count:6d} {item.label}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logfile",
        nargs="?",
        help="Log file to parse. Reads stdin when omitted or '-'.",
    )
    args = parser.parse_args(argv)
    if args.logfile and args.logfile != "-":
        text = Path(args.logfile).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    print(format_summary(parse_events(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
