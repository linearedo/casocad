from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys


_MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuMemoryInfo:
    source: str
    free_bytes: int | None
    total_bytes: int | None
    unified_memory: bool = False


@dataclass(frozen=True)
class GpuRenderDeviceInfo:
    backend_name: str
    vendor_id: int | None
    device_id: int | None
    device_name: str
    device_type: str


def query_gpu_memory_info(
    render_device: GpuRenderDeviceInfo | None = None,
) -> GpuMemoryInfo | None:
    info = _query_nvidia_smi(render_device)
    if info is not None:
        return info
    if sys.platform == "darwin":
        return _query_macos_unified_memory()
    return None


def choose_preview_budget_bytes(
    *,
    gpu_info: GpuMemoryInfo | None,
    available_ram_bytes: int | None,
    wireframe_enabled: bool,
) -> tuple[int, str]:
    if gpu_info is not None and gpu_info.free_bytes is not None:
        factor = 0.22 if wireframe_enabled else 0.45
        return _clamp_budget(int(gpu_info.free_bytes * factor)), gpu_info.source
    if gpu_info is not None and gpu_info.total_bytes is not None:
        factor = 0.12 if wireframe_enabled else 0.25
        return _clamp_budget(int(gpu_info.total_bytes * factor)), gpu_info.source
    if available_ram_bytes is not None:
        factor = 0.04 if wireframe_enabled else 0.08
        return _clamp_budget(int(available_ram_bytes * factor)), "system-ram"
    return 512 * _MIB, "fallback"


def _query_nvidia_smi(
    render_device: GpuRenderDeviceInfo | None,
) -> GpuMemoryInfo | None:
    if sys.platform not in ("linux", "win32"):
        return None
    if not _nvidia_probe_matches_render_context(render_device):
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total,pci.device_id,name",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            timeout=1.0,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError):
        return None
    rows: list[tuple[int, int, int | None, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            free = int(parts[0]) * _MIB
            total = int(parts[1]) * _MIB
        except ValueError:
            continue
        rows.append((free, total, _parse_hex_device_id(parts[2]), parts[3]))
    if not rows:
        return None
    candidates = rows
    if render_device is not None:
        matched = [
            row
            for row in rows
            if row[2] is not None
            and render_device.device_id is not None
            and row[2] == (render_device.device_id & 0xFFFF)
        ]
        if not matched:
            device_name = render_device.device_name.lower()
            matched = [row for row in rows if row[3].lower() in device_name]
        if matched:
            candidates = matched
        elif len(rows) != 1:
            return None
    best_free, best_total, _device_id, _name = max(candidates, key=lambda row: row[0])
    return GpuMemoryInfo(
        source="nvidia-smi",
        free_bytes=best_free,
        total_bytes=best_total,
    )


def _parse_hex_device_id(value: str) -> int | None:
    token = value.split()[0]
    try:
        parsed = int(token, 16)
    except ValueError:
        return None
    return parsed & 0xFFFF


def _nvidia_probe_matches_render_context(
    render_device: GpuRenderDeviceInfo | None,
) -> bool:
    if render_device is not None and render_device.vendor_id == 0x10DE:
        return True
    return (
        os.environ.get("__NV_PRIME_RENDER_OFFLOAD") == "1"
        or os.environ.get("__VK_LAYER_NV_optimus") == "NVIDIA_only"
        or os.environ.get("__GLX_VENDOR_LIBRARY_NAME") == "nvidia"
    )


def _query_macos_unified_memory() -> GpuMemoryInfo | None:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            encoding="utf-8",
            timeout=1.0,
            stderr=subprocess.DEVNULL,
        )
        total = int(output.strip())
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError, ValueError):
        return None
    return GpuMemoryInfo(
        source="macos-unified-memory",
        free_bytes=None,
        total_bytes=total,
        unified_memory=True,
    )


def _clamp_budget(value: int) -> int:
    return max(128 * _MIB, min(value, 3 * 1024 * _MIB))


__all__ = [
    "GpuMemoryInfo",
    "GpuRenderDeviceInfo",
    "choose_preview_budget_bytes",
    "query_gpu_memory_info",
]
