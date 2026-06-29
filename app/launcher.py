"""GPU + graphics-backend selection for launching casoCAD.

Neither the GPU nor the QRhi backend is chosen via an API call — both are driven
by environment variables that the GL/Vulkan driver reads when the rendering
context is first created. This module discovers the available GPUs, maps a
(gpu, backend) choice onto those env vars, and hands the result to the launcher,
which sets them and re-execs so the driver sees them from a clean start.

Discovery is best-effort and degrades gracefully: ``lspci`` gives the hardware
list, ``nvidia-smi`` refines NVIDIA naming. If nothing can be probed we still
offer the platform default.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


# Sentinel: set on the re-exec so we don't prompt / re-exec a second time.
_LAUNCHED_ENV = "CASOCAD_LAUNCHED"


@dataclass(frozen=True)
class Gpu:
    """A graphics adapter the user can target."""

    vendor: str  # "nvidia" | "intel" | "amd" | "unknown"
    name: str
    pci_id: str | None = None  # e.g. "0000:01:00.0"
    discrete: bool = False


@dataclass(frozen=True)
class Backend:
    key: str  # value for QRHI_BACKEND
    label: str


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout


_VGA_RE = re.compile(
    r"^(?P<pci>\S+)\s+(?:VGA compatible controller|3D controller|Display controller):\s+(?P<desc>.+)$",
    re.IGNORECASE,
)


def _vendor_of(desc: str) -> str:
    low = desc.lower()
    if "nvidia" in low:
        return "nvidia"
    if "intel" in low:
        return "intel"
    if "amd" in low or "advanced micro devices" in low or "ati" in low:
        return "amd"
    return "unknown"


def _normalize_pci(raw: str) -> str:
    # lspci prints "01:00.0"; the PRIME/DRI vars want a domain prefix.
    return raw if raw.count(":") >= 2 else f"0000:{raw}"


def gpu_selection_supported() -> bool:
    """Whether the user can pick a GPU for *graphics* on this OS.

    Only Linux exposes a reliable per-process GPU env switch for graphics
    (PRIME / DRI_PRIME). macOS picks the GPU itself, and on Windows the GPU is
    governed by the OS Graphics Preference / driver profile rather than a clean
    env knob — so on both we only offer the backend.
    """
    return platform.system() == "Linux"


def detect_gpus() -> list[Gpu]:
    """Return the GPUs found on this machine, integrated first, discrete last."""
    if not gpu_selection_supported():
        return []  # macOS / Windows: GPU is chosen by the OS, not by us

    gpus: list[Gpu] = []
    if shutil.which("lspci"):
        for line in _run(["lspci"]).splitlines():
            match = _VGA_RE.match(line.strip())
            if not match:
                continue
            desc = match.group("desc").strip()
            vendor = _vendor_of(desc)
            pci = _normalize_pci(match.group("pci"))
            is_3d = "3D controller" in line  # discrete on a hybrid laptop
            gpus.append(
                Gpu(vendor=vendor, name=desc, pci_id=pci, discrete=is_3d)
            )

    # Refine NVIDIA naming (lspci codenames are cryptic, e.g. "GA107M").
    if shutil.which("nvidia-smi"):
        smi = [
            line for line in _run(["nvidia-smi", "-L"]).splitlines() if line.strip()
        ]
        nv_index = 0
        refined: list[Gpu] = []
        for gpu in gpus:
            if gpu.vendor == "nvidia" and nv_index < len(smi):
                pretty = smi[nv_index].split(":", 1)[-1].split("(", 1)[0].strip()
                nv_index += 1
                refined.append(Gpu(gpu.vendor, pretty or gpu.name, gpu.pci_id, True))
            else:
                refined.append(gpu)
        gpus = refined

    if not gpus:
        gpus = [Gpu(vendor="unknown", name="Default GPU")]
    gpus.sort(key=lambda g: g.discrete)  # integrated first
    return gpus


def available_backends() -> list[Backend]:
    """QRhi backends plausibly available on this OS, best first."""
    system = platform.system()
    if system == "Darwin":
        return [Backend("metal", "Metal"), Backend("opengl", "OpenGL")]
    if system == "Windows":
        return [
            Backend("d3d11", "Direct3D 11"),
            Backend("vulkan", "Vulkan"),
            Backend("opengl", "OpenGL"),
        ]
    # Linux: OpenGL is the safe default here (see codegen perf memory).
    return [Backend("opengl", "OpenGL"), Backend("vulkan", "Vulkan")]


def build_env(gpu: Gpu | None, backend: Backend) -> dict[str, str]:
    """Env vars that pin *gpu* + *backend* for the launched process.

    ``gpu`` is ``None`` on platforms without per-process GPU selection (macOS),
    where only the backend is configurable.
    """
    env: dict[str, str] = {"QRHI_BACKEND": backend.key}
    if gpu is None:
        return env

    if platform.system() == "Windows":
        # Graphics-adapter selection on Windows is not a clean env knob (it is the
        # OS per-app Graphics Preference / driver profile / in-app DXGI pick), so
        # we only set the backend here. See the launcher discussion.
        return env

    if gpu.vendor == "nvidia" and gpu.discrete:
        # PRIME render offload onto the proprietary NVIDIA driver.
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        env["__VK_LAYER_NV_optimus"] = "NVIDIA_only"
        if backend.key == "opengl":
            env["QT_OPENGL"] = "desktop"
    else:
        # Integrated / non-NVIDIA: make sure we are NOT offloading to NVIDIA,
        # and point mesa at the chosen card by PCI id when we know it.
        env["__NV_PRIME_RENDER_OFFLOAD"] = "0"
        if gpu.pci_id:
            env["DRI_PRIME"] = f"pci-{gpu.pci_id.replace(':', '_').replace('.', '_')}"

    return env


def interactive_select(
    gpus: list[Gpu], backends: list[Backend]
) -> tuple[Gpu | None, Backend]:
    """Prompt on a TTY; return the chosen (gpu, backend).

    When ``gpus`` is empty (macOS) the GPU step is skipped and the GPU is None.
    """

    def _pick(title: str, items, render) -> int:
        print(f"\n{title}")
        for i, item in enumerate(items):
            print(f"  [{i}] {render(item)}")
        while True:
            raw = input(f"Choose [0-{len(items) - 1}] (default 0): ").strip()
            if raw == "":
                return 0
            if raw.isdigit() and 0 <= int(raw) < len(items):
                return int(raw)
            print("  invalid choice, try again")

    gpu: Gpu | None = None
    if gpus:
        gpu = gpus[
            _pick(
                "Select GPU:",
                gpus,
                lambda x: f"{x.name}  ({x.vendor}, {'discrete' if x.discrete else 'integrated'})",
            )
        ]
    b = backends[_pick("Select graphics backend:", backends, lambda x: x.label)]
    return gpu, b


def describe(gpu: Gpu | None, backend: Backend) -> str:
    if gpu is None:
        return f"default GPU via {backend.label}"
    return f"{gpu.name} [{gpu.vendor}] via {backend.label}"
