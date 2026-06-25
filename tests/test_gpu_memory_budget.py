from __future__ import annotations

from app.meshing.viewer import gpu_memory
from app.meshing.viewer.gpu_memory import (
    GpuMemoryInfo,
    GpuRenderDeviceInfo,
    choose_preview_budget_bytes,
)


def test_gpu_budget_prefers_free_vram_for_filled_preview() -> None:
    budget, source = choose_preview_budget_bytes(
        gpu_info=GpuMemoryInfo(
            source="nvidia-smi",
            free_bytes=4 * 1024 * 1024 * 1024,
            total_bytes=6 * 1024 * 1024 * 1024,
        ),
        available_ram_bytes=1024 * 1024 * 1024,
        wireframe_enabled=False,
    )

    assert source == "nvidia-smi"
    assert budget == 1932735283


def test_gpu_budget_reduces_when_wireframe_is_enabled() -> None:
    filled_budget, _ = choose_preview_budget_bytes(
        gpu_info=GpuMemoryInfo(
            source="nvidia-smi",
            free_bytes=4 * 1024 * 1024 * 1024,
            total_bytes=6 * 1024 * 1024 * 1024,
        ),
        available_ram_bytes=None,
        wireframe_enabled=False,
    )
    wire_budget, _ = choose_preview_budget_bytes(
        gpu_info=GpuMemoryInfo(
            source="nvidia-smi",
            free_bytes=4 * 1024 * 1024 * 1024,
            total_bytes=6 * 1024 * 1024 * 1024,
        ),
        available_ram_bytes=None,
        wireframe_enabled=True,
    )

    assert wire_budget < filled_budget


def test_gpu_budget_falls_back_to_system_ram() -> None:
    budget, source = choose_preview_budget_bytes(
        gpu_info=None,
        available_ram_bytes=8 * 1024 * 1024 * 1024,
        wireframe_enabled=False,
    )

    assert source == "system-ram"
    assert budget == 687194767


def test_nvidia_probe_is_ignored_for_non_nvidia_render_device(monkeypatch) -> None:
    def fake_check_output(*args, **kwargs) -> str:
        return "4096, 6144, 0x25A2, NVIDIA GeForce RTX 3050 Laptop GPU\n"

    monkeypatch.setattr(gpu_memory.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(gpu_memory.sys, "platform", "linux")

    info = gpu_memory.query_gpu_memory_info(
        render_device=GpuRenderDeviceInfo(
            backend_name="Vulkan",
            vendor_id=0x8086,
            device_id=0x1234,
            device_name="Intel Graphics",
            device_type="IntegratedDevice",
        )
    )

    assert info is None


def test_nvidia_probe_is_ignored_when_render_device_is_unknown(monkeypatch) -> None:
    def fake_check_output(*args, **kwargs) -> str:
        return "4096, 6144, 0x25A2, NVIDIA GeForce RTX 3050 Laptop GPU\n"

    monkeypatch.setattr(gpu_memory.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(gpu_memory.sys, "platform", "linux")
    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    monkeypatch.delenv("__VK_LAYER_NV_optimus", raising=False)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)

    info = gpu_memory.query_gpu_memory_info(render_device=None)

    assert info is None


def test_nvidia_probe_uses_nvidia_launcher_environment(monkeypatch) -> None:
    def fake_check_output(*args, **kwargs) -> str:
        return "4096, 6144, 0x25A2, NVIDIA GeForce RTX 3050 Laptop GPU\n"

    monkeypatch.setattr(gpu_memory.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(gpu_memory.sys, "platform", "linux")
    monkeypatch.setenv("__NV_PRIME_RENDER_OFFLOAD", "1")

    info = gpu_memory.query_gpu_memory_info(render_device=None)

    assert info == GpuMemoryInfo(
        source="nvidia-smi",
        free_bytes=4096 * 1024 * 1024,
        total_bytes=6144 * 1024 * 1024,
    )


def test_nvidia_probe_matches_nvidia_render_device(monkeypatch) -> None:
    def fake_check_output(*args, **kwargs) -> str:
        return "4096, 6144, 0x25A2, NVIDIA GeForce RTX 3050 Laptop GPU\n"

    monkeypatch.setattr(gpu_memory.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(gpu_memory.sys, "platform", "linux")

    info = gpu_memory.query_gpu_memory_info(
        render_device=GpuRenderDeviceInfo(
            backend_name="Vulkan",
            vendor_id=0x10DE,
            device_id=0x25A2,
            device_name="NVIDIA GeForce RTX 3050 Laptop GPU",
            device_type="DiscreteDevice",
        )
    )

    assert info == GpuMemoryInfo(
        source="nvidia-smi",
        free_bytes=4096 * 1024 * 1024,
        total_bytes=6144 * 1024 * 1024,
    )
