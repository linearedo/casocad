"""Phase 0 spike — confirm Approach B (compute → texture) works on real hardware.

This is the gate for the QRhiWidget renderer migration
(progress/switch_to_QRhiWidget_progress.md). It reproduces, in miniature, the
casoCAD interpreter pattern through Qt's portable QRhi:

    a storage buffer (stand-in for SDF bytecode)
      -> a COMPUTE shader reads it and writes an image into a texture
      -> we read the texture back and verify the pixels

If it works, the window turns GREEN and the console prints PASS — meaning the
"compute reads a storage buffer and writes a texture" path runs on your GPU,
which is exactly what the real renderer will do. RED / FAIL means the pixels
didn't match. A traceback means a QRhi API detail needs fixing (paste it back).

Run on your GPU machine:

    .venv/bin/python spikes/qrhi_compute_widget_spike.py

Optionally force a backend:  QRHI_BACKEND=vulkan|opengl  .venv/bin/python spikes/...
The shader is baked at startup with pyside6-qsb (no pre-built artifacts needed).
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QRhi,
    QRhiBuffer,
    QRhiComputePipeline,
    QRhiDepthStencilClearValue,
    QRhiReadbackDescription,
    QRhiReadbackResult,
    QRhiShaderResourceBinding,
    QRhiShaderStage,
    QRhiTexture,
    QShader,
)
from PySide6.QtWidgets import QApplication, QRhiWidget

TEX = 256          # texture is TEX x TEX
MAGIC = 200        # value placed in the storage buffer; becomes the blue channel

_COMPUTE_GLSL = """\
#version 440
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Params { uint vals[]; };
layout(rgba8, binding = 1) uniform writeonly image2D outImg;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(outImg);
    if (p.x >= sz.x || p.y >= sz.y) return;
    float r = float(p.x) / float(sz.x);
    float g = float(p.y) / float(sz.y);
    float b = float(vals[0] & 255u) / 255.0;   // proves the storage-buffer read
    imageStore(outImg, p, vec4(r, g, b, 1.0));
}
"""


def _bake_compute_shader() -> QShader:
    """Write the GLSL, bake it with pyside6-qsb, return a QShader."""
    tmp = tempfile.mkdtemp(prefix="qrhi_spike_")
    src = os.path.join(tmp, "raymarch.comp")
    out = os.path.join(tmp, "raymarch.comp.qsb")
    with open(src, "w") as f:
        f.write(_COMPUTE_GLSL)
    qsb = os.path.join(os.path.dirname(sys.executable), "pyside6-qsb")
    # GLSL 430 covers the OpenGL backend; SPIR-V (always emitted) covers Vulkan.
    subprocess.run([qsb, "--glsl", "430", "-o", out, src], check=True)
    with open(out, "rb") as f:
        return QShader.fromSerialized(QByteArray(f.read()))


_BACKENDS = {
    "vulkan": QRhiWidget.Api.Vulkan,
    "opengl": QRhiWidget.Api.OpenGL,
    "metal": getattr(QRhiWidget.Api, "Metal", None),
    "d3d11": getattr(QRhiWidget.Api, "Direct3D11", None),
}


class ComputeSpikeWidget(QRhiWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(360, 360)
        want = os.environ.get("QRHI_BACKEND", "").lower()
        if want in _BACKENDS and _BACKENDS[want] is not None:
            self.setApi(_BACKENDS[want])
        self._buf = None
        self._tex = None
        self._pipe = None
        self._srb = None
        self._uploaded = False
        self._readback = None
        self._verdict = None  # None=pending, True=PASS, False=FAIL

    # Qt calls this once the QRhi + render target exist.
    def initialize(self, cb) -> None:
        rhi = self.rhi()
        if self._pipe is not None:
            return  # already built (initialize can be called again on resize)
        print("BACKEND:", rhi.backendName(),
              "| compute supported:", rhi.isFeatureSupported(QRhi.Feature.Compute))

        shader = _bake_compute_shader()
        print("SHADER valid:", shader.isValid(), "stage:", shader.stage())

        self._buf = rhi.newBuffer(
            QRhiBuffer.Type.Static, QRhiBuffer.UsageFlag.StorageBuffer, 16)
        self._buf.create()

        self._tex = rhi.newTexture(
            QRhiTexture.Format.RGBA8, QSize(TEX, TEX), 1,
            QRhiTexture.Flag.UsedWithLoadStore)
        self._tex.create()

        self._srb = rhi.newShaderResourceBindings()
        self._srb.setBindings([
            QRhiShaderResourceBinding.bufferLoad(
                0, QRhiShaderResourceBinding.StageFlag.ComputeStage, self._buf),
            QRhiShaderResourceBinding.imageStore(
                1, QRhiShaderResourceBinding.StageFlag.ComputeStage, self._tex, 0),
        ])
        self._srb.create()

        self._pipe = rhi.newComputePipeline()
        self._pipe.setShaderStage(QRhiShaderStage(QRhiShaderStage.Type.Compute, shader))
        self._pipe.setShaderResourceBindings(self._srb)
        ok = self._pipe.create()
        print("PIPELINE create:", ok)

    # Qt calls this every frame and HANDS us the command buffer (no
    # beginOffscreenFrame — that is why the widget path works where the headless
    # harness did not).
    def render(self, cb) -> None:
        rhi = self.rhi()
        rub = rhi.nextResourceUpdateBatch()
        if not self._uploaded:
            rub.uploadStaticBuffer(self._buf, struct.pack("<4I", MAGIC, 0, 0, 0))
            self._uploaded = True

        # 1) compute pass: read the storage buffer, write the texture.
        cb.beginComputePass(rub)
        cb.setComputePipeline(self._pipe)
        cb.setShaderResources(self._srb)
        cb.dispatch((TEX + 7) // 8, (TEX + 7) // 8, 1)
        cb.endComputePass()

        # 2) queue a readback of the texture so we can verify on a later frame.
        if self._readback is None and self._verdict is None:
            self._readback = QRhiReadbackResult()
            rb = rhi.nextResourceUpdateBatch()
            rb.readBackTexture(QRhiReadbackDescription(self._tex), self._readback)
            cb.resourceUpdate(rb)

        # 3) check a completed readback.
        if self._readback is not None and self._verdict is None:
            data = bytes(self._readback.data)
            if data:
                # pixel (0,0) should be (r=0, g=0, b=MAGIC, a=255)
                b = data[2]
                self._verdict = (b == MAGIC)
                print(f"READBACK bytes={len(data)} pixel(0,0).blue={b} expected={MAGIC}")
                print("RESULT:", "PASS — compute→texture works on this GPU"
                      if self._verdict else "FAIL — pixel mismatch")
                self._readback = None

        # 4) paint the widget: green=PASS, red=FAIL, grey=pending.
        if self._verdict is True:
            clear = QColor.fromRgbF(0.1, 0.7, 0.2, 1.0)
        elif self._verdict is False:
            clear = QColor.fromRgbF(0.8, 0.15, 0.15, 1.0)
        else:
            clear = QColor.fromRgbF(0.2, 0.2, 0.25, 1.0)
        cb.beginPass(self.renderTarget(), clear, QRhiDepthStencilClearValue(1.0, 0))
        cb.endPass()

        if self._verdict is None:
            self.update()  # keep rendering until the readback lands


def main() -> int:
    app = QApplication(sys.argv)
    w = ComputeSpikeWidget()
    w.setWindowTitle("QRhi compute→texture spike — green = PASS")
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
