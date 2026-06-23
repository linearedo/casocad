"""SDF interpreter shader source + assembly (backend-agnostic).

The moderngl OpenGL renderers that used to live here were removed in the QRhi
migration. Only the backend-agnostic pieces remain — the GLSL ``shaders/`` and
``shader_assembly`` — which the QRhi renderer consumes. (Directory name kept for
now to avoid churn.)
"""
