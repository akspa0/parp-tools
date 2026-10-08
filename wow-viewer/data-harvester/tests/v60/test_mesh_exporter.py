"""Unit tests for OBJ and GLB terrain mesh exporters (Spec 263)."""

import struct
from pathlib import Path
import numpy as np
from PIL import Image

from harvester.v60.mesh_exporter import export_glb_mesh, export_obj_mesh


def test_export_obj_mesh(tmp_path: Path):
    h = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    tex_path = tmp_path / "test_tex.png"
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(tex_path)

    obj_path = tmp_path / "test_terrain.obj"
    export_obj_mesh(h, tex_path, obj_path)

    assert obj_path.is_file()
    assert (tmp_path / "test_terrain.mtl").is_file()

    content = obj_path.read_text(encoding="utf-8")
    assert "mtllib test_terrain.mtl" in content
    assert "usemtl terrain" in content
    # 16x16 = 256 vertices
    assert content.count("v ") == 256
    # 15x15 quads * 2 = 450 faces
    assert content.count("f ") == 450


def test_export_glb_mesh(tmp_path: Path):
    h = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    glb_path = tmp_path / "test_terrain.glb"

    export_glb_mesh(h, img, glb_path)

    assert glb_path.is_file()
    data = glb_path.read_bytes()

    # GLB header verification
    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(data)
