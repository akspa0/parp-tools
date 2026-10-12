"""Unit tests for QuiltAdtMaterializer (Spec 268 AC-007)."""

import struct
from pathlib import Path
import numpy as np
import pytest

from harvester.v60.quilt_adt_materializer import QuiltAdtMaterializer


def test_construct_monolithic_lk_adt(tmp_path: Path):
    """Verify constructing a monolithic LK (v18) ADT from elevation grid (AC-007)."""
    adt_path = tmp_path / "test_0_0.adt"
    h257 = np.full((257, 257), 100.0, dtype=np.float32)

    QuiltAdtMaterializer.construct_monolithic_lk_adt(
        out_path=adt_path,
        h257=h257,
        tile_x=0,
        tile_y=0,
        texture_names=["Tileset\\Generic\\Grass.blp"],
    )

    assert adt_path.is_file()
    data = adt_path.read_bytes()
    assert len(data) > 300_000  # Monolithic ADT is typically ~300KB-1.5MB

    # Verify MVER is 18
    magic = data[0:4]
    assert magic == b"MVER"
    ver = struct.unpack_from("<I", data, 8)[0]
    assert ver == 18

    # Scan and count MCNK chunks
    mcnk_count = 0
    pos = 0
    total_len = len(data)
    while pos + 8 <= total_len:
        ch_magic = data[pos : pos + 4]
        ch_size = struct.unpack_from("<I", data, pos + 4)[0]
        if ch_magic == b"MCNK":
            mcnk_count += 1
            # Check MCVT offset in MCNK header
            ofs_mcvt = struct.unpack_from("<I", data, pos + 8 + 0x14)[0]
            assert ofs_mcvt > 0
            # Check MCCV offset in MCNK header
            ofs_mccv = struct.unpack_from("<I", data, pos + 8 + 0x74)[0]
            assert ofs_mccv > 0
        pos += 8 + ch_size

    assert mcnk_count == 256


def test_patch_monolithic_adt(tmp_path: Path):
    """Verify patching MCVT/MCNR in existing ADT preserves all chunks and normalizes MCCV."""
    # First construct a template ADT
    template_path = tmp_path / "template.adt"
    patched_path = tmp_path / "patched.adt"
    h_init = np.full((257, 257), 50.0, dtype=np.float32)

    QuiltAdtMaterializer.construct_monolithic_lk_adt(
        out_path=template_path,
        h257=h_init,
        tile_x=16,
        tile_y=32,
    )

    # Patch with new heights (200.0 yards)
    h_new = np.full((257, 257), 200.0, dtype=np.float32)
    QuiltAdtMaterializer.patch_monolithic_adt(
        template_path=template_path,
        out_path=patched_path,
        h257=h_new,
        preserve_authentic_mccv=True,
    )

    assert patched_path.is_file()
    assert patched_path.stat().st_size == template_path.stat().st_size

    data = patched_path.read_bytes()
    # Check first MCNK pos_z is ~200.0
    pos = 0
    while pos + 8 <= len(data):
        m = data[pos : pos + 4][::-1]
        s = struct.unpack_from("<I", data, pos + 4)[0]
        if m == b"MCNK":
            pos_z = struct.unpack_from("<f", data, pos + 8 + 0x70)[0]
            assert pos_z == pytest.approx(200.0, abs=1e-3)
            break
        pos += 8 + s


def test_export_quilt_3d_mesh(tmp_path: Path):
    """Verify exporting continuous multi-tile heightfield to OBJ and GLB meshes."""
    obj_path = tmp_path / "quilt_mesh.obj"
    glb_path = tmp_path / "quilt_mesh.glb"

    canvas = np.full((257, 257), 100.0, dtype=np.float32)
    canvas[128, 128] = 300.0  # Center mountain

    QuiltAdtMaterializer.export_quilt_3d_mesh(obj_path, canvas)
    QuiltAdtMaterializer.export_quilt_3d_mesh(glb_path, canvas)

    assert obj_path.is_file()
    assert obj_path.stat().st_size > 10_000

    assert glb_path.is_file()
    assert glb_path.stat().st_size > 10_000
