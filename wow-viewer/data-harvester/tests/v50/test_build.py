"""Spec 109 (T032 coverage): the harvest-stream reader and the fresh-extraction execution gate.
Preparing/testing this tool must never itself launch a real subprocess against a client root."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from harvester.v50.build import build_harvest_stream_command, read_harvest_stream, run_fresh_extraction


def _encode_inner_blob(arrays: dict[str, np.ndarray]) -> bytes:
    """Build one inner ARRY blob exactly matching harvester.raw_reader's documented format, so the
    outer-framing reader under test is proven against the real inner format, not a stand-in."""
    stream = io.BytesIO()
    stream.write(b"ARRY")
    metadata = json.dumps({"tile_id": 0}).encode("utf-8")
    stream.write(struct.pack("<I", len(metadata)))
    stream.write(metadata)

    for name, array in arrays.items():
        name_bytes = name.encode("utf-8")
        stream.write(struct.pack("<I", len(name_bytes)))
        stream.write(name_bytes)
        stream.write(struct.pack("<I", array.ndim))
        stream.write(struct.pack(f"<{array.ndim}I", *array.shape))
        dtype_tag = {"float32": b"<f4", "uint8": b"|u1"}[str(array.dtype)]
        stream.write(dtype_tag.ljust(8, b"\x00"))
        data = array.tobytes()
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)

    stream.write(b"ENDS")
    stream.write(struct.pack("<I", 0))
    return stream.getvalue()


def _wrap_outer_frame(inner_blob: bytes) -> bytes:
    header = b"ARRY" + struct.pack("<i", len(inner_blob))
    return header + inner_blob


def test_read_harvest_stream_yields_one_record_per_tile_and_stops_at_ends():
    tile_a = _encode_inner_blob({"height_257": np.ones((2, 2), dtype=np.float32)})
    tile_b = _encode_inner_blob({"height_257": np.zeros((2, 2), dtype=np.float32)})
    stream_bytes = _wrap_outer_frame(tile_a) + _wrap_outer_frame(tile_b) + b"ENDS" + struct.pack("<I", 0)

    records = list(read_harvest_stream(io.BytesIO(stream_bytes)))

    assert len(records) == 2
    np.testing.assert_array_equal(records[0]["height_257"], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(records[1]["height_257"], np.zeros((2, 2), dtype=np.float32))


def test_read_harvest_stream_raises_on_a_truncated_inner_blob():
    tile_a = _encode_inner_blob({"height_257": np.ones((2, 2), dtype=np.float32)})
    truncated = _wrap_outer_frame(tile_a)[:-5]  # cut off the end of the declared-length blob

    with pytest.raises(ValueError, match="truncated inner blob"):
        list(read_harvest_stream(io.BytesIO(truncated)))


def test_read_harvest_stream_raises_on_an_unrecognized_tag():
    bogus = b"XXXX" + struct.pack("<i", 4) + b"data"

    with pytest.raises(ValueError, match="unexpected stream tag"):
        list(read_harvest_stream(io.BytesIO(bogus)))


def test_read_harvest_stream_on_an_empty_stream_yields_nothing():
    assert list(read_harvest_stream(io.BytesIO(b""))) == []


class TestExecutionGate:
    def test_run_fresh_extraction_without_confirmation_does_not_launch_anything(self, monkeypatch):
        launched = []
        monkeypatch.setattr(
            "subprocess.Popen", lambda *a, **k: launched.append((a, k)) or object()
        )

        result = run_fresh_extraction(["dotnet", "run"], confirm_run=False)

        assert result is None
        assert launched == []  # subprocess.Popen must never have been called

    def test_run_fresh_extraction_only_launches_when_explicitly_confirmed(self, monkeypatch):
        launched = []

        class _FakeProcess:
            pass

        def _fake_popen(command, stdout=None):
            launched.append(command)
            return _FakeProcess()

        monkeypatch.setattr("subprocess.Popen", _fake_popen)

        result = run_fresh_extraction(["dotnet", "run"], confirm_run=True)

        assert isinstance(result, _FakeProcess)
        assert launched == [["dotnet", "run"]]


def test_build_harvest_stream_command_embeds_every_caller_supplied_value(tmp_path: Path):
    project_path = tmp_path / "WowViewer.Tool.Harvest"

    command = build_harvest_stream_command(
        project_path,
        client_root=tmp_path / "clients" / "0.5.3.3368",
        map_name="Azeroth",
        stream_profile="v22",
    )

    assert any(str(project_path) in item for item in command)
    assert str(tmp_path / "clients" / "0.5.3.3368") in command
    assert "Azeroth" in command
    assert "v22" in command

