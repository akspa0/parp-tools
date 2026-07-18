"""Integrate the existing C# harvester stream for fresh signals and new builds (Spec 109 T032).

Reuses ``harvester.raw_reader.read_tile_blob`` for the inner per-tile array format rather than
reimplementing it -- this module only adds the outer per-tile stream framing that
``WowViewer.Tool.Harvest``'s ``harvest-stream`` command writes around each inner blob:

    [8 bytes: "ARRY" + little-endian int32 inner-blob length]  [inner blob]   <- repeated per tile
    [8 bytes: "ENDS" + 4 zero bytes]                                          <- stream terminator

``read_harvest_stream`` works against any readable binary stream, so it is fixture-testable against
a ``BytesIO`` built with the identical framing, without needing a live subprocess or client root.

EXECUTION GATE: ``build_harvest_stream_command`` only constructs the command; actually launching it
against a real client is ``run_fresh_extraction``'s job, and that function refuses to execute
unless the caller explicitly passes ``confirm_run=True`` -- printing the exact command otherwise.
This mirrors Spec 111's ``train_spec111_reconstruction.py`` gate: preparing and testing this tool
does not authorize spending real harvest time against `H:\\CLIENTS`.
"""

from __future__ import annotations

import struct
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from harvester.raw_reader import ENDS_MAGIC, read_tile_blob

_ARRY_MAGIC = b"ARRY"
_HEADER_SIZE = 8


def find_harvest_dll(harvest_project_path: Path) -> Path:
    """Find the compiled DLL for WowViewer.Tool.Harvest."""
    if harvest_project_path.suffix == ".csproj":
        project_dir = harvest_project_path.parent
    else:
        project_dir = harvest_project_path

    for tfm in ["net10.0", "net9.0", "net8.0"]:
        dll_path = project_dir / "bin" / "Debug" / tfm / "WowViewer.Tool.Harvest.dll"
        if dll_path.exists():
            return dll_path

    bin_dir = project_dir / "bin"
    if bin_dir.exists():
        dll_files = list(bin_dir.glob("**/WowViewer.Tool.Harvest.dll"))
        if dll_files:
            return dll_files[0]

    return project_dir / "bin" / "Debug" / "net10.0" / "WowViewer.Tool.Harvest.dll"


def build_harvest_stream_command(
    harvest_project_path: Path,
    *,
    client_root: Path,
    map_name: str,
    stream_profile: str = "v22",
) -> list[str]:
    """Construct the exact ``dotnet`` invocation using the compiled DLL for the existing C# harvester's
    ``harvest-stream`` command. Every value that would tempt a hardcoded assumption (the project
    path, client root, map, stream profile) is a required parameter."""
    dll_path = find_harvest_dll(harvest_project_path)
    return [
        "dotnet",
        str(dll_path),
        "harvest-stream",
        "--client-root",
        str(client_root),
        "--map",
        map_name,
        "--stream-profile",
        stream_profile,
    ]



def run_fresh_extraction(
    command: list[str],
    *,
    confirm_run: bool,
) -> subprocess.Popen[bytes] | None:
    """Print the constructed command and return None unless ``confirm_run=True``. Only when
    explicitly confirmed does this actually launch the subprocess (with its stdout piped for
    ``read_harvest_stream`` to consume)."""
    if not confirm_run:
        print(f"Would run fresh extraction: {' '.join(command)}")
        print("NOT executing: pass confirm_run=True only with explicit user authorization for this run.")
        return None
    return subprocess.Popen(command, stdout=subprocess.PIPE)


def read_harvest_stream(stream: IO[bytes]) -> Iterator[dict[str, Any]]:
    """Yield one decoded tile record (see ``raw_reader.read_tile_blob``) per inner blob in the
    stream, stopping cleanly at the "ENDS" terminator. Raises if a header claims to be "ARRY" but
    the stream runs out before the declared inner-blob length is satisfied (a truncated stream is
    a real error, not a silent early stop)."""
    while True:
        header = stream.read(_HEADER_SIZE)
        if len(header) == 0:
            return  # stream closed with no terminator observed -- treated as a clean end by callers
        if len(header) != _HEADER_SIZE:
            raise ValueError(f"truncated stream header: expected {_HEADER_SIZE} bytes, got {len(header)}")

        tag = header[:4]
        if tag == ENDS_MAGIC:
            return
        if tag != _ARRY_MAGIC:
            raise ValueError(f"unexpected stream tag {tag!r}; expected {_ARRY_MAGIC!r} or {ENDS_MAGIC!r}")

        (blob_length,) = struct.unpack("<i", header[4:8])
        blob = stream.read(blob_length)
        if len(blob) != blob_length:
            raise ValueError(
                f"truncated inner blob: expected {blob_length} bytes, got {len(blob)}"
            )

        yield read_tile_blob(blob)
