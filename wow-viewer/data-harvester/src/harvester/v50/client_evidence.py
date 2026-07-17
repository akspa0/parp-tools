"""Configurable client-library/build evidence (Spec 109 T010).

FR-014: client roots are runtime configuration, never a hardcoded default in source. Every
parameter that would otherwise tempt a hardcoded assumption -- the executable's filename (it
differs across client eras: ``Wow.exe``, ``WoWClient.exe``, ``WoW-64.exe``, ...), which paths are
"required" for a given build, and which glob identifies the archive set -- is supplied by the
caller rather than baked in here. This module only knows how to fingerprint whatever it is told to
look at.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from harvester.v50.contracts import HASH_PATTERN
from harvester.v50.identity import hash_file, hash_manifest

RESULT_PASS = "pass"
RESULT_FAIL = "fail"


@dataclass(frozen=True)
class ClientBuildEvidence:
    client_library_id: str
    build_id: str
    root_argument: str
    archive_catalog_identity: str
    required_paths: tuple[str, ...]
    reader_identity: str
    verification_time: str
    result: str
    executable_identity: str | None = None
    executable_relative_path: str | None = None
    missing_paths: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.client_library_id:
            raise ValueError("client_library_id is required")
        if not self.build_id:
            raise ValueError("build_id is required")
        if not self.root_argument:
            raise ValueError("root_argument is required (evidence must record the run's actual root)")
        if not HASH_PATTERN.fullmatch(self.archive_catalog_identity):
            raise ValueError("archive_catalog_identity must be a sha256:<hex> identity")
        if self.executable_identity is not None and not HASH_PATTERN.fullmatch(self.executable_identity):
            raise ValueError("executable_identity must be a sha256:<hex> identity when present")
        if self.result not in (RESULT_PASS, RESULT_FAIL):
            raise ValueError(f"result must be {RESULT_PASS!r} or {RESULT_FAIL!r}, got {self.result!r}")
        if self.result == RESULT_PASS and self.missing_paths:
            raise ValueError("result cannot be 'pass' while required_paths are missing")

    def identity(self) -> str:
        """Deterministic identity of this evidence record, for binding into a
        DatasetStoreManifest.client_build_evidence_id."""
        return hash_manifest(
            {
                "client_library_id": self.client_library_id,
                "build_id": self.build_id,
                "archive_catalog_identity": self.archive_catalog_identity,
                "executable_identity": self.executable_identity,
                "required_paths": list(self.required_paths),
                "reader_identity": self.reader_identity,
                "result": self.result,
            }
        )


def collect_client_build_evidence(
    root: Path,
    *,
    client_library_id: str,
    build_id: str,
    required_relative_paths: Sequence[str],
    reader_identity: str,
    executable_candidates: Iterable[str] = (),
    archive_glob: str = "**/*",
) -> ClientBuildEvidence:
    """Inspect a configured client root and produce fingerprinted evidence.

    Never raises for a missing/incomplete build -- that is a legitimate ``result="fail"`` finding,
    not an error in this function. It only raises if ``root`` itself does not exist, since there is
    nothing to fingerprint at all in that case.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"configured client root does not exist: {root}")

    missing = [rel for rel in required_relative_paths if not (root / rel).exists()]

    executable_identity: str | None = None
    executable_relative_path: str | None = None
    for candidate in executable_candidates:
        candidate_path = root / candidate
        if candidate_path.is_file():
            executable_identity = hash_file(candidate_path)
            executable_relative_path = candidate
            break
    executable_missing = bool(list(executable_candidates)) and executable_identity is None
    if executable_missing:
        missing = [*missing, "<no executable candidate found>"]

    catalog_entries = sorted(
        entry.relative_to(root).as_posix() for entry in root.glob(archive_glob) if entry.is_file()
    )
    archive_catalog_identity = hash_manifest({"entries": catalog_entries})

    return ClientBuildEvidence(
        client_library_id=client_library_id,
        build_id=build_id,
        root_argument=str(root),
        archive_catalog_identity=archive_catalog_identity,
        required_paths=tuple(required_relative_paths),
        reader_identity=reader_identity,
        verification_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        result=RESULT_FAIL if missing else RESULT_PASS,
        executable_identity=executable_identity,
        executable_relative_path=executable_relative_path,
        missing_paths=tuple(missing),
    )
