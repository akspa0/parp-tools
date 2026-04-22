from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_OUTPUT_NAME = "train_manifest.json"
DEFAULT_DEV_OUTPUT_NAME = "dev_holdout_manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return payload


def resolve_manifest_entry_path(manifest_path: Path, raw_value: object) -> Path | None:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return None

    candidate = Path(raw_text)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def to_posix_relative(path: Path, relative_to: Path) -> str:
    return os.path.relpath(path, relative_to).replace("\\", "/")


@dataclass(frozen=True)
class BundleManifestSpec:
    label: str
    manifest_path: Path
    output_name: str
    cache_label: str


class BundleBuilder:
    def __init__(
        self,
        *,
        bundle_root: Path,
        include_source_json: bool,
        limit: int | None,
    ) -> None:
        self.bundle_root = bundle_root
        self.include_source_json = include_source_json
        self.limit = limit if limit and limit > 0 else None
        self.manifests_dir = bundle_root / "manifests"
        self.cache_dir = bundle_root / "cache"
        self.metadata_dir = bundle_root / "metadata"
        self.source_json_dir = bundle_root / "source_json"
        self._copied_shards: dict[tuple[str, Path], Path] = {}
        self._copied_source_json: dict[tuple[str, Path], Path] = {}

    def _relative_source_subpath(self, spec: BundleManifestSpec, source_path: Path, dataset_key: str, tile_name: str, default_suffix: str) -> Path:
        try:
            relative_to_manifest = source_path.resolve().relative_to(spec.manifest_path.parent.resolve())
        except ValueError:
            relative_to_manifest = None

        if relative_to_manifest is not None and len(relative_to_manifest.parts) > 1 and relative_to_manifest.parts[0].lower() == "shards":
            return Path(*relative_to_manifest.parts)

        safe_dataset_key = dataset_key or "unknown_dataset"
        safe_tile_name = tile_name or source_path.stem or "sample"
        suffix = source_path.suffix or default_suffix
        return Path("shards") / safe_dataset_key / f"{safe_tile_name}{suffix}"

    def _target_shard_path(self, spec: BundleManifestSpec, source_shard: Path, dataset_key: str, tile_name: str) -> Path:
        relative_subpath = self._relative_source_subpath(spec, source_shard, dataset_key, tile_name, ".npz")
        return self.cache_dir / spec.cache_label / relative_subpath

    def _target_source_json_path(self, spec: BundleManifestSpec, source_json: Path, dataset_key: str, tile_name: str) -> Path:
        relative_subpath = self._relative_source_subpath(spec, source_json, dataset_key, tile_name, ".json")
        return self.source_json_dir / spec.cache_label / relative_subpath

    def _copy_file_once(self, cache: dict[tuple[str, Path], Path], cache_label: str, source_path: Path, dest_path: Path) -> Path:
        key = (cache_label, source_path.resolve())
        cached = cache.get(key)
        if cached is not None:
            return cached

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        cache[key] = dest_path
        return dest_path

    def _portable_manifest_entry(self, spec: BundleManifestSpec, manifest_output_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
        dataset_key = str(entry.get("dataset_key", ""))
        tile_name = str(entry.get("tile_name", ""))

        source_shard = resolve_manifest_entry_path(spec.manifest_path, entry.get("shard_path", ""))
        if source_shard is None or not source_shard.exists():
            raise SystemExit(
                f"Manifest '{spec.manifest_path}' entry '{tile_name or '<unknown>'}' references a missing shard: {source_shard}"
            )

        bundled_shard = self._copy_file_once(
            self._copied_shards,
            spec.cache_label,
            source_shard,
            self._target_shard_path(spec, source_shard, dataset_key, tile_name),
        )

        portable_entry = dict(entry)
        portable_entry["shard_path"] = to_posix_relative(bundled_shard, manifest_output_path.parent)

        source_json = resolve_manifest_entry_path(spec.manifest_path, entry.get("source_json", ""))
        if self.include_source_json and source_json is not None:
            if not source_json.exists():
                raise SystemExit(
                    f"Manifest '{spec.manifest_path}' entry '{tile_name or '<unknown>'}' references a missing source_json: {source_json}"
                )
            bundled_source_json = self._copy_file_once(
                self._copied_source_json,
                spec.cache_label,
                source_json,
                self._target_source_json_path(spec, source_json, dataset_key, tile_name),
            )
            portable_entry["source_json"] = to_posix_relative(bundled_source_json, manifest_output_path.parent)
        else:
            portable_entry.pop("source_json", None)

        return portable_entry

    def bundle_manifest(self, spec: BundleManifestSpec) -> dict[str, Any]:
        manifest_payload = load_json(spec.manifest_path)
        output_path = self.manifests_dir / spec.output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        source_entries = manifest_payload.get("entries", [])
        if not isinstance(source_entries, list):
            raise SystemExit(f"Manifest '{spec.manifest_path}' does not contain a valid entries list.")
        if self.limit is not None:
            source_entries = source_entries[: self.limit]

        bundled_entries = [
            self._portable_manifest_entry(spec, output_path, entry)
            for entry in source_entries
        ]

        portable_manifest = dict(manifest_payload)
        portable_manifest["entries"] = bundled_entries
        portable_manifest["source_manifest"] = str(spec.manifest_path)
        portable_manifest["portable_bundle_relative_paths"] = True
        portable_manifest["portable_bundle_generated_at_utc"] = utc_now_iso()

        write_json(output_path, portable_manifest)
        self.validate_manifest(output_path)

        return {
            "label": spec.label,
            "source_manifest": str(spec.manifest_path),
            "bundle_manifest": str(output_path),
            "bundle_manifest_relative": to_posix_relative(output_path, self.bundle_root),
            "cache_label": spec.cache_label,
            "entries": len(bundled_entries),
            "source_json_included": self.include_source_json,
        }

    def validate_manifest(self, manifest_path: Path) -> None:
        payload = load_json(manifest_path)
        missing_paths: list[str] = []
        for entry in payload.get("entries", []):
            shard_path = resolve_manifest_entry_path(manifest_path, entry.get("shard_path", ""))
            if shard_path is None or not shard_path.exists():
                missing_paths.append(f"missing shard: {entry.get('tile_name', '<unknown>')} -> {shard_path}")

            source_json = entry.get("source_json")
            if source_json:
                source_json_path = resolve_manifest_entry_path(manifest_path, source_json)
                if source_json_path is None or not source_json_path.exists():
                    missing_paths.append(f"missing source_json: {entry.get('tile_name', '<unknown>')} -> {source_json_path}")

        if missing_paths:
            joined = "\n".join(missing_paths[:20])
            raise SystemExit(f"Portable manifest validation failed for {manifest_path}:\n{joined}")


def maybe_remove_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise SystemExit(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
    shutil.rmtree(output_dir)


def maybe_create_archive(bundle_root: Path, archive_format: str, overwrite: bool) -> str | None:
    archive_format = archive_format.lower()
    if archive_format == "none":
        return None

    extension = ".zip" if archive_format == "zip" else ".tar.gz"
    archive_path = bundle_root.parent / f"{bundle_root.name}{extension}"
    if archive_path.exists() and not overwrite:
        raise SystemExit(f"Archive already exists: {archive_path}. Use --overwrite to replace it.")
    if archive_path.exists():
        archive_path.unlink()

    if archive_format == "zip":
        created = shutil.make_archive(str(bundle_root), "zip", root_dir=bundle_root.parent, base_dir=bundle_root.name)
        return str(Path(created))
    if archive_format == "tar.gz":
        created = shutil.make_archive(str(bundle_root), "gztar", root_dir=bundle_root.parent, base_dir=bundle_root.name)
        return str(Path(created))

    raise SystemExit(f"Unsupported archive format: {archive_format}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a portable v9 training bundle with Linux-friendly relative manifests."
    )
    parser.add_argument("--train-manifest", required=True, help="Path to the training split v9 tensor cache manifest.")
    parser.add_argument("--dev-manifest", help="Optional path to the dev-eval or holdout v9 tensor cache manifest.")
    parser.add_argument("--output-dir", required=True, help="Bundle output directory, e.g. output/ml-training/v9_run_bundle.")
    parser.add_argument("--train-output-name", default=DEFAULT_TRAIN_OUTPUT_NAME, help=f"Bundle manifest filename for the training split. Default: {DEFAULT_TRAIN_OUTPUT_NAME}")
    parser.add_argument("--dev-output-name", default=DEFAULT_DEV_OUTPUT_NAME, help=f"Bundle manifest filename for the dev split. Default: {DEFAULT_DEV_OUTPUT_NAME}")
    parser.add_argument("--train-cache-label", default="main", help="Cache subdirectory label for the training split. Default: main")
    parser.add_argument("--dev-cache-label", default="dev", help="Cache subdirectory label for the dev split. Default: dev")
    parser.add_argument("--include-source-json", action=argparse.BooleanOptionalAction, default=False, help="Copy source_json payloads into the bundle. Default: disabled.")
    parser.add_argument("--archive-format", choices=["none", "zip", "tar.gz"], default="none", help="Optional archive to create after the bundle is written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-manifest entry cap for bounded smoke packaging.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing bundle directory or archive.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    train_manifest = Path(args.train_manifest).resolve()
    dev_manifest = Path(args.dev_manifest).resolve() if args.dev_manifest else None
    bundle_root = Path(args.output_dir).resolve()

    if not train_manifest.exists():
        raise SystemExit(f"Training manifest not found: {train_manifest}")
    if dev_manifest is not None and not dev_manifest.exists():
        raise SystemExit(f"Dev manifest not found: {dev_manifest}")

    maybe_remove_output_dir(bundle_root, overwrite=args.overwrite)
    bundle_root.mkdir(parents=True, exist_ok=True)

    builder = BundleBuilder(
        bundle_root=bundle_root,
        include_source_json=bool(args.include_source_json),
        limit=args.limit,
    )

    specs = [
        BundleManifestSpec(
            label="train",
            manifest_path=train_manifest,
            output_name=args.train_output_name,
            cache_label=args.train_cache_label,
        )
    ]
    if dev_manifest is not None:
        specs.append(
            BundleManifestSpec(
                label="dev",
                manifest_path=dev_manifest,
                output_name=args.dev_output_name,
                cache_label=args.dev_cache_label,
            )
        )

    manifest_results = [builder.bundle_manifest(spec) for spec in specs]

    archive_path = maybe_create_archive(bundle_root, args.archive_format, overwrite=args.overwrite)

    summary = {
        "schema_version": "v9-run-bundle-summary.v1",
        "created_at_utc": utc_now_iso(),
        "bundle_root": str(bundle_root),
        "bundle_root_relative": bundle_root.name,
        "include_source_json": bool(args.include_source_json),
        "archive_format": args.archive_format,
        "archive_path": archive_path,
        "copied_shards": len(builder._copied_shards),
        "copied_source_json": len(builder._copied_source_json),
        "manifests": manifest_results,
    }
    provenance = {
        "schema_version": "v9-run-bundle-sources.v1",
        "created_at_utc": utc_now_iso(),
        "bundle_root": str(bundle_root),
        "source_manifests": manifest_results,
    }

    write_json(builder.metadata_dir / "bundle_summary.json", summary)
    write_json(builder.metadata_dir / "source_manifests.json", provenance)

    print(f"Wrote portable v9 bundle: {bundle_root}")
    for manifest_result in manifest_results:
        print(
            f"  {manifest_result['label']}: {manifest_result['entries']} entries -> "
            f"{manifest_result['bundle_manifest_relative']}"
        )
    print(f"  copied shards: {summary['copied_shards']}")
    if args.include_source_json:
        print(f"  copied source_json: {summary['copied_source_json']}")
    if archive_path:
        print(f"  archive: {archive_path}")


if __name__ == "__main__":
    main()
