#!/usr/bin/env python3
"""Spec 109 T034: v50 dataset build commands -- migrate-v18, build, verify, finalize, curriculum.

Replaces the former fail-closed placeholder now that the fixture-proven migration, store-write,
and finalization owners exist (``harvester.v50.migrate``/``store``/``verify_store``/``curriculum``).
Nothing here fabricates trust: ``migrate-v18`` only copies signals that pass the V18 audit AND
whose policy is ``copy-if-verified`` (FR-006/FR-017); everything else is recorded
``unavailable`` with a reason, leaving the store ``finalization_state=incomplete`` until ``build``
fills the fresh-extraction gaps and ``finalize`` reconciles the whole thing.

``build`` is the one subcommand that can launch a real subprocess against a configured client root
-- it never does so without ``--confirm-run`` (see ``harvester.v50.build.run_fresh_extraction``),
matching the same execution-gate discipline already used in Spec 111's
``train_spec111_reconstruction.py``. Preparing/testing these tools does not authorize spending real
harvest time.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from harvester.v50.build import build_harvest_stream_command, run_fresh_extraction
from harvester.v50.contracts import (
    DatasetSignal,
    DatasetStoreManifest,
    FinalizationState,
    MigrationPolicy,
    RowLineage,
    UnavailableSignal,
)
from harvester.v50.curriculum import CurriculumRowRef, build_curriculum
from harvester.v50.identity import hash_array
from harvester.v50.migrate import ACTION_COPY, MigrationLedger, copy_signal_row, plan_signal_migration
from harvester.v50.store import finalize_store_report, write_v50_store
from harvester.v50.verify_store import verify_store
from harvester.v50.verify_v18 import V18RowSignalObservation, V18SignalSpec, audit_v18_signal


def _load_signal_specs(config_path: Path) -> list[V18SignalSpec]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return [
        V18SignalSpec(
            name=entry["name"],
            blacklisted=bool(entry.get("blacklisted", False)),
            blacklist_reason=str(entry.get("blacklist_reason", "")),
            has_flag_name=entry.get("has_flag_name"),
        )
        for entry in raw["signals"]
    ]


def _load_manifest(path: Path) -> DatasetStoreManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signals = tuple(
        DatasetSignal(
            name=s["name"],
            dtype=s["dtype"],
            row_shape=tuple(s["row_shape"]),
            required=s["required"],
            authoritative_source=s["authoritative_source"],
            content_identity=s["content_identity"],
            coverage_count=s["coverage_count"],
            migration_policy=MigrationPolicy(s["migration_policy"]),
        )
        for s in payload["signals"]
    )
    unavailable = tuple(UnavailableSignal(name=u["name"], reason=u["reason"]) for u in payload.get("unavailable_signals", []))
    return DatasetStoreManifest(
        release=payload["release"],
        store_id=payload["store_id"],
        build_id=payload["build_id"],
        producer_identity=payload["producer_identity"],
        client_build_evidence_id=payload["client_build_evidence_id"],
        index_identity=payload["index_identity"],
        row_count=payload["row_count"],
        signals=signals,
        row_lineage_identity=payload["row_lineage_identity"],
        finalization_state=FinalizationState(payload["finalization_state"]),
        unavailable_signals=unavailable,
    )


def _load_row_lineages(path: Path) -> list[RowLineage]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RowLineage(
            store_row=row["store_row"],
            build_id=row["build_id"],
            map_name=row["map_name"],
            tile_x=row["tile_x"],
            tile_y=row["tile_y"],
            source_group=row["source_group"],
            signal_actions=row["signal_actions"],
            signal_source_hashes=row.get("signal_source_hashes", {}),
            signal_destination_hashes=row.get("signal_destination_hashes", {}),
            source_artifact_id=row.get("source_artifact_id"),
            source_row=row.get("source_row"),
            split_group=row.get("split_group"),
        )
        for row in payload["rows"]
    ]


def reconcile_manifest_policy(
    manifest: DatasetStoreManifest,
    policy_template: DatasetStoreManifest,
    row_lineages: list[RowLineage],
) -> DatasetStoreManifest:
    """Apply only a current template's signal policy to a written-build manifest.

    Build manifests carry the content identities that were observed from a particular store, while
    the frozen template owns whether a signal is required.  This is deliberately a narrow repair:
    names, dtype, shape, source, and migration policy must already agree, and the function derives
    coverage only from explicit real row-lineage actions.  It therefore cannot relabel authored
    stream data as synthesis or fabricate coverage for zero-filled missing rows.
    """
    policy_by_name = {signal.name: signal for signal in policy_template.signals}
    manifest_names = {signal.name for signal in manifest.signals}
    policy_names = set(policy_by_name)
    if manifest_names != policy_names:
        raise ValueError(
            "policy template signals do not match the written manifest: "
            f"missing={sorted(manifest_names - policy_names)!r}, "
            f"unexpected={sorted(policy_names - manifest_names)!r}"
        )

    coverage_by_name = {
        signal.name: sum(
            lineage.signal_actions.get(signal.name) not in (None, "unavailable")
            for lineage in row_lineages
        )
        for signal in manifest.signals
    }
    reconciled_signals: list[DatasetSignal] = []
    for signal in manifest.signals:
        policy = policy_by_name[signal.name]
        if (
            signal.dtype != policy.dtype
            or signal.row_shape != policy.row_shape
            or signal.authoritative_source != policy.authoritative_source
            or signal.migration_policy != policy.migration_policy
        ):
            raise ValueError(
                f"policy template disagrees with written manifest for {signal.name!r}; "
                "refusing to relabel already-written data"
            )
        reconciled_signals.append(
            replace(
                signal,
                required=policy.required,
                coverage_count=coverage_by_name[signal.name],
            )
        )

    return replace(manifest, signals=tuple(reconciled_signals))


def _cmd_migrate_v18(args: argparse.Namespace) -> int:
    v18_store = Path(args.v18_store)
    root = zarr.open_group(str(v18_store), mode="r")
    index_rows = pq.read_table(str(v18_store / "index.parquet")).to_pylist()
    if args.sample is not None:
        index_rows = index_rows[: args.sample]

    specs = _load_signal_specs(Path(args.signals_config))
    manifest_template = _load_manifest(Path(args.manifest_template))

    ledger = MigrationLedger()
    copied_arrays: dict[str, list[np.ndarray]] = {}
    row_lineage_actions: list[dict[str, str]] = [dict() for _ in index_rows]
    unavailable: list[UnavailableSignal] = []
    per_signal_report = []

    for spec in specs:
        array = root[spec.name] if spec.name in root else None
        observations = [
            V18RowSignalObservation(
                row_id=i,
                value=(np.asarray(array[int(row["tile_id"])]) if array is not None else None),
                has_flag_value=(bool(row.get(spec.has_flag_name, False)) if spec.has_flag_name else None),
            )
            for i, row in enumerate(index_rows)
        ]
        audit = audit_v18_signal(spec, observations)
        plan = plan_signal_migration(audit)

        copy_eligible_rows = [row_id for row_id, action in plan.items() if action == ACTION_COPY]
        if copy_eligible_rows and array is not None:
            copied = []
            for row_id in copy_eligible_rows:
                copied_row, ledger = copy_signal_row(row_id, spec.name, np.asarray(array[row_id]), ledger)
                copied.append(copied_row)
                row_lineage_actions[row_id][spec.name] = "copied"
            copied_arrays[spec.name] = copied
        else:
            unavailable.append(
                UnavailableSignal(name=spec.name, reason="no rows passed audit as copy-if-verified; requires fresh extraction")
            )

        per_signal_report.append(
            {
                "signal_name": spec.name,
                "migration_policy": audit.migration_policy,
                "blacklisted": audit.blacklisted,
                "copy_eligible_row_count": len(copy_eligible_rows),
                "total_row_count": len(observations),
            }
        )

    report = {
        "schema": "v50-migration-report-v1",
        "v18_store": str(v18_store),
        "rows_processed": len(index_rows),
        "signals": per_signal_report,
        "ledger_entry_count": len(ledger.entries),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"migrate-v18: processed {len(index_rows)} rows, {len(ledger.entries)} bit-preserving copies -> {report_path}")

    if args.write_store:
        # Only fully-copied signals get a real array; everything else is honestly unavailable, and
        # finalization_state stays incomplete until `build` fills the fresh-extraction gaps.
        copyable_signals = tuple(
            signal for signal in manifest_template.signals
            if signal.name in copied_arrays and len(copied_arrays[signal.name]) == len(index_rows)
        )
        stacked_arrays = {name: np.stack(rows) for name, rows in copied_arrays.items()}
        recomputed_signals = tuple(
            DatasetSignal(
                name=signal.name,
                dtype=signal.dtype,
                row_shape=signal.row_shape,
                required=signal.required,
                authoritative_source=signal.authoritative_source,
                content_identity=hash_array(stacked_arrays[signal.name]),
                coverage_count=len(index_rows),
                migration_policy=signal.migration_policy,
            )
            for signal in copyable_signals
        )
        remaining_unavailable = tuple(unavailable) + tuple(
            UnavailableSignal(name=signal.name, reason="not fully copy-eligible across all rows in this pass")
            for signal in manifest_template.signals
            if signal.name not in {s.name for s in recomputed_signals}
        )
        output_manifest = DatasetStoreManifest(
            release=manifest_template.release,
            store_id=manifest_template.store_id,
            build_id=manifest_template.build_id,
            producer_identity=manifest_template.producer_identity,
            client_build_evidence_id=manifest_template.client_build_evidence_id,
            index_identity=manifest_template.index_identity,
            row_count=len(index_rows),
            signals=recomputed_signals or manifest_template.signals[:1],
            row_lineage_identity=manifest_template.row_lineage_identity,
            finalization_state=FinalizationState.INCOMPLETE,
            unavailable_signals=remaining_unavailable,
        )
        write_v50_store(Path(args.write_store), output_manifest, stacked_arrays)
        print(f"migrate-v18: wrote partial v50 store (finalization_state=incomplete) -> {args.write_store}")

        # Copy Parquet sidecar files and flat arrays if they exist in the source store
        import shutil
        v18_store_path = Path(v18_store)
        write_store_path = Path(args.write_store)
        for fn in ["index.parquet", "placements.parquet", "asset_inventory.parquet"]:
            src_file = v18_store_path / fn
            if src_file.exists():
                shutil.copy2(src_file, write_store_path / fn)
                print(f"migrate-v18: copied sidecar -> {write_store_path / fn}")

        # Copy flat arrays
        v18_group = zarr.open_group(str(v18_store_path), mode="r")
        store_group = zarr.open_group(str(write_store_path), mode="r+")
        flat_keys = [
            "mddf_placement_offset", "mddf_count", "mddf_placement_data", "mddf_unique_ids", "mddf_model_ids",
            "modf_placement_offset", "modf_count", "modf_placement_data", "modf_unique_ids", "modf_model_ids"
        ]
        for k in flat_keys:
            if k in v18_group:
                # If array already exists, delete it first to avoid collision
                if k in store_group:
                    del store_group[k]
                store_group.create_array(k, data=v18_group[k][:])
                print(f"migrate-v18: copied flat array -> {k}")

        print("Remaining unavailable signals require `build` (fresh extraction) before `finalize`.")

    return 0



def signal_takes_stream_data(signal: DatasetSignal) -> bool:
    """Spec 112 (FR-004/US1 edge case): a signal whose authoritative source is minimap synthesis
    must NEVER be filled from the harvest stream. The v22 stream carries the AUTHORED client
    minimap under the same ``minimap_rgb`` key, and letting it stand wherever synthesis skipped
    silently mixed authored imagery into the clean-room store labeled as fresh synthesis -- the
    actual mechanism behind the 256/1024 coverage gap (220 Kalimdor rows). A synthesis-sourced
    signal with no synthesized PNG is honestly unavailable for that tile, at both resolutions."""
    return signal.authoritative_source != "synthetic-minimap"


def build_synthetic_minimap_command(
    *,
    dll_path: Path,
    client_root: Path,
    map_name: str,
    resolution: int,
    output_dir: Path | None = None,
    detail_texels: bool = False,
) -> list[str]:
    """Build one synthesis invocation while keeping the 256/1024 render intents explicit.

    Spec 113 upgrades only ``minimap_rgb_1024`` to the mip-filtered real-texel detail mode. The
    256 signal remains the phase-independent material-average render used by Spec 112.
    """
    command = [
        "dotnet", str(dll_path),
        "synthetic-minimap", "--client-root", str(client_root),
        "--map", map_name,
        "--resolution", str(resolution),
        "--per-tile",
    ]
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    if detail_texels:
        command.append("--detail")
    return command


def _cmd_build(args: argparse.Namespace) -> int:
    import tempfile
    import subprocess
    from PIL import Image

    def adjust_to_manifest_shape(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        if array.shape == target_shape:
            return array
        res = np.zeros(target_shape, dtype=array.dtype)
        slices = tuple(slice(0, min(s_in, s_target)) for s_in, s_target in zip(array.shape, target_shape))
        res[slices] = array[slices]
        return res

    write_store_path = Path(args.write_store) if args.write_store else None
    manifest_template_path = Path(args.manifest_template) if args.manifest_template else None
    report_path = Path(args.report) if args.report else None


    # Load manifest template first to get build_id for client root resolution
    if manifest_template_path:
        manifest_template = _load_manifest(manifest_template_path)
        build_id = manifest_template.build_id
    else:
        manifest_template = None
        build_id = None

    # Resolve clients_root library path to build-specific client root directory
    clients_root = Path(args.clients_root)
    resolved_client_root = clients_root
    if build_id:
        variants = [build_id, build_id.replace('.', '_'), build_id.replace('_', '.')]
        found = False
        for variant in variants:
            p = clients_root / variant
            if p.exists():
                resolved_client_root = p
                found = True
                break
        
        if not found and clients_root.exists():
            for p in clients_root.iterdir():
                if p.is_dir():
                    for variant in variants:
                        if variant in p.name:
                            resolved_client_root = p
                            found = True
                            break
                    if found:
                        break

    print(f"Resolved client root: {resolved_client_root}", flush=True)

    # Ensure harvest project is compiled first
    proj_path = Path(args.harvest_project)
    proj_file = proj_path if proj_path.suffix == ".csproj" else proj_path / "WowViewer.Tool.Harvest.csproj"
    print(f"Building C# project {proj_file}...", flush=True)
    res_build = subprocess.run(["dotnet", "build", str(proj_file), "-c", "Debug"], capture_output=True)
    if res_build.returncode != 0:
        print(res_build.stdout.decode('utf-8', errors='ignore'))
        print(res_build.stderr.decode('utf-8', errors='ignore'))
        print("Error: C# harvester compilation failed.")
        return 1

    from harvester.v50.build import find_harvest_dll
    dll_path = find_harvest_dll(proj_path)

    resolutions = [256, 1024]

    command = build_harvest_stream_command(
        proj_path,
        client_root=resolved_client_root,
        map_name=args.map,
        stream_profile=args.stream_profile,
    )

    if not args.confirm_run:
        print(f"Would run fresh extraction: {' '.join(command)}")
        for res in resolutions:
            synth_cmd = build_synthetic_minimap_command(
                dll_path=dll_path,
                client_root=resolved_client_root,
                map_name=args.map,
                resolution=res,
                detail_texels=res == 1024,
            )
            print(f"Would run synthetic minimap ({res}x{res}): {' '.join(synth_cmd)}")
        print("NOT executing: pass confirm_run=True only with explicit user authorization for this run.")
        return 0

    if not write_store_path or not manifest_template_path or not report_path or not manifest_template:
        print("Error: --write-store, --manifest-template, and --report are required when --confirm-run is passed.")
        return 1

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Parallelize the generation of the 256 and 1024 minimaps
        synth_processes = []
        for res in resolutions:
            synth_out_dir = temp_path / f"minimap_{res}"
            synth_cmd = build_synthetic_minimap_command(
                dll_path=dll_path,
                client_root=resolved_client_root,
                map_name=args.map,
                resolution=res,
                output_dir=synth_out_dir,
                detail_texels=res == 1024,
            )
            render_mode = "detail" if res == 1024 else "material-average"
            print(f"Launching synthesized {res}x{res} minimaps generation ({render_mode})...", flush=True)
            p = subprocess.Popen(synth_cmd)
            synth_processes.append((res, p))

        # Wait for both minimap processes to complete
        synthesis_manifests: dict[int, dict[str, object]] = {}
        for res, p in synth_processes:
            res_code = p.wait()
            if res_code != 0:
                print(f"Warning: synthetic-minimap resolution {res} exited with code {res_code}")
                continue
            synthesis_manifest_path = temp_path / f"minimap_{res}" / "synthesis-manifest.json"
            if not synthesis_manifest_path.exists():
                raise RuntimeError(
                    f"synthetic-minimap {res} succeeded but wrote no synthesis manifest: "
                    f"{synthesis_manifest_path}"
                )
            synthesis_manifests[res] = json.loads(
                synthesis_manifest_path.read_text(encoding="utf-8")
            )

        detail_manifest = synthesis_manifests.get(1024)
        if detail_manifest is not None and detail_manifest.get("RenderMode") != "detail":
            raise RuntimeError(
                "1024 minimap synthesis did not report RenderMode=detail; refusing to label the "
                "result as Spec 113 HR"
            )

        print(f"Launching harvest-stream...", flush=True)
        process = run_fresh_extraction(command, confirm_run=True)
        if process is None or process.stdout is None:
            raise RuntimeError("Failed to launch fresh extraction subprocess.")


        from harvester.v50.build import read_harvest_stream

        accumulated_signals: dict[str, list[np.ndarray]] = {}
        row_lineages: list[dict[str, Any]] = []
        row_id = 0

        # sidecar Parquet data lists
        index_rows: list[dict[str, Any]] = []
        placements_rows: list[dict[str, Any]] = []
        models_dict: dict[str, str] = {}
        textures_set: set[str] = set()

        global_textures: list[str] = []
        global_textures_to_idx: dict[str, int] = {}

        try:
            for tile_data in read_harvest_stream(process.stdout):
                if args.sample is not None and row_id >= args.sample:
                    break

                meta = tile_data["_metadata"]
                tile_x = int(meta["tile_x"])
                tile_y = int(meta["tile_y"])
                map_name = str(meta["map_name"])
                build_id = str(meta.get("build_key") or meta.get("build") or "")

                # 1. Accumulate tile metadata for index_rows
                mtex_paths = meta.get("mtex_texture_paths") or meta.get("mcly_texture_names") or []
                mddf_paths = meta.get("placement_mddf_asset_paths") or meta.get("placement_mddf_names") or []
                modf_paths = meta.get("placement_modf_asset_paths") or meta.get("placement_modf_names") or []

                index_rows.append({
                    "tile_id": row_id,
                    "build": build_id,
                    "map": map_name,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "mtex_texture_paths": mtex_paths,
                    "placement_mddf_asset_paths": mddf_paths,
                    "placement_modf_asset_paths": modf_paths,
                })

                # 2. Parse mddf placements
                mddf_data = tile_data.get("mddf_placement_data")
                if mddf_data is not None and mddf_data.size > 0:
                    for i in range(mddf_data.shape[0]):
                        name_idx = int(mddf_data[i, 0])
                        asset_path = mddf_paths[name_idx] if name_idx < len(mddf_paths) else ""
                        placements_rows.append({
                            "tile_id": row_id,
                            "instance_type": "mddf",
                            "instance_idx": i,
                            "asset_path": asset_path,
                            "nameId": name_idx,
                            "uniqueId": int(mddf_data[i, 1]),
                            "posX": float(mddf_data[i, 2]),
                            "posY": float(mddf_data[i, 3]),
                            "posZ": float(mddf_data[i, 4]),
                            "rotX": float(mddf_data[i, 5]),
                            "rotY": float(mddf_data[i, 6]),
                            "rotZ": float(mddf_data[i, 7]),
                            "scale": float(mddf_data[i, 8]),
                            "bbMinX": 0.0,
                            "bbMinY": 0.0,
                            "bbMinZ": 0.0,
                            "bbMaxX": 0.0,
                            "bbMaxY": 0.0,
                            "bbMaxZ": 0.0,
                        })
                        if asset_path:
                            models_dict[asset_path] = "m2"

                # 3. Parse modf placements
                modf_data = tile_data.get("modf_placement_data")
                if modf_data is not None and modf_data.size > 0:
                    for i in range(modf_data.shape[0]):
                        name_idx = int(modf_data[i, 0])
                        asset_path = modf_paths[name_idx] if name_idx < len(modf_paths) else ""
                        placements_rows.append({
                            "tile_id": row_id,
                            "instance_type": "modf",
                            "instance_idx": i,
                            "asset_path": asset_path,
                            "nameId": name_idx,
                            "uniqueId": int(modf_data[i, 1]),
                            "posX": float(modf_data[i, 2]),
                            "posY": float(modf_data[i, 3]),
                            "posZ": float(modf_data[i, 4]),
                            "rotX": float(modf_data[i, 5]),
                            "rotY": float(modf_data[i, 6]),
                            "rotZ": float(modf_data[i, 7]),
                            "scale": float(modf_data[i, 8]),
                            "bbMinX": float(modf_data[i, 11]),
                            "bbMinY": float(modf_data[i, 12]),
                            "bbMinZ": float(modf_data[i, 13]),
                            "bbMaxX": float(modf_data[i, 14]),
                            "bbMaxY": float(modf_data[i, 15]),
                            "bbMaxZ": float(modf_data[i, 16]),
                        })
                        if asset_path:
                            models_dict[asset_path] = "wmo"

                # 4. Accumulate textures and build mcly_tileset_ids
                for tex in mtex_paths:
                    textures_set.add(tex)
                    if tex not in global_textures_to_idx:
                        global_textures_to_idx[tex] = len(global_textures)
                        global_textures.append(tex)

                mcly_tileset_ids = np.full((16, 16, 4), -1, dtype=np.int32)
                local_texture_ids = tile_data.get("mcly_texture_ids")
                local_layer_mask = tile_data.get("mcly_layer_mask")
                if local_texture_ids is not None and local_layer_mask is not None:
                    local_texture_ids_arr = np.asarray(local_texture_ids, dtype=np.int32)
                    local_layer_mask_arr = np.asarray(local_layer_mask, dtype=np.float32)
                    for cy in range(16):
                        for cx in range(16):
                            for layer in range(4):
                                if local_layer_mask_arr[cy, cx, layer] <= 0.0:
                                    continue
                                texture_id = int(local_texture_ids_arr[cy, cx, layer])
                                if 0 <= texture_id < len(mtex_paths):
                                    tex_path = mtex_paths[texture_id]
                                    mcly_tileset_ids[cy, cx, layer] = global_textures_to_idx.get(tex_path, -1)
                tile_data["mcly_tileset_ids"] = mcly_tileset_ids

                # Spec 112 (user-directed): the v22 stream decodes the AUTHORED client minimap under
                # the key `minimap_rgb`. Capture it under `minimap_rgb_authored` (its own honest,
                # harvest-stream-sourced signal) BEFORE the synthesis override below replaces
                # `minimap_rgb` with the compositor's synthesized render. Where the client shipped no
                # minimap BLP the stream omits the key, so the signal is honestly unavailable for
                # that tile -- never zero-substituted. This is the real deployment input the height
                # model trains on alongside synthetic (as separate curriculum rows).
                if "minimap_rgb" in tile_data:
                    tile_data["minimap_rgb_authored"] = tile_data["minimap_rgb"]

                lineage_actions = {}
                lineage_src_hashes = {}
                lineage_dst_hashes = {}

                for signal in manifest_template.signals:
                    signal_name = signal.name
                    if signal_name in tile_data and signal_takes_stream_data(signal):
                        array = tile_data[signal_name]
                        array = adjust_to_manifest_shape(array, tuple(signal.row_shape))
                        content_hash = hash_array(array)
                        lineage_actions[signal_name] = "freshly_extracted"
                        lineage_src_hashes[signal_name] = content_hash
                        lineage_dst_hashes[signal_name] = content_hash
                    else:
                        np_dtype = np.dtype(signal.dtype)
                        array = np.zeros(signal.row_shape, dtype=np_dtype)
                        content_hash = hash_array(array)
                        lineage_actions[signal_name] = "unavailable"
                        lineage_src_hashes[signal_name] = content_hash
                        lineage_dst_hashes[signal_name] = content_hash

                    if signal_name not in accumulated_signals:
                        accumulated_signals[signal_name] = []
                    accumulated_signals[signal_name].append(array)

                # Override/inject synthesized minimaps
                m256_path = temp_path / "minimap_256" / "tiles" / f"{map_name}_{tile_x:02d}_{tile_y:02d}_synthesized.png"
                if m256_path.exists():
                    img_256 = np.array(Image.open(m256_path).convert("RGB"), dtype=np.uint8)
                    if "minimap_rgb" not in accumulated_signals:
                        accumulated_signals["minimap_rgb"] = []
                    if len(accumulated_signals["minimap_rgb"]) > row_id:
                        accumulated_signals["minimap_rgb"][row_id] = img_256
                    else:
                        accumulated_signals["minimap_rgb"].append(img_256)
                    
                    mhash = hash_array(img_256)
                    lineage_actions["minimap_rgb"] = "freshly_extracted"
                    lineage_src_hashes["minimap_rgb"] = mhash
                    lineage_dst_hashes["minimap_rgb"] = mhash

                m1024_path = temp_path / "minimap_1024" / "tiles" / f"{map_name}_{tile_x:02d}_{tile_y:02d}_synthesized.png"
                if m1024_path.exists():
                    img_1024 = np.array(Image.open(m1024_path).convert("RGB"), dtype=np.uint8)
                    if "minimap_rgb_1024" not in accumulated_signals:
                        accumulated_signals["minimap_rgb_1024"] = []
                    if len(accumulated_signals["minimap_rgb_1024"]) > row_id:
                        accumulated_signals["minimap_rgb_1024"][row_id] = img_1024
                    else:
                        accumulated_signals["minimap_rgb_1024"].append(img_1024)
                    
                    mhash = hash_array(img_1024)
                    lineage_actions["minimap_rgb_1024"] = "freshly_extracted"
                    lineage_src_hashes["minimap_rgb_1024"] = mhash
                    lineage_dst_hashes["minimap_rgb_1024"] = mhash


                row_lineages.append({
                    "store_row": row_id,
                    "build_id": build_id,
                    "map_name": map_name,
                    "tile_x": tile_x,
                    "tile_y": tile_y,

                    "source_group": args.map,
                    "signal_actions": lineage_actions,
                    "signal_source_hashes": lineage_src_hashes,
                    "signal_destination_hashes": lineage_dst_hashes,
                    "source_artifact_id": None,
                    "source_row": None,
                    "split_group": None,
                })
                row_id += 1

        finally:
            process.stdout.close()
            process.wait()

        if row_id == 0:
            print("Error: zero tiles harvested.")
            return 1

        stacked_arrays = {name: np.stack(lst) for name, lst in accumulated_signals.items()}

        coverage_by_signal = {
            signal.name: sum(
                lineage["signal_actions"].get(signal.name) not in (None, "unavailable")
                for lineage in row_lineages
            )
            for signal in manifest_template.signals
        }
        recomputed_signals = []
        for signal in manifest_template.signals:
            if signal.name in stacked_arrays:
                arr = stacked_arrays[signal.name]
                recomputed_signals.append(
                    DatasetSignal(
                        name=signal.name,
                        dtype=str(arr.dtype),
                        row_shape=arr.shape[1:],
                        required=signal.required,
                        authoritative_source=signal.authoritative_source,
                        content_identity=hash_array(arr),
                        coverage_count=coverage_by_signal[signal.name],
                        migration_policy=signal.migration_policy,
                    )
                )

        unavailable_signals = [
            UnavailableSignal(name=signal.name, reason="missing in stream or synthesized files")
            for signal in manifest_template.signals
            if signal.name not in stacked_arrays
        ]
        # Spec 112: carry the template's era-unavailability records (e.g. mccv_rgb on 0.5.3) into
        # the built store's manifest -- the reason must survive into store attrs, not vanish here.
        unavailable_signals.extend(manifest_template.unavailable_signals)

        output_manifest = DatasetStoreManifest(
            release=manifest_template.release,
            store_id=manifest_template.store_id,
            build_id=manifest_template.build_id,
            producer_identity=manifest_template.producer_identity,
            client_build_evidence_id=manifest_template.client_build_evidence_id,
            index_identity=manifest_template.index_identity,
            row_count=row_id,
            signals=tuple(recomputed_signals),
            row_lineage_identity=manifest_template.row_lineage_identity,
            finalization_state=FinalizationState.INCOMPLETE,
            unavailable_signals=tuple(unavailable_signals),
        )

        write_v50_store(write_store_path, output_manifest, stacked_arrays)
        print(f"build: wrote partial v50 store (finalization_state=incomplete) -> {write_store_path}")

        if args.write_manifest:
            manifest_out_path = Path(args.write_manifest)
            manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_out_path.write_text(json.dumps(output_manifest.to_dict(), indent=2), encoding="utf-8")
            print(f"build: wrote real manifest (row_count={row_id}) -> {manifest_out_path}")

        # Compute and write flat placement arrays directly to the Zarr group
        mddf_counts = []
        modf_counts = []
        mddf_offsets = []
        modf_offsets = []
        mddf_data_list = []
        mddf_uids_list = []
        mddf_mids_list = []
        modf_data_list = []
        modf_uids_list = []
        modf_mids_list = []

        mddf_cursor = 0
        modf_cursor = 0

        from collections import defaultdict
        placements_by_tile = defaultdict(list)
        for p in placements_rows:
            placements_by_tile[p["tile_id"]].append(p)

        sorted_models_keys = sorted(models_dict.keys())

        for r_id in range(row_id):
            tile_placements = placements_by_tile[r_id]
            mddf_tile = [p for p in tile_placements if p["instance_type"] == "mddf"]
            modf_tile = [p for p in tile_placements if p["instance_type"] == "modf"]

            n_mddf = len(mddf_tile)
            n_modf = len(modf_tile)

            mddf_counts.append(n_mddf)
            modf_counts.append(n_modf)
            mddf_offsets.append(mddf_cursor)
            modf_offsets.append(modf_cursor)

            mddf_cursor += n_mddf
            modf_cursor += n_modf

            for p in mddf_tile:
                mddf_data_list.append([
                    p["nameId"], p["uniqueId"],
                    p["posX"], p["posY"], p["posZ"],
                    p["rotX"], p["rotY"], p["rotZ"],
                    p["scale"]
                ])
                mddf_uids_list.append(p["uniqueId"])
                model_path = p["asset_path"]
                mids_idx = sorted_models_keys.index(model_path) if model_path in models_dict else -1
                mddf_mids_list.append(mids_idx)

            for p in modf_tile:
                modf_data_list.append([
                    p["nameId"], p["uniqueId"],
                    p["posX"], p["posY"], p["posZ"],
                    p["rotX"], p["rotY"], p["rotZ"],
                    p["scale"],
                    0.0, 0.0, # flags, lgtFlags
                    p["bbMinX"], p["bbMinY"], p["bbMinZ"],
                    p["bbMaxX"], p["bbMaxY"], p["bbMaxZ"]
                ])
                modf_uids_list.append(p["uniqueId"])
                model_path = p["asset_path"]
                mids_idx = sorted_models_keys.index(model_path) if model_path in models_dict else -1
                modf_mids_list.append(mids_idx)

        store_root = zarr.open_group(str(write_store_path), mode="r+")
        if detail_manifest is not None:
            store_root.attrs.update(
                {
                    "minimap_rgb_1024_render_mode": str(detail_manifest["RenderMode"]),
                    "minimap_rgb_1024_texture_repeats_per_chunk": float(
                        detail_manifest["TextureRepeatsPerChunk"]
                    ),
                    "minimap_rgb_1024_synthesis_format": str(detail_manifest["Format"]),
                }
            )
        store_root.create_array("mddf_placement_offset", data=np.array(mddf_offsets, dtype=np.int64))
        store_root.create_array("mddf_count", data=np.array(mddf_counts, dtype=np.int32))

        mddf_data_arr = np.array(mddf_data_list, dtype=np.float32) if mddf_data_list else np.zeros((0, 9), dtype=np.float32)
        store_root.create_array("mddf_placement_data", data=mddf_data_arr)

        mddf_uids_arr = np.array(mddf_uids_list, dtype=np.int32) if mddf_uids_list else np.array([], dtype=np.int32)
        store_root.create_array("mddf_unique_ids", data=mddf_uids_arr)

        mddf_mids_arr = np.array(mddf_mids_list, dtype=np.int32) if mddf_mids_list else np.array([], dtype=np.int32)
        store_root.create_array("mddf_model_ids", data=mddf_mids_arr)

        store_root.create_array("modf_placement_offset", data=np.array(modf_offsets, dtype=np.int64))
        store_root.create_array("modf_count", data=np.array(modf_counts, dtype=np.int32))

        modf_data_arr = np.array(modf_data_list, dtype=np.float32) if modf_data_list else np.zeros((0, 17), dtype=np.float32)
        store_root.create_array("modf_placement_data", data=modf_data_arr)

        modf_uids_arr = np.array(modf_uids_list, dtype=np.int32) if modf_uids_list else np.array([], dtype=np.int32)
        store_root.create_array("modf_unique_ids", data=modf_uids_arr)

        modf_mids_arr = np.array(modf_mids_list, dtype=np.int32) if modf_mids_list else np.array([], dtype=np.int32)
        store_root.create_array("modf_model_ids", data=modf_mids_arr)


        # Write Parquet sidecar files
        import pyarrow as pa
        write_store_path_obj = Path(write_store_path)

        # 1. Write index.parquet
        if index_rows:
            index_table = pa.Table.from_pylist(index_rows)
        else:
            index_table = pa.Table.from_pydict({
                "tile_id": pa.array([], type=pa.int64()),
                "build": pa.array([], type=pa.string()),
                "map": pa.array([], type=pa.string()),
                "tile_x": pa.array([], type=pa.int32()),
                "tile_y": pa.array([], type=pa.int32()),
                "mtex_texture_paths": pa.array([], type=pa.list_(pa.string())),
                "placement_mddf_asset_paths": pa.array([], type=pa.list_(pa.string())),
                "placement_modf_asset_paths": pa.array([], type=pa.list_(pa.string())),
            })
        pq.write_table(index_table, str(write_store_path_obj / "index.parquet"))
        print(f"build: wrote index sidecar -> {write_store_path_obj / 'index.parquet'}")

        # 2. Write placements.parquet
        if placements_rows:
            placements_table = pa.Table.from_pylist(placements_rows)
        else:
            placements_table = pa.Table.from_pydict({
                "tile_id": pa.array([], type=pa.int64()),
                "instance_type": pa.array([], type=pa.string()),
                "instance_idx": pa.array([], type=pa.int32()),
                "asset_path": pa.array([], type=pa.string()),
                "nameId": pa.array([], type=pa.int32()),
                "uniqueId": pa.array([], type=pa.int32()),
                "posX": pa.array([], type=pa.float64()),
                "posY": pa.array([], type=pa.float64()),
                "posZ": pa.array([], type=pa.float64()),
                "rotX": pa.array([], type=pa.float64()),
                "rotY": pa.array([], type=pa.float64()),
                "rotZ": pa.array([], type=pa.float64()),
                "scale": pa.array([], type=pa.float64()),
                "bbMinX": pa.array([], type=pa.float64()),
                "bbMinY": pa.array([], type=pa.float64()),
                "bbMinZ": pa.array([], type=pa.float64()),
                "bbMaxX": pa.array([], type=pa.float64()),
                "bbMaxY": pa.array([], type=pa.float64()),
                "bbMaxZ": pa.array([], type=pa.float64()),
            })
        pq.write_table(placements_table, str(write_store_path_obj / "placements.parquet"))
        print(f"build: wrote placements sidecar -> {write_store_path_obj / 'placements.parquet'}")

        # 3. Write asset_inventory.parquet
        inventory_rows = []
        for path, kind in sorted(models_dict.items()):
            inventory_rows.append({
                "asset_path": path,
                "source_path": path,
                "source_in_listfile": 1,
                "source_kind": "archive_listed",
                "kind": kind,
                "load_error": 0,
            })
        for path in sorted(textures_set):
            inventory_rows.append({
                "asset_path": path,
                "source_path": path,
                "source_in_listfile": 1,
                "source_kind": "archive_listed",
                "kind": "texture_rgb",
                "load_error": 0,
            })

        if inventory_rows:
            inventory_table = pa.Table.from_pylist(inventory_rows)
        else:
            inventory_table = pa.Table.from_pydict({
                "asset_path": pa.array([], type=pa.string()),
                "source_path": pa.array([], type=pa.string()),
                "source_in_listfile": pa.array([], type=pa.int32()),
                "source_kind": pa.array([], type=pa.string()),
                "kind": pa.array([], type=pa.string()),
                "load_error": pa.array([], type=pa.int32()),
            })
        pq.write_table(inventory_table, str(write_store_path_obj / "asset_inventory.parquet"))
        print(f"build: wrote asset_inventory sidecar -> {write_store_path_obj / 'asset_inventory.parquet'}")

        report_payload = {"rows": row_lineages}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"build: wrote lineages -> {report_path}")

    return 0



def _cmd_verify(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    row_lineages = _load_row_lineages(Path(args.row_lineages))
    observed_hashes = json.loads(Path(args.observed_hashes).read_text(encoding="utf-8"))

    result = verify_store(manifest, row_lineages, observed_signal_hashes=observed_hashes)
    payload = {
        "store_id": result.store_id,
        "checks_performed": list(result.checks_performed),
        "passed": result.passed,
        "failure_reasons": list(result.failure_reasons),
        "proof_level": result.proof_level.value,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


def _cmd_finalize(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    row_lineages = _load_row_lineages(Path(args.row_lineages))
    if args.policy_template:
        policy_template = _load_manifest(Path(args.policy_template))
        manifest = reconcile_manifest_policy(manifest, policy_template, row_lineages)

    report = finalize_store_report(Path(args.store), manifest, row_lineages)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report.manifest.to_dict(), indent=2), encoding="utf-8")
    print(f"finalize: finalization_state={report.manifest.finalization_state.value} -> {output_path}")
    if report.mismatches:
        print(f"finalize: {len(report.mismatches)} reason(s) this store is not complete:")
        for reason in report.mismatches:
            print(f"  - {reason}")
    return 0 if report.manifest.finalization_state is FinalizationState.COMPLETE else 1


def _cmd_curriculum(args: argparse.Namespace) -> int:
    rows_payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = [
        CurriculumRowRef(store_id=r["store_id"], row_id=r["row_id"], source_group=r["source_group"], split=r["split"])
        for r in rows_payload["rows"]
    ]
    manifest = build_curriculum(
        release=args.release,
        rows=rows,
        selection_reason=args.selection_reason,
        policy_identity=args.policy_identity,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(f"curriculum: {len(rows)} row references -> {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate_parser = subparsers.add_parser("migrate-v18", help="bit-preserving copy of verified V18 signals")
    migrate_parser.add_argument("--v18-store", required=True)
    migrate_parser.add_argument("--signals-config", required=True)
    migrate_parser.add_argument("--manifest-template", required=True)
    migrate_parser.add_argument("--sample", type=int, default=None)
    migrate_parser.add_argument("--report", required=True)
    migrate_parser.add_argument("--write-store", default=None, help="also write a (likely partial) v50 store here")
    migrate_parser.set_defaults(handler=_cmd_migrate_v18)

    build_parser = subparsers.add_parser("build", help="fresh client-backed extraction (execution-gated)")
    build_parser.add_argument("--harvest-project", required=True)
    build_parser.add_argument("--clients-root", required=True)
    build_parser.add_argument("--map", required=True)
    build_parser.add_argument("--stream-profile", default="v22")
    build_parser.add_argument("--confirm-run", action="store_true",
                               help="actually launch the C# harvester; omit to only print the command")
    build_parser.add_argument("--write-store", default=None, help="write v50 Zarr store here")
    build_parser.add_argument("--manifest-template", default=None, help="metadata template json")
    build_parser.add_argument("--report", default=None, help="lineage report json output")
    build_parser.add_argument(
        "--write-manifest", default=None,
        help="also write the real recomputed manifest (row_count and content hashes actually "
             "written) here; pass this file, not --manifest-template, as `finalize`'s --manifest",
    )
    build_parser.add_argument("--signals-config", default=None, help="optional signals config json")
    build_parser.add_argument("--sample", type=int, default=None, help="limit tiles processed")
    build_parser.set_defaults(handler=_cmd_build)

    verify_parser = subparsers.add_parser("verify", help="run the FR-005 promotion gate against a manifest")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--row-lineages", required=True)
    verify_parser.add_argument("--observed-hashes", required=True)
    verify_parser.set_defaults(handler=_cmd_verify)

    finalize_parser = subparsers.add_parser("finalize", help="recompute observed hashes and finalize a written store")
    finalize_parser.add_argument("--store", required=True)
    finalize_parser.add_argument("--manifest", required=True)
    finalize_parser.add_argument("--row-lineages", required=True)
    finalize_parser.add_argument(
        "--policy-template",
        default=None,
        help="optional frozen template used to reconcile required flags and observed coverage before finalization",
    )
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.set_defaults(handler=_cmd_finalize)

    curriculum_parser = subparsers.add_parser("curriculum", help="build an immutable row-selection curriculum manifest")
    curriculum_parser.add_argument("--release", required=True)
    curriculum_parser.add_argument("--rows", required=True, help="JSON {rows: [{store_id, row_id, source_group, split}, ...]}")
    curriculum_parser.add_argument("--selection-reason", required=True)
    curriculum_parser.add_argument("--policy-identity", required=True)
    curriculum_parser.add_argument("--output", required=True)
    curriculum_parser.set_defaults(handler=_cmd_curriculum)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
