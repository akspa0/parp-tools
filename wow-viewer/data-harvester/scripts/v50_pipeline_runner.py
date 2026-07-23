#!/usr/bin/env python3
"""Unified pipeline runner to build, finalize, and pre-curate the v50 dataset.
Sequentially processes Kalimdor, Azeroth, PVPZone02, and Kalidar from the client library,
finalizes their manifests, and runs the Spec 103 curation to pre-bucket and filter out dirty tiles.

A map's `finalize` step legitimately comes back `finalization_state=incomplete` whenever any real
client tile lacks a required signal (e.g. a tile with terrain but no texture data to synthesize a
minimap from, so `minimap_rgb` -- required -- has no real lineage action for that row). That is not
a build failure: the store itself is real and mostly good, and the very next step (Spec 103
curation) exists specifically to drop exactly those dirty rows. So a non-complete `finalize` does
not stop this script -- it prints every reason (`finalize` now reports them, not just the bare
state), still runs curation for that map, and continues to the next map. Only a `build` failure
(the extraction subprocess itself erroring out) skips the rest of that map's steps; other maps still
run. See Spec 109 Phase 9 for the incident this fixes: a single dirty tile out of 685 previously
aborted the entire multi-map run via an unconditional `check=True`.

Each map gets two curation manifests, not one: the strict manifest (`curation-<build>-<Map>/`,
`--max-object-coverage 0.0`) is correct for minimap-to-height reconstruction specifically -- an
object occludes the ground, making "true height under it" an impossible target from the minimap
alone -- but is not a general data-quality filter, and dropping every object-touched tile from the
only curated view would silently discard real, wanted data for anything object-aware (v50 keeps
`object_precise_mask`/`object_instance_mask` as first-class signals for exactly that reason). The
object-inclusive manifest (`curation-<build>-<Map>-object-inclusive/`) applies the same
missing_signal/blank_minimap/height_normal_mismatch checks but leaves object tiles in. Neither
manifest duplicates array data -- both are Parquet row-reference lists over the same raw store.
"""

import argparse
import subprocess
import sys
from pathlib import Path

MAPS = ["Kalimdor", "Azeroth", "PVPZone02", "Kalidar"]

ESTIMATED_TIMES = {
    "Kalimdor": "8 to 12 minutes",
    "Azeroth": "5 to 8 minutes",
    "PVPZone02": "less than 30 seconds",
    "Kalidar": "less than 30 seconds"
}


def run_command(cmd: list[str], dry_run: bool, check: bool = True) -> int:
    print(f"\nExecuting: {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="v50 Dataset Extraction and Curation Pipeline")
    parser.add_argument("--confirm", action="store_true", help="actually execute the pipeline commands")
    parser.add_argument("--sample", type=int, default=None, help="limit tiles processed per map (for quick smoke tests)")
    parser.add_argument(
        "--stream-profile", default="v22",
        help="C# harvest-stream profile. 'v22' (default) is the PROVEN v50 profile: its array names "
             "(normal_xyz, minimap_rgb) match the v50 catalog, and as of Spec 118 it ALSO emits the "
             "strict object-geometry arrays (object_geometry_visible_mask/source/instance_257). Do "
             "NOT use 'full' for a v50 build: it renames core signals (normal_xyz->mcnr_normal_xyz, "
             "minimap_rgb->minimap_rgb_256), which the catalog's exact-name matcher does not select, "
             "so it silently zero-fills normals and the authored minimap.",
    )
    args = parser.parse_args()

    dry_run = not args.confirm

    if dry_run:
        print("=== DRY RUN MODE ===")
        print("The pipeline will process the following maps sequentially:")
        for m in MAPS:
            print(f"  - {m} (estimated time: {ESTIMATED_TIMES[m]})")
        print("\nSpecify --confirm to start execution.")

    # 1. Ensure directory structures exist
    reports_dir = Path("../output/reports/v50/v50.1")
    datasets_dir = Path("../output/datasets/v50/v50.1")

    if not dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)
        datasets_dir.mkdir(parents=True, exist_ok=True)

    # Per-map outcome, reported as a summary at the end so a long run's result isn't buried in
    # scrollback: "build" is only ever ok/failed; "finalize" and "curate" are None until attempted.
    outcomes: dict[str, dict[str, object]] = {}

    for map_name in MAPS:
        print(f"\n==============================================")
        print(f" Processing Map: {map_name}")
        print(f"==============================================")

        outcomes[map_name] = {"build": None, "finalize": None, "curate": None, "curate_object_inclusive": None}

        store_path = datasets_dir / f"0_5_3_3368-{map_name}.zarr"
        build_manifest_path = reports_dir / f"build-manifest-0_5_3_3368-{map_name}.json"
        manifest_path = datasets_dir / f"0_5_3_3368-{map_name}.manifest.json"
        report_path = reports_dir / f"build-0_5_3_3368-{map_name}.json"
        curation_path = datasets_dir / f"curation-0_5_3_3368-{map_name}"
        curation_path_object_inclusive = datasets_dir / f"curation-0_5_3_3368-{map_name}-object-inclusive"

        # Step A: Build/Extract. A real failure here (the harvester subprocess itself erroring out)
        # means there is no store to finalize or curate -- skip the rest of this map, but keep
        # going with the remaining maps rather than aborting the whole run.
        build_cmd = [
            "uv", "run", "python", "scripts/v50_build_dataset.py", "build",
            "--harvest-project", "../tools/harvest/WowViewer.Tool.Harvest",
            "--clients-root", "H:\\CLIENTS",
            "--map", map_name,
            "--stream-profile", args.stream_profile,
            "--signals-config", "./v50_configs/v50-signals-0_5_3_3368.json",
            "--manifest-template", "./v50_configs/v50-manifest-template-0_5_3_3368.json",
            "--report", str(report_path),
            "--write-store", str(store_path),
            "--write-manifest", str(build_manifest_path),
            "--confirm-run"
        ]
        if args.sample is not None:
            build_cmd.extend(["--sample", str(args.sample)])

        try:
            run_command(build_cmd, dry_run, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\n!! build FAILED for {map_name} (exit {exc.returncode}) -- skipping finalize/curate for this map, continuing to the next map.")
            outcomes[map_name]["build"] = "failed"
            continue
        outcomes[map_name]["build"] = "ok"

        # Step B: Finalize manifest.
        # IMPORTANT: --manifest must be the REAL manifest build just wrote (row_count and content
        # hashes from what actually landed on disk), never the blank v50-manifest-template file --
        # the template always declares row_count=0 and placeholder hashes, so finalize would always
        # report finalization_state=incomplete/exit 1 against a perfectly good store (see Spec 109
        # Phase 8 incident write-up).
        #
        # finalize exiting non-zero here means finalization_state=incomplete, which is a normal,
        # expected outcome for real client data (some tiles legitimately lack a required signal) --
        # it does not stop this script. finalize itself now prints every concrete reason.
        finalize_cmd = [
            "uv", "run", "python", "scripts/v50_build_dataset.py", "finalize",
            "--store", str(store_path),
            "--manifest", str(build_manifest_path),
            "--row-lineages", str(report_path),
            "--output", str(manifest_path)
        ]
        finalize_rc = run_command(finalize_cmd, dry_run, check=False)
        if finalize_rc == 0:
            outcomes[map_name]["finalize"] = "complete"
        else:
            print(f"\n!! {map_name}: finalize reported finalization_state=incomplete (see reasons above). "
                  f"This is expected when some real tiles lack a required signal -- continuing to curation, "
                  f"which drops exactly those rows.")
            outcomes[map_name]["finalize"] = "incomplete"

        # Step C: Pre-Curate and Pre-Bucket (Spec 103 Curation)
        # Filters out:
        #   - missing_signal
        #   - blank_minimaps (RGB std < 1.0)
        #   - object_contaminated (object_precise_mask coverage > 0.0)
        #   - height_normal_mismatch (height flat but normals show relief)
        curate_cmd = [
            "uv", "run", "python", "scripts/spec103_curate_dataset.py",
            "--store", str(store_path),
            "--output", str(curation_path),
            "--max-object-coverage", "0.0"  # Strictly filter out ANY object-contaminated tiles
        ]
        if args.sample is not None:
            curate_cmd.extend(["--batch", str(args.sample)])

        try:
            run_command(curate_cmd, dry_run, check=True)
            outcomes[map_name]["curate"] = "ok"
        except subprocess.CalledProcessError as exc:
            print(f"\n!! curate FAILED for {map_name} (exit {exc.returncode}) -- continuing to the next map.")
            outcomes[map_name]["curate"] = "failed"

        # Step D: Object-inclusive curation manifest. Step C's --max-object-coverage 0.0 is correct
        # for minimap-to-height reconstruction specifically (an object occludes the ground, so
        # "true height under it" is an impossible target from the minimap alone -- see
        # spec103_curate_dataset.py's own docstring) but it is NOT a general data-quality filter:
        # it silently drops every tile touched by any MDDF/MODF footprint, which is real, wanted
        # data for anything object-aware (v50's frozen signal catalog keeps object_precise_mask /
        # object_instance_mask as first-class signals for exactly that reason). This second manifest
        # applies the same missing_signal / blank_minimap / height_normal_mismatch checks but leaves
        # object tiles in, so object-aware work has a clean manifest that doesn't require reopening
        # the raw store. Nothing here duplicates array data -- like Step C, this only writes a
        # Parquet row-reference manifest.
        curate_object_inclusive_cmd = [
            "uv", "run", "python", "scripts/spec103_curate_dataset.py",
            "--store", str(store_path),
            "--output", str(curation_path_object_inclusive),
            "--max-object-coverage", "1.0"  # effectively disables the object filter; coverage is bounded to [0, 1]
        ]
        if args.sample is not None:
            curate_object_inclusive_cmd.extend(["--batch", str(args.sample)])

        try:
            run_command(curate_object_inclusive_cmd, dry_run, check=True)
            outcomes[map_name]["curate_object_inclusive"] = "ok"
        except subprocess.CalledProcessError as exc:
            print(f"\n!! object-inclusive curate FAILED for {map_name} (exit {exc.returncode}) -- continuing to the next map.")
            outcomes[map_name]["curate_object_inclusive"] = "failed"

    print(f"\n==============================================")
    print(" Pipeline Summary")
    print(f"==============================================")
    any_build_failed = False
    for map_name in MAPS:
        o = outcomes[map_name]
        if o["build"] == "failed":
            any_build_failed = True
        print(f"  {map_name}: build={o['build']} finalize={o['finalize']} curate={o['curate']} curate_object_inclusive={o['curate_object_inclusive']}")

    if dry_run:
        print("\nDry run completed successfully. Specify --confirm to run the actual extraction.")
        return 0

    if any_build_failed:
        print("\nAt least one map's build step failed -- see per-map summary above.")
        return 1

    print("\nAll maps extracted and pre-curated. Any finalize=incomplete map had curation drop its dirty rows; see reasons above and the per-map manifest JSON for detail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
