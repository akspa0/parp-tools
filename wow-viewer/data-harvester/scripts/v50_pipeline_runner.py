#!/usr/bin/env python3
"""Unified pipeline runner to build, finalize, and curate the v50 dataset.
Sequentially processes Kalimdor, Azeroth, PVPZone02, and Kalidar from the client library,
finalizes their manifests, and runs the canonical Spec 122 curation pass (`WowViewer.Tool.Harvest
curate`) to classify every tile into quality buckets and mismatch findings.

A map's `finalize` step legitimately comes back `finalization_state=incomplete` whenever any real
client tile lacks a required signal (e.g. a tile with terrain but no texture data to synthesize a
minimap from, so `minimap_rgb` -- required -- has no real lineage action for that row). That is not
a build failure: the store itself is real and mostly good, and curation classifies exactly those
tiles rather than requiring a complete store. So a non-complete `finalize` does not stop this
script -- it prints every reason (`finalize` now reports them, not just the bare state), still runs
curation for that map, and continues to the next map. Only a `build` failure (the extraction
subprocess itself erroring out) skips the rest of that map's steps; other maps still run. See Spec
109 Phase 9 for the incident this fixes: a single dirty tile out of 685 previously aborted the
entire multi-map run via an unconditional `check=True`.

Each map gets ONE curation manifest (Spec 122: partition, never filter) under
`<store>/curation/<curation_run_id>/` -- every tile is classified, including object-touched,
blank, and mismatched ones; a downstream consumer selects a subset (e.g. object-excluded, for
minimap-to-height reconstruction specifically, since an object occludes the ground and makes "true
height under it" an impossible target from the minimap alone) by filtering the manifest's
`coverage_bucket`/findings at read time via `harvester.curation_store`, not by requesting a second,
separately-materialized manifest the way the retired `spec103_curate_dataset.py` two-manifest pass
did.
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

        outcomes[map_name] = {"build": None, "finalize": None, "curate": None}

        store_path = datasets_dir / f"0_5_3_3368-{map_name}.zarr"
        build_manifest_path = reports_dir / f"build-manifest-0_5_3_3368-{map_name}.json"
        manifest_path = datasets_dir / f"0_5_3_3368-{map_name}.manifest.json"
        report_path = reports_dir / f"build-0_5_3_3368-{map_name}.json"

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
                  f"which classifies (never drops) those rows.")
            outcomes[map_name]["finalize"] = "incomplete"

        # Step C: Canonical curation (Spec 122 -- WowViewer.Core.Curation, via the `curate`
        # subcommand). Supersedes the old two-manifest spec103_curate_dataset.py pass: that script
        # produced a STRICT (object-excluded) manifest and a separate OBJECT-INCLUSIVE manifest
        # because it could only ever drop rows, never label them. `curate` classifies every tile
        # into buckets and findings (including object coverage) in one pass and never drops a row,
        # so a downstream consumer selects "object-touched or not" by filtering the one manifest's
        # `coverage_bucket`/findings at read time (see harvester.curation_store) instead of needing
        # a second, separately-materialized manifest. Nothing here duplicates array data -- like the
        # old pass, this only writes a Parquet classification alongside the raw store.
        # Note: `curate` has no tile-count-limiting flag (unlike the old `--batch`/args.sample path)
        # -- it always classifies every tile already in the store's index.parquet. A --sample run
        # still limits how many tiles `build` writes, so curation naturally runs over just that
        # smaller store; there is no separate sample-limiting needed here.
        curate_cmd = [
            "dotnet", "run", "--project", "../tools/harvest/WowViewer.Tool.Harvest", "-c", "Debug", "--",
            "curate",
            "--client-root", "H:\\CLIENTS\\0_5_3_3368",
            "--store", str(store_path),
            "--map", map_name,
            "--write",
        ]

        try:
            run_command(curate_cmd, dry_run, check=True)
            outcomes[map_name]["curate"] = "ok"
        except subprocess.CalledProcessError as exc:
            print(f"\n!! curate FAILED for {map_name} (exit {exc.returncode}) -- continuing to the next map.")
            outcomes[map_name]["curate"] = "failed"

    print(f"\n==============================================")
    print(" Pipeline Summary")
    print(f"==============================================")
    any_build_failed = False
    for map_name in MAPS:
        o = outcomes[map_name]
        if o["build"] == "failed":
            any_build_failed = True
        print(f"  {map_name}: build={o['build']} finalize={o['finalize']} curate={o['curate']}")

    if dry_run:
        print("\nDry run completed successfully. Specify --confirm to run the actual extraction.")
        return 0

    if any_build_failed:
        print("\nAt least one map's build step failed -- see per-map summary above.")
        return 1

    print("\nAll maps extracted and curated. Any finalize=incomplete map's dirty rows are still present "
          "but classified (never dropped) by curation -- see reasons above and query the curation "
          "manifest (harvester.curation_store) to select which buckets a given consumer wants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
