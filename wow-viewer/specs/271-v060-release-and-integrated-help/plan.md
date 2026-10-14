# Spec 271 — Plan

## Phase A (release) — mechanics

1. Version: single source `eng/Version.props` (`VersionPrefix 0.6.0`, `InformationalVersion 0.6.0`).
2. Release notes read by `.github/workflows/wowviewer-release.yml` from `wow-viewer/docs/releases/<tag>.md`;
   an unsuffixed tag is a full release automatically, and the notes carry `<!-- release: full -->` anyway.
3. Git: commit docs on `v0.6.0-dev`; `git checkout main`; `git merge --no-ff v0.6.0-dev`; `git tag v0.6.0`;
   `git push origin main v0.6.0`; `git checkout -b v0.6.1-dev`; `git push -u origin v0.6.1-dev`.
4. Verify with `dotnet build WowViewer.slnx -c Debug` (exit 0) before tagging.

## Phase B (help system) — design sketch for v0.6.1-dev

- New owned service `HelpGuideService` (AGENTS §10: no members added to `WorldScene` / `ViewerApp`), fed by
  a manifest `docs/WoWViewer/help/guide.json` (topic id → title, markdown body, image list).
- Rendering via `SharedUiWidgets` primitives (AGENTS §11); register a row in the UI inventory.
- Images shipped as content next to the executable; the markdown guide and the in-app help share one source.
- Shot list generated from the workbench destination inventory; operator captures into `docs/WoWViewer/images/`.

## Phase C (operator-directed corrective release v0.6.0.1, 2026-10-14)

1. Add every `wow-viewer/tools/**/*.csproj` to `WowViewer.slnx`; publish the viewer to the platform's
   main `publish/<rid>/` tree and every CLI project into its shared `publish/<rid>/tools/` directory.
   Common runtime libraries, DBDs and listfiles are thereby emitted once. Keep apphost/DLL/deps/runtimeconfig
   names project-specific; copy the tool README/helper scripts.
2. Add a CI guard that verifies every discovered tool emitted its DLL and runtimeconfig; include
   `v0.6.1-dev` in the branch trigger while preserving the four-platform release matrix.
3. Carry terrain-ground-effect tint as a per-instance RGB multiplier from the placement result to
   the shared MDX geoset material color. Neutral MCCV remains 1.0 (`raw * 2`), `IgnoreMCCV` remains
   white, and shadow attenuation is applied before submission.
4. Upload viewer and tools as distinct artifacts; the release workflow creates four viewer ZIPs and
   four tools ZIPs. Set version `0.6.0.1`, add full notes, and tag/push only after all eight bundles
   and focused tests pass. Restore `0.6.1-dev` metadata on the development branch.
