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
