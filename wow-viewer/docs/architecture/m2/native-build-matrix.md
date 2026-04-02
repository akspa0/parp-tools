# Native Build Matrix

## Purpose

This matrix consolidates what is actually known per M2-relevant build.

It is the proof boundary for implementation work. If a row is only `static-only`, do not implement later-build behavior as though runtime parity has already been proven.

## Matrix

| Build | Era | Identity | Skin strategy | External anim | Effect families | Runtime flags | Proof level | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2.0.0.5610` | beta TBC | canonical `MD20`-family model path | root-contained active profile; no `%02d.skin` proof on traced path | not the main confirmed seam | not the focus of the current note | not the focus of the current note | static-only | important boundary: do not inherit later numbered-skin behavior by default |
| `3.0.1.8303` | pre-release Wrath | strict `MD20`, version `0x104..0x108` | pre-release path; do not treat missing external `.skin` as definitive failure | not the main current focus | not the main current focus | not the main current focus | static-only | own pre-release parser contract, not just “older 3.3.5” |
| `3.3.5.12340` | Wrath | `.mdl/.mdx/.m2` normalize to `.m2` | exact choose -> `%02d.skin` load -> init -> rebuild | exact `%04d-%02d.anim` | explicit `Diffuse_*` + `Combiners_*` with runtime samples | z-fill, clip planes, threads, doodad batching, particle batching, additive sort, optimization masks | static + runtime | strongest current baseline; includes world-path choose/load/init/effect evidence |
| `4.0.0.11927` | early Cataclysm | M2 continuity plus model/effect stack evidence | exact `%02d.skin` builder and explicit choose/load/init chain recovered | exact `%04d-%02d.anim` builder and external anim loader recovered | explicit `Diffuse_*` + `Combiners_*`; broader `.bls` effect stack also present | same broad option surface, current default-return mask observed as `0x2008` | static-only | first real Cataclysm static anchor map recovered; runtime chain still missing |

## Not Yet Closed

These build slots remain open for direct evidence in the current doc set:

- `4.3.4.15595`
- `5.4.8.18414`
- `6.0.3.19116`
- `6.2.4.21355`

Do not infer their behavior from Wrath or `4.0.0.11927` without direct native evidence.

## Highest-Confidence Cross-Build Rules

1. Build-aware parsing is required.
2. Later numbered-skin behavior is real, but not universal across all eras.
3. External animation sidecars are a real runtime seam for later clients.
4. Effect-family selection is explicit and meaningful in native clients.
5. Runtime-flag behavior should remain visible as a first-class contract.
