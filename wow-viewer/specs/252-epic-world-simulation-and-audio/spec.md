# Epic 252 — World Simulation, Audio & Interaction

**Created**: 2026-09-23 (spec reconciliation) · **Branch**: `v0.6.0-dev` · **Status**: Triage pending

> Primary successor of 15 archived specs (split specs also contribute items; see the ledger). **No new scope** (§9.1). Evidence:
> [reconciliation ledger](../archived/reconciliation-2026-09-23/README.md), audits
> [G1](../archived/reconciliation-2026-09-23/audit/batch-G1.md) · [G2](../archived/reconciliation-2026-09-23/audit/batch-G2.md) ·
> [B](../archived/reconciliation-2026-09-23/audit/batch-B.md).

## Goal

The world behaves: correct area context, audio, camera paths, interaction and (5.0.1) physics and
weather — each era-correct and evidenced against the native client.

## Delivered baseline (verified in code 2026-09-23 — do not re-plan)

| Capability | Source |
|---|---|
| Camera paths: author / save / replay, client M2/MDX camera import with `CinematicCamera.dbc` origin, overlay, tile-footprint preload (tests) | 144 |
| `WorldAudioRuntime`: area ambience, MCSE positional emitters, liquid sound emitters, OpenAL PCM-WAV playback, audio diagnostics | 146 US2, 148 Ph 1 |
| AreaTable resolution to ZoneText/SubzoneText in the status bar; Alpha world clock (LIT header fix, source switcher, 2,880-unit / 24-min contract) | 143 US1, T030a–e |
| WMO group area names via `WMOAreaTable.dbc` | 236 |
| 3D scene cursor + spatial selection | 210 |
| WMO interior ray picking, doodad selection, ghost transparent wireframes | 211 |
| WTF sweep tool (2 of ~10 builds swept; 2,217-name probe was a real negative) | 159 (158 US1) |
| 5.0.1 `.phys` reader, era/admission policy, Ghidra caller map, solver-selection research | 214 |

## Known "we thought we had it" gap

Zone music is **not** playing. `WorldAudioPlaybackPolicy.AutomaticZoneMusicPlaybackEnabled = false`,
and `WorldAudioRuntime` still treats `AreaTable.ZoneMusic` as a direct SoundEntries id. The 0.5.3
Ghidra evidence shows the client resolves it through `ZoneMusic.dbc` (146 T017a = 148 T006a).

## Backlog (spec-stated residue — each item awaits operator triage in [TRIAGE.md](../TRIAGE.md))

### A. Audio (146 + 148 merged)

| ID | Item | Source |
|---|---|---|
| W-01 | Build-aware `ZoneMusic` reader + indirection; then enable zone music | 146 T017a; 148 T006/T006a |
| W-02 | Explicit camera-path audio binding + shared transport | 146 US1 T009–T014 |
| W-03 | Play+Video audio muxing (video capture is silent) | 146 US3 T025–T027 |
| W-04 | Audio capability diagnostics export + bus controls | 146 US4 T029–T032 |
| W-05 | Audio playback lifecycle & correctness (no runaway sound; 7 stories) | 217 |
| W-06 | World/session event seam; optional backends | 146 US5; 148 Ph 5 |

### B. Context, camera & interaction

| ID | Item | Source |
|---|---|---|
| W-10 | WMO interior area context (containment evaluator + fallback) | 143 US2 |
| W-11 | Player-head camera rig feeding context/fog/lighting in one frame snapshot | 143 US3 |
| W-12 | Cross-era / performance release gate for context + lighting | 143 US5 |
| W-13 | Game mode: head camera + bounded physics | 151 US3/US4 |
| W-14 | Triangle-precise mesh picking | 156 FR (mesh test) |
| W-15 | Alpha demo restoration: WTF command execution, WTF browser, Alt+P, camera follow, torchlight | 158 US2–US6 |
| W-16 | Finish the WTF build sweep; re-run `--listfile` validation | 159 FR-005 |

### C. Simulation (5.0.1 and museum)

| ID | Item | Source |
|---|---|---|
| W-20 | Physics: solver integration (third-party package approval T021), cloth, joints, viewer binding, budget, BOXS re-measure | 214 T002, T021–T031 |
| W-21 | Weather system decode + implementation (wind feeds 214 cloth) | 215 |
| W-22 | Model cursor as a scene light source | 216 |
| W-23 | Creature staging: spawn, equip, pose, reconstruct | 218 |
| W-24 | Server data transformer (client DBC + server SQL → NPCs/spells/text in viewer) | 186 |
| W-25 | Single-player museum world simulation (after W-24) | 187 |

## Operator verification owed on shipped code

144 T009/T014/T017/T019/T022 real-client capture proofs; 210 T401–T404 (unit test + interactive
cursor checks); 211 T407/T505 interactive picking checks; 146/148 audible playback (148 T010 STOP gate).
