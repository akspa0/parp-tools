# Spec Status Router

This is the current routing index for `wow-viewer`. It is intentionally short. The owning spec and
task file are authoritative; this file only tells a new agent where to start.

## Start here

| Priority | Spec | State | Next bounded action |
|---|---|---|---|
| 1 | [151 Portal-aware rendering, game mode, and simple viewer surface](151-portal-game-mode-surface/spec.md) | Implementing; Phase 1 portal checkpoint passes source proof | Implement Phase 2 pure game-mode state/physics and character-head anchor; preserve editor camera state and stop at the focused physics checkpoint |
| 2 | [149 PM4 region navigation and audio trigger controls](149-pm4-region-audio-controls/spec.md) | Implementing; resident Zone/SubZone overlay slice wired, focused tests and PM4/audio stories open | Add area aggregation tests, then implement Phase 1 contracts/caller audit and validate region navigation before retiring correlation UI |
| 3 | [150 Alpha 0.5.3 renderer performance](150-alpha-renderer-performance/spec.md) | Draft evidence/planning pack complete; Spec 151 now owns its first portal-specific implementation slice | Keep broad profiling planning here; use Spec 151 for the bounded portal optimization and later return to repeated `profile-render` baselines |
| 4 | [148 Artifact world simulator runtime](148-world-simulator/spec.md) | Phase 1 audio diagnostics partially implemented; Alpha 0.5.3 MCSE and packed AreaNumber contracts landed | Add ZoneMusic row indirection, then finish source/read/decode coverage and user client inspection |
| 5 | [147 Minimap, fog, and doodad instancing](147-minimap-fog-instancing/spec.md) | Phase 2 implemented; Phase 3/4 open | User-run fullscreen drag/triple-click proof, then implement fog coverage before expanding doodad batch diagnostics |
| 6 | [146 Audio and camera playback](146-audio-camera-playback/spec.md) | Shared high/low AreaNumber Zone/SubZone selection and master mute control implemented; client audio contracts recovered | Add ZoneMusic row resolution; MIDI/DLS remains explicitly unsupported until a backend is proven |
| 7 | [144 Camera capture paths](144-camera-capture-paths/spec.md) | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions |
| 8 | [145 WoW UI overhaul](145-wow-ui-overhaul/spec.md) | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks |
| 9 | [143 World context and lighting](143-world-context-lighting/spec.md) | LIT source/fallback and pre-alpha v2 parser implemented with user gate | Validate variant switching and no-Light-DBC fallback, then continue WMOAreaID and lighting evidence |
| 10 | [142 World scene graph](142-world-scene-graph/spec.md) | In progress | User-run dense-WMO capture to compare internal-doodad batch submissions against the previous placement-local path |

| 11 | [080 WoW UI consolidation](080-wow-ui-consolidation/spec.md) | Phase 2E Inspect/terrain page consolidation in progress | Validate Scene Placements/LOD, Experimental Terrain Lab tiles+clipboard, and Inspect Archeology dropdown |

## Other lanes

- Terrain/minimap reconstruction: Specs 139–141. Use only when the user names that lane or the
  status row in the selected task points there.
- Cross-era renderer research: Spec 138. Evidence/planning lane; do not generalize one client build
  to all 4.x clients.
- PM4: Specs 128–131 and `memory-bank/workstream-pm4-decode.md`; use the PM4 spec pack before code.
- PM4 region navigation/audio controls: Spec 149 owns the replacement UI and default-off trigger policy;
  it does not reopen PM4 decode or implement game mode/MIDI/DLS.
- Older numbered specs not listed above are background, superseded, or archived unless their task
  file is explicitly named.

## Status rules

- `Draft` means design exists; it is not implementation proof.
- `Implementing` means code may exist but required validation is open.
- `Implemented with user gates` means focused source/build proof exists and real-client proof remains.
- `Complete` requires the owning spec's validation gates to pass; do not infer it from task counts.
- When a spec becomes inactive, update this file and move detailed session history to its workstream
  or archive. Do not keep stale “current” prose in the agent guide.
