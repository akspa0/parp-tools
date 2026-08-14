# Spec Status Router

This is the current routing index for `wow-viewer`. It is intentionally short. The owning spec and
task file are authoritative; this file only tells a new agent where to start.

## Start here

| Priority | Spec | State | Next bounded action |
|---|---|---|---|
| 1 | [147 Minimap, fog, and doodad instancing](147-minimap-fog-instancing/spec.md) | Phase 2 implemented; Phase 3/4 open | User-run fullscreen drag/triple-click proof, then implement fog coverage before expanding doodad batch diagnostics |
| 2 | [146 Audio and camera playback](146-audio-camera-playback/spec.md) | Area music resolution implemented; playback gates open | User-run DBC ZoneMusic proof; MIDI/DLS remains explicitly unsupported until a backend is proven |
| 3 | [144 Camera capture paths](144-camera-capture-paths/spec.md) | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions |
| 4 | [145 WoW UI overhaul](145-wow-ui-overhaul/spec.md) | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks |
| 5 | [143 World context and lighting](143-world-context-lighting/spec.md) | LIT source/fallback and pre-alpha v2 parser implemented with user gate | Validate variant switching and no-Light-DBC fallback, then continue WMOAreaID and lighting evidence |
| 6 | [142 World scene graph](142-world-scene-graph/spec.md) | In progress | User-run dense-WMO capture to compare internal-doodad batch submissions against the previous placement-local path |

## Other lanes

- Terrain/minimap reconstruction: Specs 139–141. Use only when the user names that lane or the
  status row in the selected task points there.
- Cross-era renderer research: Spec 138. Evidence/planning lane; do not generalize one client build
  to all 4.x clients.
- PM4: Specs 128–131 and `memory-bank/workstream-pm4-decode.md`; use the PM4 spec pack before code.
- Older numbered specs not listed above are background, superseded, or archived unless their task
  file is explicitly named.

## Status rules

- `Draft` means design exists; it is not implementation proof.
- `Implementing` means code may exist but required validation is open.
- `Implemented with user gates` means focused source/build proof exists and real-client proof remains.
- `Complete` requires the owning spec's validation gates to pass; do not infer it from task counts.
- When a spec becomes inactive, update this file and move detailed session history to its workstream
  or archive. Do not keep stale “current” prose in the agent guide.
