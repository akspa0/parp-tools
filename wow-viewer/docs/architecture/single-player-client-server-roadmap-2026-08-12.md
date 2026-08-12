# Single-Player Client and Server Direction

**Status**: Long-term architecture direction; not implemented
**Related**: Spec 146 world audio and camera playback

## Intent

WoWViewer is intended to grow beyond an inspection/rendering tool into a drop-in single-player
client-like experience: the world is explorable, built-in camera tracks and authored paths can be
played, client data is presented with its intended terrain/object/audio behavior, and a local session
authority can eventually drive NPCs, game objects, quests, and other world state.

The current viewer is already moving toward that shape through terrain/WMO/MDX/M2 rendering,
camera-path capture, configured client data access, AreaTable context, and Alpha-Core SQL-backed
NPC/game-object population. Those are foundations, not proof that a single-player client/server
already exists.

## Future ownership boundary

| Capability | Eventual authority | Current state |
|---|---|---|
| Terrain and client assets | Viewer/runtime plus existing readers | Implemented in partial, client-era-specific routes |
| Camera paths and capture | Viewer camera/capture runtime | Implemented with open real-client proof gates |
| Area ambience and positional audio | Backend-neutral viewer audio runtime | Specified in Spec 146; not implemented |
| NPC/game-object world population | Local session/server authority over Alpha-Core SQL inputs | Existing SQL population/inspection foundations; no complete server contract |
| Movement, collision, quests, AI, persistence | Future single-player session/server design | Not implemented |
| Terrain reconstruction models | Data/reconstruction workstream feeding visual coverage | Active research/engineering lane; separate from session authority |

## Sequencing constraints

1. Establish the audio runtime contract and capability evidence before tying sound to camera capture.
2. Keep client-file readers and DBC/DB2 schemas authoritative; do not hardcode era-specific layouts.
3. Keep viewer presentation, audio playback, and capture independent from future SQL/session ownership.
4. Define the local session/server contract before implementing authoritative movement, NPC AI, quests,
   or world mutation.
5. Treat terrain reconstruction outputs as data products with provenance; do not make the future
   server depend on unproven reconstructed terrain semantics.

## Immediate next bounded slice

Spec 146 begins with world-audio contracts, capability diagnostics, area ambience bindings, MCSE
emitter candidates, and one shared transport for camera preview/capture. It explicitly does not
implement login, networking, NPC AI, quests, or a local server. A later single-player session spec
must consume the runtime contracts without moving SQL or server authority into the viewer UI.
