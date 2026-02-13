# 0.5.3 Network Opcode Deep Dive

## Objective
Produce a complete, evidence-based understanding of 0.5.3 network opcode coverage and protocol behavior, with emphasis on world entry, movement, NPC visibility, and quest interactions.

## Sources
- Dispatch map (authoritative for handled inbound opcodes):
  - `alpha-core-mods/game/world/opcode_handling/Definitions.py`
- Server world-transition/send behavior:
  - `alpha-core-mods/game/world/managers/objects/units/player/PlayerManager.py`
- Client-side loading transition evidence (`SMSG_NEW_WORLD`):
  - `specifications/outputs/053/LoadingScreens/05-StateMachine.md`
- Machine-readable extracted inventory (generated from dispatch map):
  - `specifications/0.5.3/network_opcode_inventory.json`

## Coverage Snapshot
- Total handled inbound opcodes in dispatch table: **222**
- Prefix split:
  - `CMSG_*`: **182**
  - `MSG_*`: **40**
  - `SMSG_*` are mostly outbound from server (not in inbound dispatch table)

This is effectively the protocol surface the server expects from a connected client.

## Protocol State Model (Practical)

### 1) Auth/session bootstrap
Inbound entrypoints exist for:
- `CMSG_AUTH_SRP6_BEGIN`
- `CMSG_AUTH_SRP6_PROOF`
- `CMSG_AUTH_SESSION`
- `CMSG_PING`

### 2) Character phase
- `CMSG_CHAR_ENUM`
- `CMSG_CHAR_CREATE`
- `CMSG_CHAR_DELETE`
- `CMSG_PLAYER_LOGIN`

### 3) World transfer / map entry
Server send path in `PlayerManager.py` confirms long-map transition sequence:
1. `SMSG_TRANSFER_PENDING`
2. `SMSG_NEW_WORLD` (packed as `<B4f` in this implementation)
3. client acknowledges worldport (`MSG_MOVE_WORLDPORT_ACK` handled)

Client Ghidra notes confirm `SMSG_NEW_WORLD` drives loading-screen transition.

### 4) In-world loop
- Continuous movement/status packets (`MSG_MOVE_*` family)
- Gameplay interaction opcodes (NPC, quest, inventory, spells, social, etc.)

## Dispatch Architecture
- All inbound packets are routed by `HANDLER_DEFINITIONS` in `Definitions.py`.
- Resolution is direct opcode→handler-function lookup.
- Unknown-but-valid opcode logs warning; unknown numeric opcode returns `(None, False)`.
- A small ignored set is intentionally routed to `NullHandler`.

This makes the dispatch table the single best source of protocol truth for supported behavior.

## Movement Deep Dive

### Movement coverage
Movement has the densest runtime path and includes:
- `MSG_MOVE_HEARTBEAT`
- start/stop forward/back/strafe/turn/pitch/swim/jump variants
- root/unroot/speed-change ACK variants
- collision redirect/stuck signals

Most of these map to one canonical handler: `MovementHandler.handle_movement_status`.

### Teleport/worldport semantics
- `MSG_MOVE_TELEPORT_ACK` and `MSG_MOVE_WORLDPORT_ACK` are explicitly wired.
- Same-map teleports can emit `MSG_MOVE_TELEPORT_ACK` with packed movement payload.
- Long teleports emit loading packets first and re-spawn/create state after transfer.

### Viewer implication
If implementing passive world observation, heartbeat + worldport ACK are mandatory to stay in sync and avoid session desync.

## NPC/Object Query Surface
The dispatch table confirms support for the expected object discovery/query opcodes:
- `CMSG_NAME_QUERY`
- `CMSG_CREATURE_QUERY`
- `CMSG_GAMEOBJECT_QUERY`
- interaction/use opcodes (`CMSG_GAMEOBJ_USE`, vendor/trainer/banker/binder flows)

These are the key path to convert GUID-only entities into named, typed world content.

## Quest Surface
Quest flow is fully represented in handlers:
- status + hello + query + accept + complete + request reward + choose reward
- quest log remove + confirm accept + direct `CMSG_QUEST_QUERY`

This is sufficient for a viewer/debug client to inspect and exercise quest lifecycle traffic.

## Social/Group/Guild Surface
Broad coverage exists for:
- Chat/who/text emote/LFG
- Friends/ignore
- Group lifecycle
- Guild lifecycle and emblem save
- Channel management (join/leave/moderation/owner/password/list)

This indicates a near-complete interactive world protocol rather than minimal map-stream only.

## Inventory/Loot/Trade/Spell/Pet Surface
All major gameplay systems are represented:
- Inventory operations (swap/split/autoequip/read/open/etc.)
- Loot lifecycle (request/money/release/autostore)
- Trade lifecycle
- Core spell interactions (cast/cancel/aura/channel)
- Pet actions/state/name

For viewer-only implementation these can be deferred, but they document what the server expects from a full client.

## GM/Debug Surface
The fork includes explicit debug/control handlers (e.g., `CMSG_DBLOOKUP`, invis/ghost/nuke/bind and other cheat opcodes), corroborating `alpha-core-mods/README.md`.

## Ignored Opcodes
Two opcodes are intentionally ignored through `NullHandler` in this dispatch map:
- `CMSG_TUTORIAL_CLEAR`
- `CMSG_SCREENSHOT`

`CMSG_COMPLETE_CINEMATIC` is also in ignored routing.

## What “All Opcodes” Means Here
Given available repo evidence, “all” is interpreted as:
1. Every opcode explicitly handled by `HANDLER_DEFINITIONS` (222 entries).
2. Known outbound transition packets visible in server send code (`SMSG_TRANSFER_PENDING`, `SMSG_NEW_WORLD`, etc.).

The full extracted inbound set is preserved in:
- `specifications/0.5.3/network_opcode_inventory.json`

## Gaps / Unknowns
1. Numeric opcode IDs are not mirrored in this repo snapshot for the full set (`OpCodes.py` constants are not present under `alpha-core-mods`).
2. Byte-exact payload schemas for many `SMSG_*` responses are still not documented here.
3. Compression/encryption/framing specifics for each deployment profile still need packet-capture or direct core source pass.

## Recommended Next Deep-Dive Passes
1. Import/lock full numeric opcode constants table for 0.5.3 into `specifications/0.5.3/`.
2. Build packet-structure sheets for high-priority responses:
   - world/object update packets
   - quest response packets
   - name/creature/gameobject query responses
3. Capture and annotate one full session trace:
   - auth → char enum → player login → transfer/new world → steady movement.

## Appendix A — Key Critical Opcode Families (Names)

### Auth + Character
- `CMSG_AUTH_SRP6_BEGIN`
- `CMSG_AUTH_SRP6_PROOF`
- `CMSG_AUTH_SESSION`
- `CMSG_PING`
- `CMSG_CHAR_ENUM`
- `CMSG_CHAR_CREATE`
- `CMSG_CHAR_DELETE`
- `CMSG_PLAYER_LOGIN`
- `CMSG_LOGOUT_REQUEST`
- `CMSG_PLAYER_LOGOUT`
- `CMSG_LOGOUT_CANCEL`

### World Transfer + Movement Core
- `CMSG_WORLD_TELEPORT`
- `MSG_MOVE_TELEPORT_CHEAT`
- `MSG_MOVE_TELEPORT_ACK`
- `MSG_MOVE_WORLDPORT_ACK`
- `MSG_MOVE_HEARTBEAT`
- `MSG_MOVE_START_FORWARD`
- `MSG_MOVE_STOP`
- `MSG_MOVE_JUMP`
- `MSG_MOVE_START_TURN_LEFT`
- `MSG_MOVE_START_TURN_RIGHT`
- `MSG_MOVE_START_STRAFE_LEFT`
- `MSG_MOVE_START_STRAFE_RIGHT`
- `CMSG_FORCE_SPEED_CHANGE_ACK`
- `CMSG_FORCE_SWIM_SPEED_CHANGE_ACK`

### NPC + Quest Core
- `CMSG_NAME_QUERY`
- `CMSG_CREATURE_QUERY`
- `CMSG_GAMEOBJECT_QUERY`
- `CMSG_QUESTGIVER_STATUS_QUERY`
- `CMSG_QUESTGIVER_HELLO`
- `CMSG_QUESTGIVER_QUERY_QUEST`
- `CMSG_QUESTGIVER_ACCEPT_QUEST`
- `CMSG_QUESTGIVER_COMPLETE_QUEST`
- `CMSG_QUESTGIVER_REQUEST_REWARD`
- `CMSG_QUESTGIVER_CHOOSE_REWARD`
- `CMSG_QUEST_QUERY`
