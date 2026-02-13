# 0.5.3 Network Plan for MdxViewer Sandbox Connectivity

## Goal
Enable `MdxViewer` to connect to 0.5.3-compatible sandbox servers and display live world state (players/NPCs/objects), plus basic quest interaction context.

## Current State
- `MdxViewer` currently has no client network layer (no packet/socket/auth flow in `src/MdxViewer/**/*.cs`).
- Server-side opcode dispatch evidence exists in `alpha-core-mods`.
- Client-side world transition behavior is documented from Ghidra (`SMSG_NEW_WORLD` loading sequence).

## Primary Evidence Used
- Server dispatch map:
  - `alpha-core-mods/game/world/opcode_handling/Definitions.py`
- Server world-transfer packet flow:
  - `alpha-core-mods/game/world/managers/objects/units/player/PlayerManager.py`
- Client loading-screen trigger from network packet:
  - `specifications/outputs/053/LoadingScreens/05-StateMachine.md`
- Existing fork opcode notes (subset with numeric IDs):
  - `alpha-core-mods/README.md`

## Proven Flow Fragments

### 1) Session/Auth entry points exist on server
`Definitions.py` confirms handlers for:
- `CMSG_AUTH_SRP6_BEGIN`
- `CMSG_AUTH_SRP6_PROOF`
- `CMSG_AUTH_SESSION`
- `CMSG_PING`
- `CMSG_CHAR_ENUM`
- `CMSG_PLAYER_LOGIN`

### 2) Map/world transfer flow is explicit
`PlayerManager.py` sends, on long teleport/map change:
1. `SMSG_TRANSFER_PENDING`
2. `SMSG_NEW_WORLD` with payload packed as `<B4f`:
   - `destination_map` (`uint8` in this server implementation)
   - `x, y, z, o` (`float`)

Client Ghidra notes confirm `SMSG_NEW_WORLD` is the loading-screen transition trigger.

### 3) Movement opcodes are broadly handled
`Definitions.py` maps many movement messages to `MovementHandler.handle_movement_status`, including:
- `MSG_MOVE_HEARTBEAT`
- movement start/stop/turn/strafe/jump variants
- force-move/speed ACK opcodes

### 4) NPC/quest opcodes are present server-side
`Definitions.py` includes core quest/NPC handlers:
- `CMSG_QUESTGIVER_STATUS_QUERY`
- `CMSG_QUESTGIVER_HELLO`
- `CMSG_QUESTGIVER_QUERY_QUEST`
- `CMSG_QUESTGIVER_ACCEPT_QUEST`
- `CMSG_QUESTGIVER_COMPLETE_QUEST`
- `CMSG_QUESTGIVER_REQUEST_REWARD`
- `CMSG_QUESTGIVER_CHOOSE_REWARD`
- `CMSG_QUEST_QUERY`
- plus query ops like `CMSG_CREATURE_QUERY`, `CMSG_NAME_QUERY`

## Minimum Opcode Sets for Viewer Integration

## Phase A — Connect & Enter World (required)
Implement first:
- Outbound: auth/session/login + ping + minimal movement ack path
- Inbound: transfer/new-world + object/update stream + time/ping responses

Minimum practical set (names):
- Outbound: `CMSG_AUTH_*`, `CMSG_CHAR_ENUM`, `CMSG_PLAYER_LOGIN`, `CMSG_PING`, `MSG_MOVE_WORLDPORT_ACK`, `MSG_MOVE_HEARTBEAT`
- Inbound: `SMSG_TRANSFER_PENDING`, `SMSG_NEW_WORLD`, create/update/destroy object packets, movement correction packets

Notes:
- Exact numeric opcode IDs are not mirrored in this repo snapshot because `utils/constants/OpCodes.py` is not included under `alpha-core-mods`.
- Use opcode names as canonical contract in this phase; resolve numeric IDs from the running alpha-core codebase.

## Phase B — Visibility & Identity (NPC presence)
To show useful live entities in viewer:
- Outbound: `CMSG_NAME_QUERY`, `CMSG_CREATURE_QUERY` (on unknown GUID/entry)
- Inbound: name/template/query responses + update packets carrying position/state

Expected viewer result:
- NPCs/players appear with stable names and model references, not only GUID placeholders.

## Phase C — Quest Interaction Surface
To inspect/drive quest flows from viewer tooling:
- Outbound:
  - `CMSG_QUESTGIVER_STATUS_QUERY`
  - `CMSG_QUESTGIVER_HELLO`
  - `CMSG_QUESTGIVER_QUERY_QUEST`
  - `CMSG_QUESTGIVER_ACCEPT_QUEST`
  - `CMSG_QUESTGIVER_COMPLETE_QUEST`
  - `CMSG_QUESTGIVER_REQUEST_REWARD`
  - `CMSG_QUESTGIVER_CHOOSE_REWARD`
  - `CMSG_QUEST_QUERY`
- Inbound:
  - quest status/details/reward packets

Expected viewer result:
- Viewer can inspect nearby quest givers and quest metadata (and optionally trigger accept/turn-in in debug mode).

## Recommended MdxViewer Architecture

## 1) Add a protocol module (isolated)
Create a new folder such as:
- `src/MdxViewer/Network/`

Core components:
- `WorldConnection` (socket lifecycle)
- `PacketCodec` (header encode/decode)
- `OpcodeRegistry` (name↔id table)
- `SessionState` (auth/char/world states)
- `ObjectMirror` (GUID-indexed runtime cache)

## 2) Keep rendering decoupled from packets
Packet handlers should update an intermediate world model, not call renderer directly.

Flow:
`Packet -> Decoder -> Domain Event -> World Model -> Render sync`

## 3) Start read-mostly
Implement receive path before full gameplay send path:
- connect/login/world enter
- object create/update/destroy decode
- movement heartbeat send

This yields the fastest path to “see live sandbox entities”.

## High-Risk Areas / Unknowns
1. Opcode numeric IDs for all required messages are not present in mirrored `alpha-core-mods` files.
2. Packet payload schemas for many inbound `SMSG_*` responses are not documented here byte-for-byte.
3. Compression/encryption details (if enabled in target sandbox) are not yet captured in this pass.

## Practical Next Steps
1. Pull opcode numeric table from actual `alpha-core` (`utils/constants/OpCodes.py`) and freeze it in this repo under a versioned doc.
2. Capture packet traces for a single login+world-entry session against sandbox.
3. Implement `Phase A` only in MdxViewer and validate:
   - connect
   - world transfer packets received
   - live object stream rendered
4. Add `Phase B/C` query and quest opcodes incrementally.

## Initial “Must-Have” Checklist
- [ ] Resolve opcode numeric IDs for `Phase A` names.
- [ ] Implement world socket transport + packet framing.
- [ ] Implement auth/char/world session state machine.
- [ ] Decode transfer/new-world packets.
- [ ] Decode object create/update/destroy packets into runtime mirror.
- [ ] Emit periodic heartbeat/ping.
- [ ] Render live entity transforms from network mirror.
- [ ] Add NPC name/template queries.
- [ ] Add questgiver status/details queries.
