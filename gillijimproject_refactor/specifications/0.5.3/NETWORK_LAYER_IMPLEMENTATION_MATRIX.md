# 0.5.3 Network Layer Implementation Matrix

## Purpose
Translate opcode research into an implementation plan that can eventually support the full 0.5.3 protocol surface while delivering value early.

## Inputs
- Inbound opcode catalog: `network_opcode_inventory.json` (222 entries)
- Per-opcode behavior notes: `NETWORK_OPCODE_BEHAVIOR_REFERENCE.md`
- Protocol flow analysis: `NETWORK_OPCODE_DEEP_DIVE.md`
- Outbound inventory seen in mirrored code: `network_server_smsg_inventory.json`

## Principles
1. **State-gated protocol**: only allow opcodes valid for the current session phase.
2. **Read-safe first**: decode/observe before sending mutating gameplay opcodes.
3. **Version-locked registry**: opcode constants and packet schemas pinned to 0.5.3.
4. **Unknown-safe behavior**: unknown packets never crash the client loop; they log and continue.
5. **Trace-driven validation**: every phase must pass replay/pcap-style verification.

## Session State Model
- `Disconnected`
- `Auth`
- `CharacterSelect`
- `WorldTransfer`
- `InWorld`
- `Recovering` (desync/reconnect fallback)

Each send opcode must include an `allowedStates` list; decoder handlers may emit state transition events.

## Implementation Workstream Matrix

| Phase | Goal | Opcode Scope | Required Packet Schemas | Engine Integration | Exit Criteria | Validation Tests |
|---|---|---|---|---|---|---|
| 0 | Core transport and framing | packet header + opcode dispatch plumbing | world packet header, length/opcode framing, optional compression marker handling | `WorldConnection`, `PacketCodec`, `OpcodeRegistry` | stable connect/read loop with unknown-op logging | malformed-length fuzz, unknown-op resilience, reconnect loop |
| 1 | Auth + character bootstrap | `CMSG_AUTH_SRP6_BEGIN`, `CMSG_AUTH_SRP6_PROOF`, `CMSG_AUTH_SESSION`, `CMSG_PING`, `CMSG_CHAR_ENUM`, `CMSG_PLAYER_LOGIN` | auth challenge/proof payloads, character enum response, ping/pong | `SessionState`, account/character models | login reaches `WorldTransfer` deterministically | scripted auth replay; invalid proof rejection; ping cadence test |
| 2 | World transfer and spawn handoff | `SMSG_TRANSFER_PENDING`, `SMSG_NEW_WORLD`, `MSG_MOVE_WORLDPORT_ACK`, `MSG_MOVE_TELEPORT_ACK` | transfer pending payload, new world payload (`map + xyz + o`), worldport ack payload | world-load coordinator, loading-screen bridge | successful transition to `InWorld` without desync | cross-map teleport replay, repeated transfer stress test |
| 3 | Movement synchronization | `MSG_MOVE_HEARTBEAT` + all `MSG_MOVE_*` status/turn/jump/strafe/start/stop + force-speed/root ACKs | movement packet variants, flags/timestamp/position/orientation blocks | movement mirror, interpolation/correction layer | remote entities move correctly and local motion stays acknowledged | jitter simulation, packet drop/out-of-order tolerance, correction convergence |
| 4 | Object lifecycle visibility | inbound create/update/destroy family (`SMSG_*` object stream) + name/creature/gameobject query pairings (`CMSG_NAME_QUERY`, `CMSG_CREATURE_QUERY`, `CMSG_GAMEOBJECT_QUERY`) | update blocks, object descriptors, GUID mapping, query response payloads | `ObjectMirror` GUID index + renderer sync adapter | NPC/player/object presence is stable with names/templates | crowded-area replay, spawn/despawn churn, GUID reuse test |
| 5 | Quest data plane | `CMSG_QUESTGIVER_*`, `CMSG_QUEST_QUERY`, quest response packets | quest status/detail/reward payloads, questgiver dialog payloads | quest view model + interaction controller | viewer can inspect questgiver status/details/reward trees | quest accept/turn-in happy path + rejection/error path |
| 6 | Service systems | inventory/loot/trade/spell/pet/social/group/guild/channel/taxi families | per-family packet schemas from behavior reference | modular handlers per subsystem | no protocol errors when exercising common UI interactions | subsystem smoke suite, mixed-op stress run |
| 7 | Full coverage hardening | remaining low-frequency and debug opcodes | completion of schema backlog + edge-case payloads | registry completeness checks, telemetry | all 222 inbound opcodes mapped with at least fallback-safe behavior | coverage report = 100% mapped; long-session soak test |

## Deliverables by Module

| Module | Responsibility | Must Be Ready By |
|---|---|---|
| `Network/WorldConnection` | socket lifecycle, reconnect, heartbeat scheduler | Phase 0 |
| `Network/PacketCodec` | frame decode/encode, endian-safe readers/writers | Phase 0 |
| `Network/OpcodeRegistry` | name↔id constants, state gating, handler lookup | Phase 1 |
| `Network/SessionStateMachine` | auth/character/world transitions and guardrails | Phase 1 |
| `Network/ObjectMirror` | GUID-indexed entity cache and update application | Phase 4 |
| `Network/QuestFacade` | questgiver/quest query orchestration | Phase 5 |
| `Network/SubsystemHandlers/*` | inventory/loot/trade/social/guild/etc. | Phase 6 |

## Coverage Rules (to “abide by all opcodes”)
1. Every opcode in `network_opcode_inventory.json` must be in one of:
   - `Implemented`
   - `ObservedNoOp` (intentionally ignored with rationale)
   - `BlockedByState` (valid but not for current phase)
2. No opcode may remain in an unclassified “unknown” bucket.
3. Decoder must preserve unknown packet bytes for forensic logging when schema is incomplete.

## Tracking Template

Use this status model per opcode:
- `Schema`: None / Partial / Complete
- `Decoder`: None / ParseOnly / Full
- `Encoder`: None / Limited / Full
- `StateGate`: Missing / Partial / Complete
- `Tests`: None / Unit / Integration / Replay

## Minimum CI Checks
1. Registry completeness check: all 222 inbound opcodes have classification.
2. Session safety check: invalid-state sends are blocked and logged.
3. Replay regression: auth→world transfer→movement scenario passes.
4. Stability check: unknown opcode does not crash loop.

## Immediate Next Steps
1. Build a machine-readable tracker (CSV/JSON) from `network_opcode_inventory.json` with the status template fields.
2. Populate tracker for Phase 1-3 opcodes first.
3. Implement replay harness using captured sandbox packet traces.
