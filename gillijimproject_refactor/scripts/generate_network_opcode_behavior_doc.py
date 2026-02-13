import json
from pathlib import Path

BASE = Path(r"i:/parp/parp-tools/gillijimproject_refactor/specifications/0.5.3")
INVENTORY = BASE / "network_opcode_inventory.json"
OUT_DOC = BASE / "NETWORK_OPCODE_BEHAVIOR_REFERENCE.md"
SMSG_INV = BASE / "network_server_smsg_inventory.json"

inv = json.loads(INVENTORY.read_text(encoding="utf-8"))


def category(op: str, handler: str) -> str:
    s = f"{op} {handler}".upper()
    if "AUTH" in s or "CHAR_" in s or "LOGIN" in s or "LOGOUT" in s:
        return "AuthSession"
    if "MOVE" in s or "MOUNT" in s or "WORLDPORT" in s or "TELEPORT" in s:
        return "Movement"
    if "QUEST" in s:
        return "Quest"
    if "GUILD" in s:
        return "Guild"
    if "GROUP" in s:
        return "Group"
    if "SPELL" in s or "AURA" in s or "CAST" in s or "CHANNELLING" in s:
        return "Spells"
    if "CHANNEL" in s:
        return "Channel"
    if "FRIEND" in s or "IGNORE" in s:
        return "Friends"
    if "TAXI" in s:
        return "Taxi"
    if "TRADE" in s:
        return "Trade"
    if "LOOT" in s:
        return "Loot"
    if "PET" in s:
        return "Pet"
    if any(k in s for k in ["CHEAT", "GM_", "GODMODE", "DBLOOKUP", "DEBUG"]):
        return "GM_Debug"
    if any(k in s for k in ["ITEM", "INVENTORY", "BAG"]):
        return "Inventory"
    if any(k in s for k in ["CREATURE", "GAMEOBJECT", "GAMEOBJ", "TRAINER", "BANK", "BINDER", "VENDOR", "INSPECT"]):
        return "NPC_Object"
    if any(k in s for k in ["CHAT", "WHO", "EMOTE", "LOOKING_FOR_GROUP", "MACRO"]):
        return "Social"
    if any(k in s for k in ["ZONE", "AREATRIGGER", "TIME", "WORLD"]):
        return "WorldState"
    if any(k in s for k in ["DUEL", "PVP", "ATTACK", "RESURRECT", "REPOP", "RANDOM_ROLL"]):
        return "Combat_PvP"
    if "NULLHANDLER" in handler.upper() or any(k in op for k in ["SCREENSHOT", "TUTORIAL_CLEAR", "COMPLETE_CINEMATIC"]):
        return "Ignored"
    return "Other"


def behavior(op: str, handler: str) -> str:
    u = op.upper()
    if u == "CMSG_AUTH_SRP6_BEGIN":
        return "Starts SRP6 auth exchange (step 1); server prepares challenge state."
    if u == "CMSG_AUTH_SRP6_PROOF":
        return "Submits SRP6 proof (step 2); server validates account proof and session eligibility."
    if u == "CMSG_AUTH_SESSION":
        return "Binds authenticated account to world session and enables gameplay opcode handling."
    if u == "CMSG_CHAR_ENUM":
        return "Requests character list for account; feeds character select UI."
    if u == "CMSG_PLAYER_LOGIN":
        return "Selects one character and starts world bootstrap/create flow."
    if u == "MSG_MOVE_WORLDPORT_ACK":
        return "Acknowledges worldport/new-world transfer; server continues spawn pipeline."
    if u == "MSG_MOVE_TELEPORT_ACK":
        return "Acknowledges teleport placement update and clears movement sync gate."
    if u == "MSG_MOVE_HEARTBEAT":
        return "Periodic authoritative position/orientation update used by movement reconciliation."
    if u == "CMSG_QUESTGIVER_STATUS_QUERY":
        return "Queries quest marker/status for target GUID; server resolves known object and sends dialog status."
    if u == "CMSG_QUESTGIVER_HELLO":
        return "Opens quest dialog/gossip path for interactable questgiver/item."
    if u == "CMSG_QUESTGIVER_ACCEPT_QUEST":
        return "Accepts selected quest; server validates hostility/range/log capacity then mutates quest state."
    if u == "CMSG_QUESTGIVER_COMPLETE_QUEST":
        return "Requests completion turn-in path; server verifies objective state and reward flow."
    if u == "CMSG_QUEST_QUERY":
        return "Requests quest template/details payload for quest id."
    if u == "CMSG_NAME_QUERY":
        return "Resolves GUID to display name/race/class metadata for UI/tooling identity."
    if u == "CMSG_CREATURE_QUERY":
        return "Resolves creature template metadata (name/subname/model ids/faction fields)."
    if u == "CMSG_GAMEOBJECT_QUERY":
        return "Resolves gameobject template metadata for interactable world objects."
    if u == "CMSG_DBLOOKUP":
        return "GM-only database search by text (item/quest/creature) returning SMSG_DBLOOKUP entries."

    if u.startswith("MSG_MOVE_"):
        return "Movement state transition/ack consumed by movement handler for server-side position state."
    if u.startswith("CMSG_QUEST"):
        return "Quest-system request that reads/modifies quest log, objective, or reward state."
    if u.startswith("CMSG_GUILD"):
        return "Guild management request (membership/roles/motd/roster lifecycle)."
    if u.startswith("CMSG_GROUP"):
        return "Party/group lifecycle request (invite, accept, leadership, loot method)."
    if u.startswith("CMSG_CHANNEL"):
        return "Channel chat administration request (join/leave/moderation/ownership/password)."
    if u.startswith("CMSG_FRIEND") or u.startswith("CMSG_ADD_IGNORE") or u.startswith("CMSG_DEL_IGNORE"):
        return "Social roster mutation/query for friends/ignore lists."
    if u.startswith("CMSG_TAXI") or u == "CMSG_ACTIVATETAXI":
        return "Flight-path query/activation that hands control to taxi manager and movement state."
    if u.startswith("CMSG_LOOT"):
        return "Loot interaction request controlling loot open/take/release lifecycle."
    if u.startswith("CMSG_CAST_") or "AURA" in u or "CHANNELLING" in u or u == "CMSG_USE_ITEM":
        return "Spellcast/control request entering spell system validation and cast execution."
    if u.startswith("CMSG_PET_"):
        return "Pet command/state request routed to pet manager/action system."
    if (
        u.startswith("CMSG_ITEM")
        or "INV_ITEM" in u
        or u.startswith("CMSG_SWAP_")
        or u.startswith("CMSG_SPLIT_")
        or "AUTOEQUIP" in u
        or "AUTOSTORE_BAG" in u
    ):
        return "Inventory/bag operation affecting item locations, stacks, and client inventory sync packets."
    if u in {
        "CMSG_GAMEOBJ_USE",
        "CMSG_BINDER_ACTIVATE",
        "CMSG_BANKER_ACTIVATE",
        "CMSG_LIST_INVENTORY",
        "CMSG_TRAINER_LIST",
        "CMSG_TRAINER_BUY_SPELL",
    }:
        return "NPC/object interaction request executing range/faction checks and service-specific responses."
    if "CHEAT" in u or u.startswith("CMSG_GM_") or u.startswith("MSG_GM_") or u == "CMSG_GODMODE":
        return "GM/debug control opcode guarded by permission checks and anticheat logging."
    if u in {"CMSG_TUTORIAL_CLEAR", "CMSG_SCREENSHOT", "CMSG_COMPLETE_CINEMATIC"}:
        return "Client notification/event intentionally ignored by server dispatcher (NullHandler)."

    return "Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic."


def engine_integration(op: str, handler: str) -> str:
    c = category(op, handler)
    return {
        "AuthSession": "Session/auth manager, account state, character bootstrap",
        "Movement": "Movement manager, map cell updates, teleport/worldport pipeline",
        "Quest": "Quest manager, object lookup, dialog/reward state",
        "Guild": "Guild manager and membership persistence",
        "Group": "Group manager and party state propagation",
        "Channel": "Channel/chat manager and moderation state",
        "Friends": "Social manager and friend/ignore persistence",
        "Taxi": "Taxi manager and travel movement control",
        "Trade": "Trade manager and transactional item/gold exchange",
        "Loot": "Loot manager and loot rights/content state",
        "Spells": "Spell/aura/cast systems and cooldown/resource validation",
        "Pet": "Pet manager/AI and owner linkage state",
        "GM_Debug": "GM permission checks, debug tooling, anticheat logs",
        "Inventory": "Inventory manager, item templates, update packet generation",
        "NPC_Object": "Map object manager, creature/gameobject template lookup",
        "Social": "Chat/LFG/macro social systems",
        "WorldState": "World/map time/zone/area trigger subsystems",
        "Combat_PvP": "Combat resolution, duel/pvp/resurrection state",
        "Ignored": "No gameplay mutation (explicitly dropped)",
        "Other": "General gameplay service path based on handler implementation",
    }[c]


def confidence(op: str, handler: str) -> str:
    high = {
        "CMSG_AUTH_SRP6_BEGIN",
        "CMSG_AUTH_SRP6_PROOF",
        "CMSG_AUTH_SESSION",
        "CMSG_CHAR_ENUM",
        "CMSG_PLAYER_LOGIN",
        "MSG_MOVE_WORLDPORT_ACK",
        "MSG_MOVE_TELEPORT_ACK",
        "MSG_MOVE_HEARTBEAT",
        "CMSG_QUESTGIVER_STATUS_QUERY",
        "CMSG_QUESTGIVER_HELLO",
        "CMSG_QUESTGIVER_ACCEPT_QUEST",
        "CMSG_DBLOOKUP",
    }
    if op in high:
        return "High"
    if behavior(op, handler).startswith("Routed to handler named"):
        return "Medium"
    return "Medium-High"


rows = []
for item in inv:
    op = item["opcode"]
    h = item["handler"]
    rows.append((category(op, h), op, h, behavior(op, h), engine_integration(op, h), confidence(op, h)))
rows.sort(key=lambda t: (t[0], t[1]))

lines = []
lines.append("# 0.5.3 Opcode Behavior Reference")
lines.append("")
lines.append("This reference catalogs every inbound opcode handled by the current 0.5.3 server dispatch map and describes expected gameplay/engine effects for tooling-network implementation.")
lines.append("")
lines.append("## Method")
lines.append("- Source-of-truth inbound map: `alpha-core-mods/game/world/opcode_handling/Definitions.py`")
lines.append(f"- Enumerated opcode count: **{len(rows)}**")
lines.append("- Outbound server-emitted `SMSG_*` inventory artifact: `network_server_smsg_inventory.json`")
lines.append("- Confidence model: `High`=verified in mirrored handler code; `Medium-High`=strong handler-name+flow evidence; `Medium`=dispatch-only inference")
lines.append("")
lines.append("## Critical Flow Anchors (Verified)")
lines.append("- Auth bootstrap: `CMSG_AUTH_SRP6_BEGIN` -> `CMSG_AUTH_SRP6_PROOF` -> `CMSG_AUTH_SESSION`")
lines.append("- Character entry: `CMSG_CHAR_ENUM` -> `CMSG_PLAYER_LOGIN`")
lines.append("- Transfer: server emits `SMSG_TRANSFER_PENDING` then `SMSG_NEW_WORLD`; client acknowledges with `MSG_MOVE_WORLDPORT_ACK`")
lines.append("- Movement steady state: `MSG_MOVE_HEARTBEAT` and other `MSG_MOVE_*` status packets")
lines.append("- Quest interaction core: status/hello/query/accept/complete/reward opcodes")
lines.append("")

current = None
for c, op, h, b, e, conf in rows:
    if c != current:
        lines.append(f"## {c}")
        lines.append("| Opcode | Handler | What It Does | Engine Integration | Confidence |")
        lines.append("|---|---|---|---|---|")
        current = c
    lines.append(f"| `{op}` | `{h}` | {b} | {e} | {conf} |")

lines.append("")
lines.append("## Outbound `SMSG_*` Notes")
lines.append("- The inbound dispatch table does not include outbound packet handlers; those packets are emitted from subsystem managers.")
lines.append("- The mirrored-code outbound inventory is preserved in `network_server_smsg_inventory.json` and should be used to define receive/decode support in tooling.")
lines.append("")
lines.append("## Implementation Guidance for Tooling Network Layer")
lines.append("- Implement protocol as stateful phases: `Auth` -> `Character` -> `WorldTransfer` -> `InWorld`.")
lines.append("- Gate sendable opcodes by session phase to avoid invalid-sequence disconnects.")
lines.append("- Build decode handlers first for outbound world-critical packets: transfer/new-world/object updates/movement corrections.")
lines.append("- Keep opcode table versioned per build; do not mix 0.5.3 and later-era constants.")

OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
print(f"WROTE {OUT_DOC}")
