import csv
import json
from pathlib import Path

BASE = Path(r"i:/parp/parp-tools/gillijimproject_refactor/specifications/0.5.3")
INBOUND_INV = BASE / "network_opcode_inventory.json"
OUTBOUND_INV = BASE / "network_server_smsg_inventory.json"

TRACKER_JSON = BASE / "network_opcode_tracker.json"
TRACKER_CSV = BASE / "network_opcode_tracker.csv"


def phase_target(opcode: str) -> str:
    op = opcode.upper()

    phase1 = {
        "CMSG_AUTH_SRP6_BEGIN",
        "CMSG_AUTH_SRP6_PROOF",
        "CMSG_AUTH_SESSION",
        "CMSG_PING",
        "CMSG_CHAR_ENUM",
        "CMSG_PLAYER_LOGIN",
    }
    phase2 = {
        "MSG_MOVE_WORLDPORT_ACK",
        "MSG_MOVE_TELEPORT_ACK",
        "SMSG_TRANSFER_PENDING",
        "SMSG_NEW_WORLD",
    }

    if op in phase1:
        return "Phase 1"
    if op in phase2:
        return "Phase 2"
    if op.startswith("MSG_MOVE_") or op in {"CMSG_FORCE_SPEED_CHANGE_ACK", "CMSG_FORCE_SWIM_SPEED_CHANGE_ACK"}:
        return "Phase 3"
    if op in {"CMSG_NAME_QUERY", "CMSG_CREATURE_QUERY", "CMSG_GAMEOBJECT_QUERY"}:
        return "Phase 4"
    if op.startswith("CMSG_QUEST"):
        return "Phase 5"
    if op.startswith("SMSG_"):
        return "Phase 4"
    return "Phase 6"


def subsystem(opcode: str, handler: str) -> str:
    s = f"{opcode} {handler}".upper()
    if "AUTH" in s or "CHAR_" in s or "LOGIN" in s or "LOGOUT" in s or "PING" in s:
        return "AuthSession"
    if "MOVE" in s or "WORLDPORT" in s or "TELEPORT" in s:
        return "Movement"
    if "QUEST" in s:
        return "Quest"
    if "GUILD" in s:
        return "Guild"
    if "GROUP" in s:
        return "Group"
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
    if "SPELL" in s or "AURA" in s or "CAST" in s or "CHANNELLING" in s:
        return "Spells"
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
    if "NULLHANDLER" in handler.upper() or any(k in opcode for k in ["SCREENSHOT", "TUTORIAL_CLEAR", "COMPLETE_CINEMATIC"]):
        return "Ignored"
    return "Other"


def make_record(opcode: str, direction: str, handler: str = "", source: str = ""):
    return {
        "opcode": opcode,
        "direction": direction,
        "handler": handler,
        "subsystem": subsystem(opcode, handler),
        "phaseTarget": phase_target(opcode),
        "schemaStatus": "None",      # None | Partial | Complete
        "decoderStatus": "None",     # None | ParseOnly | Full
        "encoderStatus": "None",     # None | Limited | Full
        "stateGateStatus": "Missing",# Missing | Partial | Complete
        "testsStatus": "None",       # None | Unit | Integration | Replay
        "priority": "High" if phase_target(opcode) in {"Phase 1", "Phase 2", "Phase 3"} else "Medium",
        "notes": "",
        "source": source,
    }


inbound = json.loads(INBOUND_INV.read_text(encoding="utf-8"))
records = []
for item in inbound:
    records.append(
        make_record(
            opcode=item["opcode"],
            direction="Inbound",
            handler=item.get("handler", ""),
            source="network_opcode_inventory.json",
        )
    )

if OUTBOUND_INV.exists():
    outbound = json.loads(OUTBOUND_INV.read_text(encoding="utf-8"))
    for op in outbound:
        records.append(
            make_record(
                opcode=op,
                direction="Outbound",
                handler="",
                source="network_server_smsg_inventory.json",
            )
        )

# stable sort
records.sort(key=lambda r: (r["direction"], r["phaseTarget"], r["opcode"]))

TRACKER_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")

fields = [
    "opcode",
    "direction",
    "handler",
    "subsystem",
    "phaseTarget",
    "schemaStatus",
    "decoderStatus",
    "encoderStatus",
    "stateGateStatus",
    "testsStatus",
    "priority",
    "notes",
    "source",
]

with TRACKER_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

print(f"WROTE {TRACKER_JSON}")
print(f"WROTE {TRACKER_CSV}")
print(f"TOTAL_RECORDS {len(records)}")
print(f"INBOUND {sum(1 for r in records if r['direction'] == 'Inbound')}")
print(f"OUTBOUND {sum(1 for r in records if r['direction'] == 'Outbound')}")
