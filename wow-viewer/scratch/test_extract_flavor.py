import os
import re
import json

def parse_sql_tuples(line):
    tuples = []
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i] != '(':
            i += 1
        if i >= n:
            break
        i += 1 # skip '('
        fields = []
        cur = []
        in_str = False
        while i < n:
            c = line[i]
            if in_str:
                if c == '\\' and i + 1 < n:
                    i += 1
                    cur.append(line[i])
                elif c == "'":
                    if i + 1 < n and line[i+1] == "'":
                        cur.append("'")
                        i += 1
                    else:
                        in_str = False
                else:
                    cur.append(c)
            else:
                if c == "'":
                    in_str = True
                elif c == ',':
                    fields.append("".join(cur).strip())
                    cur = []
                elif c == ')':
                    fields.append("".join(cur).strip())
                    cur = []
                    i += 1
                    break
                else:
                    cur.append(c)
            i += 1
        if fields:
            tuples.append(fields)
    return tuples

def clean(s):
    s = s.strip()
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        return s[1:-1]
    return s

def extract_alpha():
    path = r"I:\parp\parp-tools\wow-viewer\libs\alpha-core\etc\databases\world\world.sql"
    if not os.path.exists(path):
        return None
    print("Extracting Alpha flavor...")
    quests = {}
    cr_quests = {}
    go_quests = {}
    npc_texts = {}
    barks = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tbl = None
        for line in f:
            l = line.strip()
            if l.startswith("INSERT INTO"):
                if "`quest_template`" in l: tbl = "quest"
                elif "`creature_quest_starter`" in l: tbl = "cr_quest"
                elif "`gameobject_quest_starter`" in l: tbl = "go_quest"
                elif "`npc_text`" in l: tbl = "npc_text"
                else: tbl = None
            elif not l.startswith("(") and not l.startswith(",("):
                tbl = None

            if tbl:
                tuples = parse_sql_tuples(l)
                for t in tuples:
                    if tbl == "quest" and len(t) >= 32:
                        try:
                            qid = int(clean(t[0]))
                            min_lvl = int(clean(t[3])) if clean(t[3]).isdigit() else 0
                            qlvl = int(clean(t[5])) if clean(t[5]).isdigit() else 0
                            title = clean(t[29])
                            details = clean(t[30])
                            obj = clean(t[31])
                            end_text = clean(t[32]) if len(t) > 32 else ""
                            quests[qid] = {
                                "questId": qid,
                                "title": title,
                                "questLevel": qlvl,
                                "minLevel": min_lvl,
                                "objectives": obj,
                                "details": details,
                                "endText": end_text
                            }
                        except Exception:
                            pass
                    elif tbl == "cr_quest" and len(t) >= 2:
                        try:
                            cid = int(clean(t[0]))
                            qid = int(clean(t[1]))
                            cr_quests.setdefault(cid, []).append(qid)
                        except Exception:
                            pass
                    elif tbl == "go_quest" and len(t) >= 2:
                        try:
                            goid = int(clean(t[0]))
                            qid = int(clean(t[1]))
                            go_quests.setdefault(goid, []).append(qid)
                        except Exception:
                            pass
                    elif tbl == "npc_text" and len(t) >= 2:
                        try:
                            nid = int(clean(t[0]))
                            t0 = clean(t[1])
                            if t0 and t0.lower() != "null":
                                npc_texts.setdefault(nid, []).append(t0)
                        except Exception:
                            pass

    print(f"Alpha: {len(quests)} quests, {len(cr_quests)} creature starters, {len(go_quests)} GO starters, {len(npc_texts)} NPC texts")
    return {
        "version": "0.5.3",
        "generatedUtc": "2026-10-19T00:00:00Z",
        "creatureBarks": barks,
        "creatureGossip": npc_texts,
        "creatureQuestStarters": cr_quests,
        "gameObjectQuestStarters": go_quests,
        "quests": quests
    }

def extract_tc():
    path = r"H:\trinitycore\TDB_full_world_335.26091_2026_09_09.sql"
    if not os.path.exists(path):
        return None
    print("Extracting TrinityCore flavor...")
    quests = {}
    cr_quests = {}
    go_quests = {}
    barks = {}
    gossip = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tbl = None
        for line in f:
            l = line.strip()
            if l.startswith("INSERT INTO"):
                if "`creature_text`" in l: tbl = "creature_text"
                elif "`creature_queststarter`" in l: tbl = "cr_quest"
                elif "`gameobject_queststarter`" in l: tbl = "go_quest"
                elif "`quest_template`" in l: tbl = "quest"
                elif "`npc_text`" in l: tbl = "npc_text"
                else: tbl = None
            elif not l.startswith("(") and not l.startswith(",("):
                tbl = None

            if tbl:
                tuples = parse_sql_tuples(l)
                for t in tuples:
                    if tbl == "creature_text" and len(t) >= 5:
                        try:
                            cid = int(clean(t[0]))
                            text = clean(t[3])
                            ctype = int(clean(t[4])) if clean(t[4]).isdigit() else 12
                            prob = float(clean(t[6])) if len(t) > 6 and clean(t[6]).replace('.','',1).isdigit() else 100.0
                            comm = clean(t[12]) if len(t) > 12 else ""
                            if text:
                                barks.setdefault(cid, []).append({
                                    "entryId": cid,
                                    "text": text,
                                    "type": ctype,
                                    "probability": prob,
                                    "comment": comm
                                })
                        except Exception:
                            pass
                    elif tbl == "cr_quest" and len(t) >= 2:
                        try:
                            cid = int(clean(t[0]))
                            qid = int(clean(t[1]))
                            cr_quests.setdefault(cid, []).append(qid)
                        except Exception:
                            pass
                    elif tbl == "go_quest" and len(t) >= 2:
                        try:
                            goid = int(clean(t[0]))
                            qid = int(clean(t[1]))
                            go_quests.setdefault(goid, []).append(qid)
                        except Exception:
                            pass
                    elif tbl == "quest" and len(t) >= 78:
                        try:
                            qid = int(clean(t[0]))
                            qlvl = int(clean(t[2])) if clean(t[2]).lstrip('-').isdigit() else 0
                            min_lvl = int(clean(t[3])) if clean(t[3]).isdigit() else 0
                            title = clean(t[75])
                            obj = clean(t[76])
                            desc = clean(t[77])
                            quests[qid] = {
                                "questId": qid,
                                "title": title,
                                "questLevel": qlvl,
                                "minLevel": min_lvl,
                                "objectives": obj,
                                "details": desc,
                                "endText": ""
                            }
                        except Exception:
                            pass
                    elif tbl == "npc_text" and len(t) >= 2:
                        try:
                            nid = int(clean(t[0]))
                            t0 = clean(t[1])
                            if t0 and t0.lower() != "null":
                                gossip.setdefault(nid, []).append(t0)
                        except Exception:
                            pass

    print(f"TC: {len(quests)} quests, {len(cr_quests)} creature starters, {len(go_quests)} GO starters, {len(barks)} creature barks, {len(gossip)} gossip entries")
    return {
        "version": "3.3.5a",
        "generatedUtc": "2026-10-19T00:00:00Z",
        "creatureBarks": barks,
        "creatureGossip": gossip,
        "creatureQuestStarters": cr_quests,
        "gameObjectQuestStarters": go_quests,
        "quests": quests
    }

alpha_data = extract_alpha()
if alpha_data:
    out_alpha = r"I:\parp\parp-tools\wow-viewer\data\population\0.5.3\dialogue_flavor.json"
    with open(out_alpha, "w", encoding="utf-8") as f:
        json.dump(alpha_data, f, indent=2)
    print("Wrote Alpha flavor to", out_alpha)

tc_data = extract_tc()
if tc_data:
    out_tc = r"I:\parp\parp-tools\wow-viewer\data\population\3.3.5a\dialogue_flavor.json"
    with open(out_tc, "w", encoding="utf-8") as f:
        json.dump(tc_data, f, indent=2)
    print("Wrote TC flavor to", out_tc)
