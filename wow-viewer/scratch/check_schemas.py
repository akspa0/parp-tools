import re

def inspect_tables(sql_path, target_tables):
    print("=== " + sql_path + " ===")
    current_table = None
    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if "CREATE TABLE" in line_str.upper():
                parts = line_str.replace("`", " ").split()
                if len(parts) >= 3 and parts[0].upper() == "CREATE" and parts[1].upper() == "TABLE":
                    tbl = parts[2].lower().strip("(")
                    if tbl in target_tables:
                        current_table = tbl
                        print("Table:", current_table)
                    else:
                        current_table = None
            elif current_table:
                if line_str.startswith(")"):
                    current_table = None
                elif not line_str.startswith(("KEY", "PRIMARY", "UNIQUE", "CONSTRAINT", "/*")):
                    tokens = line_str.replace("`", " ").split()
                    if tokens:
                        print("  col:", tokens[0])

inspect_tables(r"I:\parp\parp-tools\wow-viewer\libs\alpha-core\etc\databases\world\world.sql",
    {"creature_quest_starter", "gameobject_quest_starter", "quest_template", "npc_text", "broadcast_text"})

inspect_tables(r"H:\trinitycore\TDB_full_world_335.26091_2026_09_09.sql",
    {"creature_text", "broadcast_text", "npc_text", "creature_queststarter", "gameobject_queststarter", "quest_template", "quest_details"})
