def get_col_indices(sql_path, tbl_name):
    print("=== " + sql_path + " : " + tbl_name + " ===")
    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        in_tbl = False
        idx = 0
        for line in f:
            l = line.strip()
            if "CREATE TABLE" in l.upper() and tbl_name in l.lower():
                in_tbl = True
                idx = 0
            elif in_tbl:
                if l.startswith(")"):
                    break
                if not l.startswith(("KEY", "PRIMARY", "UNIQUE", "CONSTRAINT", "/*")):
                    tokens = l.replace("`", " ").split()
                    if tokens:
                        col = tokens[0]
                        print(f"  {idx}: {col}")
                        idx += 1

get_col_indices(r"I:\parp\parp-tools\wow-viewer\libs\alpha-core\etc\databases\world\world.sql", "quest_template")
get_col_indices(r"H:\trinitycore\TDB_full_world_335.26091_2026_09_09.sql", "quest_template")
