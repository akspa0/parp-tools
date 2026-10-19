with open(r"I:\parp\parp-tools\wow-viewer\libs\alpha-core\etc\databases\world\world.sql", "r", encoding="utf-8", errors="ignore") as f:
    target = None
    for line in f:
        l = line.strip()
        if "CREATE TABLE `creature_quest_starter`" in l or "CREATE TABLE `npc_text`" in l or "CREATE TABLE `broadcast_text`" in l:
            target = l
            print(target)
        elif target:
            print("  ", l)
            if l.startswith(")"):
                target = None
