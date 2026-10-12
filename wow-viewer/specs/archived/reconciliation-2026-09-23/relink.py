"""Rewrite links after the 2026-09-23 spec reconciliation moved spec directories into specs/archived/.

Stdlib only. Run from anywhere:  python relink.py [--dry-run]

1. Relative markdown links `](target)` are re-resolved: the link is interpreted against the file's
   PRE-move location, the target is mapped through the move table, and a new relative path is
   computed from the file's POST-move location. Only links whose (mapped) target exists are touched.
2. Bare textual references `specs/<moved-name>` (code comments, backticked paths, file:/// URLs)
   become `specs/archived/<moved-name>` unless already prefixed by `archived/`.
"""
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SPECS = os.path.join(REPO, "wow-viewer", "specs")
ARCH = os.path.join(SPECS, "archived")
DRY = "--dry-run" in sys.argv

# Move table (old absolute -> new absolute), derived from the staged renames.
out = subprocess.run(["git", "-C", REPO, "diff", "--cached", "--name-status", "-M"],
                     capture_output=True, text=True, check=True).stdout
moves = {}
for line in out.splitlines():
    parts = line.split("\t")
    if parts[0].startswith("R") and len(parts) == 3:
        moves[os.path.normpath(os.path.join(REPO, parts[1]))] = os.path.normpath(os.path.join(REPO, parts[2]))
old_of = {v: k for k, v in moves.items()}

moved_dirs = sorted({os.path.relpath(k, SPECS).split(os.sep)[0] for k in moves
                     if os.path.relpath(k, SPECS).split(os.sep)[0] not in ("archived",)})
dir_moves = {os.path.join(SPECS, d): os.path.join(ARCH, d) for d in moved_dirs if os.path.isdir(os.path.join(ARCH, d))}
for old, new in moves.items():  # single moved files (NEXT-DAY-PLAN, active-epics)
    dir_moves.setdefault(old, new)


def map_path(p):
    p = os.path.normpath(p)
    if p in dir_moves:
        return dir_moves[p]
    for old, new in dir_moves.items():
        if p.startswith(old + os.sep):
            return new + p[len(old):]
    return p


LINK = re.compile(r"(\]\()(<?)([^)\s>]+)(>?)(\))")
TEXT = re.compile(r"(?<!archived/)(?<![\w.-])specs/(" + "|".join(re.escape(d) for d in moved_dirs if re.match(r"\d{3}-|epic-", d)) + r")(?=[/`)\s\"'\]#:,.]|$)")

files = subprocess.run(["git", "-C", REPO, "ls-files", "-z"], capture_output=True, text=True, check=True).stdout.split("\0")
EXT = (".md", ".json", ".gitignore")  # docs only: code comments are left untouched (AGENTS.md section 4 reader freeze)
changed = 0
for rel in files:
    if not rel or rel.startswith("gillijimproject_refactor/") or not rel.endswith(EXT):
        continue
    new_abs = os.path.normpath(os.path.join(REPO, rel))
    if not os.path.isfile(new_abs):
        continue
    old_abs = old_of.get(new_abs, new_abs)
    try:
        with open(new_abs, encoding="utf-8", newline="") as f:  # preserve CRLF/LF as-is
            text = f.read()
    except (UnicodeDecodeError, OSError):
        continue
    orig = text

    if rel.endswith(".md"):
        def fix(m):
            target = m.group(3)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "/")):
                return m.group(0)
            path, sep, frag = target.partition("#")
            if not path:
                return m.group(0)
            old_target = os.path.normpath(os.path.join(os.path.dirname(old_abs), path.replace("%20", " ")))
            new_target = map_path(old_target)
            if not os.path.exists(new_target):
                return m.group(0)
            new_rel = os.path.relpath(new_target, os.path.dirname(new_abs)).replace(os.sep, "/")
            if path.endswith("/") and not new_rel.endswith("/"):
                new_rel += "/"
            if new_rel == path:
                return m.group(0)
            return m.group(1) + m.group(2) + new_rel + sep + frag + m.group(4) + m.group(5)
        text = LINK.sub(fix, text)

    text = TEXT.sub(lambda m: "specs/archived/" + m.group(1), text)
    text = text.replace("specs/NEXT-DAY-PLAN.md", "specs/archived/status-history/2026-09-02-next-day-plan.md")
    text = text.replace("specs/epics/active-epics.md", "specs/archived/status-history/2026-09-06-active-epics.md")

    if text != orig:
        changed += 1
        print(("would fix " if DRY else "fixed ") + rel)
        if not DRY:
            with open(new_abs, "w", encoding="utf-8", newline="") as f:
                f.write(text)
print(f"{changed} files {'would change' if DRY else 'changed'}")
