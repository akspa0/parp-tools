"""Insert a one-paragraph archival banner under the H1 of each spec archived by the 2026-09-23
reconciliation. Idempotent (skips files that already carry the marker). Stdlib only."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)
MARK = "<!-- reconciliation-2026-09-23 -->"

EPICS = {
    "248": "248-epic-formats-and-conversion",
    "249": "249-epic-renderer-performance-and-correctness",
    "250": "250-epic-reconstruction-and-editor-platform",
    "251": "251-epic-viewer-ux-and-code-health",
    "252": "252-epic-world-simulation-and-audio",
    "253": "253-epic-pm4-navmesh-research",
    "254": "254-epic-datasets-and-terrain-ml",
}
BATCH = {}
for line in """A1 108 109 111 112 114 115 117 118 123 124
A2 125 126 127 132 133 134 139 140 141 194 196
B 046 065 128 129 130 149 184 185 186 187 188 189
C1 056 136 138 150 152 153 226 236
C2 160 198 199 200 201 202 204 206 207 242
D1 104 105 154 193 197 205 209 221 235
D2 237 238 239 240 241 243 244 245 246 247
E 009 069 072 073 110 212 223 224 225 227 228 229 231 233
F1 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183
F2 190 191 192 203 208 219 220 222 230 232 234
G1 106 107 135 137 142 143 144 146 147 148
G2 151 155 156 157 158 159 210 211 213 214 215 216 217 218""".splitlines():
    b, *ids = line.split()
    for i in ids:
        BATCH[i] = b

WORD = {"COMPLETE": "complete (code verified; any listed operator gates carried forward)",
        "FOLDED": "open residue folded into an epic",
        "SUPERSEDED": "superseded",
        "COLD": "cold (historical reference, no live residue)"}

rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(HERE, "dispositions.tsv"), encoding="utf-8") if l.strip()]
EXTRA = {"epic-client-datastore": "254", "epic-editor-platform": "250", "epic-pm4-restoration": "253"}
targets = [(sid, disp, epic, note) for sid, disp, epic, note in rows]
targets += [(k, "SUPERSEDED", v, "Old epic index; members and residue now owned by the new epic.") for k, v in EXTRA.items()]

done = 0
for sid, disp, epic, note in targets:
    dirs = [d for d in os.listdir(ARCH) if d == sid or d.startswith(sid + "-")]
    if len(dirs) != 1:
        print("SKIP (no unique dir)", sid, dirs)
        continue
    d = os.path.join(ARCH, dirs[0])
    for name in ("spec.md", "epic.md", "tasks.md", "plan.md"):
        path = os.path.join(d, name)
        if os.path.isfile(path):
            break
    else:
        print("SKIP (no md)", sid)
        continue
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    if MARK in text:
        continue
    nl = "\r\n" if "\r\n" in text else "\n"
    audit = f"[audit](../reconciliation-2026-09-23/audit/batch-{BATCH[sid]}.md)" if sid in BATCH else \
        "[reconciliation ledger](../reconciliation-2026-09-23/README.md)"
    banner = (f"{MARK}{nl}> **ARCHIVED 2026-09-23 — {WORD[disp]}.** {note} "
              f"Successor: [Epic {epic}](../../{EPICS[epic]}/spec.md). Status lines and checkboxes below are "
              f"historical and were audited against the code ({audit}); they are not implementation authority.{nl}{nl}")
    m = re.search(r"^# .*(\r?\n)", text, re.M)
    text = text[:m.end()] + nl + banner + text[m.end():].lstrip("\r\n") if m else banner + text
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    done += 1
print(done, "banners stamped")
