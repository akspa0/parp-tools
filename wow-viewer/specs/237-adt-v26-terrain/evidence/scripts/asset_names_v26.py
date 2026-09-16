"""Write every distinct ATEX/ADOO name from the DAT v26 corpus, one per line (input for `inspect casc exists`)."""
import os, struct, sys

corpus = r"I:\parp\parp-tools\wow-viewer\test_data\v22_adts\unknown"
out = sys.argv[1] if len(sys.argv) > 1 else "v26_asset_names.txt"

names = set()
for f in os.listdir(corpus):
    if "." in f:
        continue
    data = open(os.path.join(corpus, f), "rb").read()
    pos = 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if cid in ("ATEX", "ADOO"):
            names.add(data[pos + 8:pos + 8 + size].split(b"\0")[0].decode("latin-1"))
        pos += 8 + size

with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(sorted(names)) + "\n")
print(f"{len(names)} names -> {out}")
