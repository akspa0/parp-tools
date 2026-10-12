"""Resolve every ATEX/ADOO name in the DAT v26 corpus to a FileDataID using a community listfile
(`id;path` CSV lines). Default: the vendored libs/wowdev/wow-listfile/parts/*.csv. Pass a newer
listfile path as argv[1] to re-run against it. Reports coverage and the unresolved names."""
import os, sys, struct, glob

corpus = r"I:\parp\parp-tools\wow-viewer\test_data\v22_adts\unknown"
default_parts = r"I:\parp\parp-tools\wow-viewer\libs\wowdev\wow-listfile\parts\*.csv"
sources = [sys.argv[1]] if len(sys.argv) > 1 else sorted(glob.glob(default_parts))

by_path = {}
for src in sources:
    with open(src, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fid, sep, path = line.rstrip("\r\n").partition(";")
            if sep and fid.isdigit():
                by_path[path.replace("\\", "/").lower()] = int(fid)

def names(data, want):
    pos = 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if cid == want:
            yield data[pos + 8:pos + 8 + size].split(b"\0")[0].decode("latin-1")
        pos += 8 + size

tex, doo = set(), set()
for f in os.listdir(corpus):
    if "." in f:
        continue
    data = open(os.path.join(corpus, f), "rb").read()
    tex.update(names(data, "ATEX"))
    doo.update(names(data, "ADOO"))

print(f"listfile entries loaded: {len(by_path)} from {len(sources)} file(s)")
for label, group in (("ATEX textures", tex), ("ADOO models", doo)):
    hits = {n: by_path.get(n.replace("\\", "/").lower()) for n in group}
    found = sum(1 for v in hits.values() if v is not None)
    exts = sorted({os.path.splitext(n)[1].lower() for n in group})
    print(f"\n{label}: {found}/{len(group)} resolve to a FileDataID; extensions={exts}")
    for n, v in sorted(hits.items())[:5]:
        print(f"  {v!s:>9}  {n}")
    missing = sorted(n for n, v in hits.items() if v is None)
    if missing:
        print(f"  unresolved ({len(missing)}), first 10:")
        for n in missing[:10]:
            print(f"    {n}")
