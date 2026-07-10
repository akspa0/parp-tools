"""Throwaway: list US datacenters and their in-stock GPUs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PREFERRED = {
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 4070",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 5090",
}


def main(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(text.lstrip())
    lines: list[str] = ["US datacenters / in-stock GPUs (stockStatus non-empty):"]
    pref_lines: list[str] = ["", "Preferred GPUs in stock in US:"]
    found_pref = False
    for dc in data:
        if dc.get("location") != "United States":
            continue
        in_stock = [
            (g["gpuId"], g.get("stockStatus") or "")
            for g in dc.get("gpuAvailability", []) or []
            if g.get("stockStatus")
        ]
        if not in_stock:
            continue
        lines.append(f"  {dc['id']}:")
        for gid, stock in in_stock:
            mark = " *" if gid in PREFERRED else ""
            lines.append(f"      {gid}{mark}  [{stock}]")
            if gid in PREFERRED:
                found_pref = True
                pref_lines.append(f"  {dc['id']:10} {gid}  [{stock}]")
    out = "\n".join(lines + pref_lines if found_pref else lines + pref_lines[:1])
    print(out)
    Path(path).with_suffix(".usgpus.txt").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))