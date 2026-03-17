# Context Lookup Quick Guide

Use this local tool to search memory-bank and documentation quickly before editing code.

## Tool

- Script: gillijimproject_refactor/tools/doc_lookup.py
- Index file: gillijimproject_refactor/.cache/doc_lookup_index.json

## Build Index

```powershell
python gillijimproject_refactor/tools/doc_lookup.py build
```

## Query

```powershell
python gillijimproject_refactor/tools/doc_lookup.py query "alpha debug overlay terrain renderer" --limit 8
python gillijimproject_refactor/tools/doc_lookup.py query "343dadf baseline recovery cherry-pick"
```

## Stats

```powershell
python gillijimproject_refactor/tools/doc_lookup.py stats
```

## Recommended Query Topics

- "alpha terrain baseline 343dadf"
- "standardterrainadapter mcal decode"
- "terrain renderer alpha debug overlay"
- "memory-bank data-paths validation"
- "viewerapp build detection profile"

## Notes

- This is a local lookup helper, not a replacement for runtime validation.
- Rebuild the index after large documentation updates.
- Prefer this tool before broad grep passes when context is fragmented.
