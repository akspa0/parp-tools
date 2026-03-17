---
description: "Use local context indexes and doc lookup tooling to find authoritative project guidance quickly before coding."
name: "Context Lookup"
argument-hint: "Topic, symptom, file, or question to look up"
agent: "agent"
---
Find the best guidance quickly before editing code.

Workflow:
1. Read concise indexes first:
   - gillijimproject_refactor/memory-bank/context-index.md
   - gillijimproject_refactor/src/MdxViewer/memory-bank/context-index.md
2. Build or refresh lookup index:
   - python gillijimproject_refactor/tools/doc_lookup.py build
3. Query for the requested topic:
   - python gillijimproject_refactor/tools/doc_lookup.py query "<topic>" --limit 12
4. Return:
   - top matching files
   - why each is relevant
   - exact next files to open for implementation

Rules:
- Prefer active codepath docs under gillijimproject_refactor.
- De-prioritize archived and old directories unless explicitly requested.
- If guidance conflicts, prefer memory-bank active/progress/data-paths files and current .github instructions.
