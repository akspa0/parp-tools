---
description: "Continue Ghidra-backed reverse engineering for pre-release 3.0.1 M2 files when field mapping, validator order, or loader behavior is still unresolved."
name: "Pre-release 3.0.1 M2 Ghidra Follow-up"
argument-hint: "Optional function family, failing asset, or unresolved structure size to focus on"
agent: "agent"
---

Continue the `wow.exe` reverse-engineering pass for pre-release `3.0.1` M2 handling.

## Read First

1. `gillijimproject_refactor/documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
2. `.github/prompts/pre-release-m2-rendering-recovery.prompt.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`

## Goal

Resolve the remaining unknowns in the pre-release `3.0.1` model path directly from the binary.

This prompt is for when implementation is blocked by missing structure or loader facts, not for speculative code edits.

## Required Investigation Areas

1. Reconfirm the shared loader chain:
   - `FUN_0077e2c0`
   - `FUN_0077d3c0`
   - `FUN_0079bc70`
   - `FUN_0079bc50`
   - `FUN_0079bb30`
   - `FUN_0079a8c0`
2. Map unresolved record families one at a time:
   - `0x70`
   - `0x2C`
   - `0x38`
   - `0xD4`
   - `0x7C`
   - legacy `0xDC` and `0x1F8`
   - later `0xE0` and `0x234`
3. If useful, rename key functions or add disassembly comments so future chats do not have to rediscover them.
4. Keep Track B separate unless the binary clearly proves shared renderer behavior is also involved.

## Required Principles

- Prefer call-graph and xref evidence over guessed struct names.
- Record exact record sizes and branch points before naming fields.
- If a function meaning is still unclear, say that clearly instead of inventing a label.
- Update the guide when new facts become high confidence.

## Deliverables

Return all items:

1. New high-confidence findings
2. Exact function addresses and inferred roles
3. Any renames or comments applied in Ghidra
4. Which unresolved family is now understood and which still is not
5. Concrete implementation implications for `MdxViewer`
6. Guide or memory-bank updates required

## First Output

Start with:

1. the unresolved part you will map first
2. the exact function addresses you will inspect
3. whether the goal is implementation support, loader tracing, or field naming