# Pre-release 3.0.1 M2 Variant Guide From wow.exe

## Scope

This guide captures the high-confidence `wow.exe` findings for build `3.0.1.8303` so a fresh chat can implement support without falling back to later-`3.3.5` assumptions.

It is intentionally limited to what was confirmed in the binary. Where field names are still unresolved, this guide keeps the structure in terms of validator families and load-path behavior instead of inventing semantics.

## Evidence Base

- Binary: `wow.exe` build `3.0.1.8303`
- Method: live Ghidra decompilation, call-graph tracing, and helper-function inspection
- Goal: document the actual client load and validation path for pre-release `3.0.1` M2-family assets

## High-Confidence Loader Chain

All currently inspected model-family callers converge on one shared path:

1. `FUN_0077e2c0`
   - High-level scene/model request wrapper
   - Loads the primary model
   - Optionally loads a second model path
   - Falls back to `Spells\\ErrorCube.mdx` when the primary load fails
   - Builds the scene/object wrapper and releases temporary refs
2. `FUN_0077d3c0`
   - Cache and file loader
   - Accepts `.mdx`, `.mdl`, and `.m2`
   - Internally normalizes accepted model-family extensions to `.m2` before continuing
   - Checks cache buckets, opens the file, allocates the model object, and calls the parse bootstrap
3. `FUN_0079bc70`
   - Parse bootstrap / async-load setup
   - Stores file/buffer state
   - Allocates aligned working storage
   - Installs callback `FUN_0079bc50`
4. `FUN_0079bc50`
   - Thin worker callback
   - Calls `FUN_00797450()` then `FUN_0079bb30()`
5. `FUN_0079bb30`
   - Shared model init
   - Calls the root validator `FUN_0079a8c0(...)`
   - Marks versions `< 0x108`
   - Calls `FUN_007988c0()` for later initialization
   - Emits failure strings equivalent to `Corrupt model data` and `Failed to initialize model`
6. `FUN_0079a8c0`
   - Root validator and layout walker
   - This is the authoritative file-structure gate for the pre-release client

## Entry-Gate Contract

The root validator enforces these conditions before later initialization:

1. Root magic is `MD20`
2. Accepted version range is `0x104..0x108`
3. The file is validated through typed count/offset span checks, not through a later `MD21` chunk-container model
4. Parser behavior splits structurally at `0x108`

Implementation consequence:

- For build `3.0.1.8303`, do not treat `MD21` as a normal accepted root.
- For unknown `3.0.x` profiles, the safest default is still the `MD20` + `0x104..0x108` gate until contradicted by binary evidence.

## Confirmed Span Validators

The validator uses reusable helper routines for count/offset spans. These are important because they show the file is organized around typed tables rather than later container semantics.

- `FUN_00797540`: stride `0x01`
- `FUN_00797950`: stride `0x02`
- `FUN_00797710`: stride `0x04`
- `FUN_00797830`: stride `0x08`
- `FUN_007975D0`: stride `0x0C`
- `FUN_007977A0`: stride `0x30`
- `FUN_00797680`: stride `0x44`

These helpers should be mirrored in a fresh parser as generic `ValidateSpan(count, offset, stride, fileLength)`-style checks, with overflow protection and exact end-bound validation.

## Confirmed Nested Record Families

The root validator also dispatches into nested fixed-stride families. These are the current high-confidence sizes observed in the binary:

- `FUN_00798DA0`: record stride `0x70`
- `FUN_00798320`: record stride `0x2C`
  - Includes nested helper `FUN_00797A40`
- `FUN_00798F40`: record stride `0x38`
- `FUN_007985F0`: later-stage record stride `0x2C`
- `FUN_00799340`: record stride `0xD4`
- `FUN_0079A720`: record stride `0x7C`

Practical reading:

- The pre-release file is not just a small header delta from `3.3.5`.
- There are multiple typed table families with their own nested validation.
- A clean implementation should preserve these families as explicit versioned structures instead of trying to coerce them into later Warcraft.NET assumptions.

## Version Split

The decisive branch happens at `0x108`.

### Legacy Side: `< 0x108`

- `FUN_00799EE0`: family stride `0xDC`
- `FUN_0079A1C0`: family stride `0x1F8`

Extra confirmed behavior inside `FUN_0079A1C0`:

- Contains a special-case mutation when version `< 0x106`
- Sets a flag when a short at `+0x2E` is nonzero

### Later Side: `>= 0x108`

- `FUN_00799640`: family stride `0xE0`
- `FUN_00799920`: family stride `0x234`

Implementation consequence:

- `0x108` is not a cosmetic version bump.
- A fresh parser should split the relevant record families into separate legacy and later structs instead of trying to parse both with a single permissive layout.

## Path Normalization Details

Two path-level behaviors matter because they can mask parser failures if reimplemented incorrectly:

1. `FUN_0077d3c0` accepts `.mdx`, `.mdl`, and `.m2`, then normalizes accepted model-family requests to `.m2` before continuing.
2. The high-level loader falls back to `Spells\\ErrorCube.mdx` only after the shared load path fails.

Implementation consequence:

- The viewer should treat extension aliasing and canonical-path resolution as a separate concern from binary-format support.
- A successful alias resolution does not prove the model format is supported.

## Root-Contained Profile Evidence

The latest Ghidra pass tightened one important point that was still open in the earlier guide:

1. On the traced `3.0.1.8303` path, the only confirmed model-file open in the inspected M2 chain is the primary `.m2` load in `FUN_0077d3c0`.
2. `FUN_0079bb30` then calls `FUN_0079a8c0(...)` on that in-memory `MD20` blob.
3. `FUN_007988c0()` selects a root-contained `0x2C` profile record from the parsed model buffer and stores it at `param_1 + 0x13C`.
4. `FUN_00797D20()` builds the shared vertex buffer from that selected root profile.
5. `FUN_00797AD0()` builds the shared index buffer from that selected root profile.

High-confidence shape of that root-contained `0x2C` profile family from `FUN_00798320` and `FUN_00797A40`:

- `+0x00`: typed span using stride `0x02`
- `+0x08`: typed span using stride `0x02`
- `+0x10`: typed span using stride `0x04`
- `+0x18`: typed span using stride `0x30`
- `+0x20`: typed span using stride `0x18`
- `+0x28`: scalar selector consumed by `FUN_007988C0()` when choosing the active profile

Implementation consequence:

- For the traced `3.0.1.8303` `CM2Shared` path, a missing external `.skin` should not be treated as proof of the root cause.
- The client is demonstrably capable of building geometry from root-contained profile tables inside `MD20`.
- MdxViewer still needs its own root-profile geometry/material path before it can claim full `3.0.1` support.

## What This Means For MdxViewer

### Do

1. Treat pre-release `3.0.1` as its own explicit model profile.
2. Build or route through a byte-level `MD20` parser for `0x104..0x108`.
3. Mirror the typed span validation model before trying to map the file to runtime geometry.
4. Split the legacy and later record families at `0x108`.
5. Keep the empty-conversion guardrail, but only as failure hygiene.

### Do Not

1. Do not accept `MD21` for `3.0.1` just because later branches do.
2. Do not model this as a slightly older `3.3.5` parser.
3. Do not widen `.skin` or converter fallbacks and call that format support.
4. Do not treat missing external `.skin` files as definitive proof of failure on the traced `3.0.1.8303` path.
5. Do not conflate this parser track with the neon-pink transparent-material issue.

## Recommended Implementation Order

1. Keep the current profile guard in place.
2. Introduce a dedicated pre-release `MD20` parser layer that only owns validation and structured reads.
3. Implement the generic typed-span helpers first.
4. Implement the shared nested record families next.
5. Split the `0xDC`/`0x1F8` and `0xE0`/`0x234` families behind the `0x108` gate.
6. Only after the on-disk structures are trustworthy, re-evaluate any remaining external `.skin` expectations, plus submesh and material mapping.

## Known Open Mapping Work

These items still need more Ghidra work before naming final C# fields with confidence:

- Exact semantic names for the `0x70`, `0x38`, `0x7C`, `0xDC`, `0xE0`, `0x1F8`, and `0x234` families
- The full field-level contract inside `FUN_007988c0()`
- Whether any alternate `3.0.1` caller materially differs from the traced `CM2Shared` path and still opens companion skin/view data externally
- Whether any caller-specific behavior exists above `FUN_0077e2c0` that materially changes model setup after successful parse

## Separate Track Reminder

The neon-pink transparent-surface bug is still a separate Track B issue.

Current evidence says:

- Track A: pre-release `3.0.1` model-format compatibility is still incomplete
- Track B: shared transparent-material or shader parity is also broken because the symptom reproduces on classic `MDX` and M2-family assets

Do not treat progress on one track as proof that the other is fixed.

## Fresh-Chat Entry Points

Use these prompt files together with this guide:

- `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
- `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
- `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`