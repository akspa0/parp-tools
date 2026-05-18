# wow-viewer Spec Kit Doc Audit (Fresh Chat)

## Goal

Audit `wow-viewer/docs/` and classify each architecture/spec document as:

- `implemented` (landed + validated)
- `partial` (some code landed, gaps remain)
- `planned` (design-only / not started)
- `stale` (contradicted by current code/workflow)

Then convert the highest-value active lane into Spec Kit artifacts.

## Mandatory Context

1. Read `AGENTS.md` at repo root (`I:/parp/parp-tools/AGENTS.md`).
2. Read `wow-viewer/AGENTS.md`.
3. Read `wow-viewer/README.md`.
4. Read `wow-viewer/docs/architecture/v16-terrain-model-spec-2026-05-16.md`.
5. Read `wow-viewer/data-harvester/README.md`.

Respect guardrails:

- `gillijimproject_refactor` is read-only reference.
- New work belongs in `wow-viewer`.
- Keep changes bounded and evidence-based.

## Spec Kit Surfaces To Use

- `.agents/skills/speckit-specify/SKILL.md`
- `.agents/skills/speckit-plan/SKILL.md`
- `.agents/skills/speckit-tasks/SKILL.md`
- `.agents/skills/speckit-analyze/SKILL.md`
- `.agents/skills/speckit-checklist/SKILL.md`

Note: Git extension hooks are intentionally disabled in `.specify/extensions.yml`.

## Audit Workflow

1. Inventory candidate docs under `wow-viewer/docs/architecture/`.
2. For each doc, extract:
   - intent
   - claimed status
   - files/scripts it references
3. Verify against code and scripts in:
   - `wow-viewer/src/`
   - `wow-viewer/tools/`
   - `wow-viewer/data-harvester/scripts/`
   - `wow-viewer/data-harvester/src/`
4. Produce an audit table:
   - `doc_path`, `status`, `evidence`, `action`
5. Choose one active, high-impact lane and run Spec Kit flow on it:
   - `$speckit-specify`
   - optional `$speckit-clarify`
   - `$speckit-plan`
   - `$speckit-tasks`
   - optional `$speckit-analyze`
6. Write outputs to the default `wow-viewer/specs/<NNN>-<feature>/` structure.

## Initial Feature Candidate

Use this first if no override is provided:

`v16-inference-output-contract-and-patch-pipeline`

Scope:

- lock input->output dataset pairing contract for inference
- define patch-ready outputs for LK ADT patching
- define alphaWDT creation handoff boundaries
- include provenance/evidence artifacts for train/val/infer

## Deliverables

1. Audit summary markdown in `wow-viewer/docs/architecture/`:
   - `speckit-doc-audit-<YYYY-MM-DD>.md`
2. One Spec Kit feature directory under `wow-viewer/specs/`.
3. Short implementation queue (top 3 slices) linked to audit findings.

## Reporting Format

- Findings first (with file references).
- Explicitly separate `implemented` vs `planned`.
- Call out stale docs that need updates.
- Include exact commands used for verification.
