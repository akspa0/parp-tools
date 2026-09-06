# Plan: Agent Governance — Scope Fidelity, Receipts & Spec Hygiene (Spec 224)

**Created**: 2026-09-06
**Spec**: [spec.md](spec.md)

## Architecture

Governance is enforced at three layers, cheapest first:

1. **Static rules** — a new `## 9. Governance` section in the repo-root `AGENTS.md`, read by every
   harness at session start. This is the primary enforcement: rules exist before reasoning begins.
2. **Mechanized audit** — a `speckit-cleanup` skill (markdown procedure, harness-agnostic) that any
   agent can execute on command or on the monthly cadence. It produces receipts of its own: a dated
   report of specs audited, tasks un-checked, files archived, and memory-bank lines compressed.
3. **Ledger** — the AGENTS.md governance section carries `Last cleanup: <date>` and
   `Next cleanup due: <date>`; every session can see staleness in one glance.

## Key Decisions

### D1 — Rules live in AGENTS.md, mechanism lives in a skill
AGENTS.md states the rules and points at the skill; the skill file contains the executable audit
procedure. This keeps the always-loaded context small and the procedure versionable.

### D2 — Skill is repo-authored, user-installed
The skill source of truth is `.roo/skills/speckit-cleanup/SKILL.md` in the repo. Installation
copies it to `C:\Users\akspa\.roo\skills\speckit-cleanup\SKILL.md` so every harness discovers it.
Reinstallation is a single `Copy-Item`; the repo copy wins on conflict.

### D3 — Cleanup is receipt-driven, not judgment-driven
The audit's checks are mechanical: task checked + (receipt present?) + (named symbols exist in
code?) + (spec edited after last code change without amendment?). Anything failing is un-checked or
flagged, never rationalized.

### D4 — Archival target
Implemented/closed specs move under `wow-viewer/specs/archived/` (created on first use) and their
STATUS.md rows move to a collapsed "Archived" table. Memory-bank landed narrative older than the
current handoff moves to `memory-bank/archive/` per the existing README convention.

## File Changes

| File | Change |
|---|---|
| `AGENTS.md` (repo root) | Add `## 9. Governance` section with FR-1..FR-6 rules and the timestamp ledger. |
| `.roo/skills/speckit-cleanup/SKILL.md` | New: the audit procedure. |
| `C:\Users\akspa\.roo\skills\speckit-cleanup\SKILL.md` | Installed copy of the above. |
| `wow-viewer/specs/STATUS.md` | Registry rows updated; archived section added when first used. |
| `wow-viewer/memory-bank/activeContext.md`, `progress.md` | Compressed per cleanup runs. |
| `wow-viewer/specs/212-spatial-ui-shell/*` | 2026-09-06 operator correction: decorative-instrumentation tasks struck; scope re-aimed at ImGui panels on camera-frame surfaces. |

## Phased Roadmap

### Phase 1 — Rules and mechanism (this session)
- Author the AGENTS.md governance section (US1–US3, FR-1/2/3, ledger).
- Author and install the `speckit-cleanup` skill (US4, FR-4).
- Apply the 212 scope correction and STATUS/memory-bank reconciliation.

### Phase 2 — First cleanup run (next session boundary or 2026-10-01)
- Execute the skill across all active specs; produce its dated report; update the ledger.

### Phase 3 — Steady state
- Cleanup runs monthly; violations found by any session are corrected in that session and noted.
