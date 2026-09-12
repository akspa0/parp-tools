# Workspace Instructions & Multi-LLM Routing Guide

These instructions apply to `I:/parp/parp-tools` and its active projects. Keep this file
operational, clear, and actionable across all LLM models and agent harnesses (Codex, Claude,
Gemini, DeepSeek, ChatGPT, OpenCode, Copilot).

---

## 1. Start Here

For `wow-viewer` work, read in this order:

1. `wow-viewer/specs/STATUS.md`
2. `wow-viewer/memory-bank/activeContext.md`
3. The selected spec's `spec.md`, `plan.md`, and `tasks.md`
4. Only the linked research/workstream files needed for the current task

Use **Spec Kit / Specify** for all design and implementation planning: **specify (`spec.md`) → plan (`plan.md`) → tasks (`tasks.md`) → implement one validated phase at a time**. All feature specifications, architecture plans, and task breakdowns live under `wow-viewer/specs/<spec-id>/`, **never in memory-bank**. If the request is a small fix, use the existing spec/checklist when one applies.

---

## 2. Multi-LLM Model & Agent Routing Matrix

### General Rule: NEVER USE TERRA
Never route implementation, planning, exploration, or review to Terra.

### Model-Specific Routing Profiles

| LLM Family / Ecosystem | Planning & Review Role | Implementation & Sub-Agent Role | Recommended Effort / Parameters |
|---|---|---|---|
| **Codex / OpenAI** | **Sol** (Planning / Review) | **Luna** (Implementation) | Sol at **High / Extra-High**; Luna at **Extra-High / Max**. Sol returns review findings only; Luna remediates. |
| **Anthropic Claude** | **Claude Opus** (Master Architecture & Guidance) | **Claude Sonnet** (Sub-agents & Implementation) | Opus provides global plan/review; Sonnet sub-agents run focused slices. |
| **Google Gemini** | **Gemini 3.7 Flash** | **Gemini 3.7 Flash** | Use **High Effort / Thinking High** across all Gemini primary tasks and subagents. |
| **DeepSeek** | **DeepSeek Pro / Reasoner (v4 via OpenRouter)** | **DeepSeek Flash (v4 via OpenRouter)** | DeepSeek Pro v4 owns architectural planning & review; DeepSeek Flash v4 executes implementation & sub-agents. |

### Hybrid / Cross-Model Workflow
When operating across multiple tools or models:
1. **Plan & Specify**: Plan with **DeepSeek Pro (v4)**, **Claude Opus**, or **Sol High**.
2. **Execute & Remediate**: Implement separately with **DeepSeek Flash (v4)**, **Luna (Max)**, **Claude Sonnet**, **Gemini 3.7 Flash (High)**, or **ChatGPT/Codex**.
3. **Review & Gate**: Review against the plan before closing the phase or passing to the operator.

### Codex / Standard Review Loop
```text
[Planning] Sol (High/Extra-High) -> plan.md / tasks.md
    ↓
[Implementation] Luna (Max) implements phase
    ↓
[Focused Verification] Run unit/integration tests for changed owner
    ↓
[Scope Gate] Run affected project test suites
    ↓
[Review] Sol reviews diff + done-when criteria
    ↓
CHANGES REQUIRED → Luna remediates → focused verification → re-review (max 6 cycles)
BLOCKED          → Stop and report to operator
PASS             → Report completion / await operator approval (Do not self-merge)
```

---

## 3. Sub-Agent Execution Guidelines

- Use sub-agents by default for independent, safely parallelizable discovery, analysis, focused
  verification, or bounded implementation slices. Look for these opportunities before starting
  the main work, and run independent slices in parallel when that improves throughput.
- **Disjoint Scopes**: Give each sub-agent an explicit question or deliverable and a strictly bounded
  read/write scope. Never assign overlapping write sets or broad, unfocused repository cleanup.
- **Critical Path**: The primary agent retains the critical path: architecture decisions, spec interpretation,
  conflict resolution, reviewing/integrating sub-agent patches, and final user handoffs.
- **Verification of Work**: Inspect sub-agent code and evidence before integrating or claiming completion.
- **Operator-Owned Operations**: Never delegate training, GPU/heavy jobs, broad harvests, long captures,
  billed operations, or real-client visual/FPS/audio proof to sub-agents. Prepare commands for the user.

---

## 4. Code Ownership & Architectural Boundaries

- **Core Library First**:
  - Shared data models: `wow-viewer/src/core/WowViewer.Core/`
  - Shared format I/O: `wow-viewer/src/core/WowViewer.Core.IO/`
  - Runtime/M2/world contracts: `wow-viewer/src/core/WowViewer.Core.Runtime/`
  - PM4 algorithms: `wow-viewer/src/core/WowViewer.Core.PM4/`
  - Editor contracts & operations: `wow-viewer/src/core/WowViewer.Core.Editor/`
  - Viewer shell and rendering: `wow-viewer/src/viewer/WoWViewer/`
  - CLI tools (thin wrappers only): `wow-viewer/tools/`
  - C# tests: `wow-viewer/tests/`
  - Python ML/data tooling: `wow-viewer/data-harvester/`
- **Do Not Touch Working Format Readers**:
  - Do NOT modify or break existing MPQ readers (`MpqArchiveCatalog.cs`, `NativeMpqService.cs`), ADT readers, WMO readers, or M2/MDX readers unless explicitly instructed by the operator for a verified format bug.
  - `gillijimproject_refactor/` is read-only reference code unless explicitly requested by the user for a bounded legacy fix.
  - `AlphaWdtWriter.cs` remains frozen unless a proven compatibility regression requires reopening.
- **Layer New Tooling Above Proven Base Tooling**:
  - When building new generators, inspectors, experiments, game-mode slices, or SpecKit features on top of existing renderer/editor/format behavior, preserve the existing base implementation by default even when new evidence suggests it may be incomplete or disagree with synthetic output.
  - Prefer opt-in adapters, writer modes, probes, manifests, and regression tests for new tooling. Do not change shared renderer, camera, terrain-loading, editor, or format-reader behavior merely to make new tooling pass.
  - If the base implementation itself appears wrong, open or route to a separate explicit spec with evidence and user approval. Treat future player-model camera/game-mode work as a new layer beside the editor tooling, not as an implicit rewrite of the current editor base.
- **Maintain Clear Separation**: Keep UI out of core libraries. Maintain the Alpha vs Standard terrain adapter separation.

---

## 5. Execution Boundaries & Environment

- **User-Owned Runs**: The operator runs training, GPU jobs, data harvests, long/heavy/billed operations, and real-client visual proof. Prepare exact commands and stop before execution.
- **Client Roots**: Client roots are runtime configuration. `H:\CLIENTS` is an approved local library path; never hardcode machine-local client paths into source code or portable tests/documentation. Record exact build/root/fingerprint for validation.
- **Python Environment**: Python work belongs under `wow-viewer/data-harvester/` and uses its `uv` environment. Do not launch Python from the repository root when package imports depend on that project.
- **PowerShell 7 Syntax**: Every command handed to the user must be PowerShell 7 compatible:
  - Use backticks (`` ` ``) for line continuation.
  - Use PowerShell variables and parameter quoting.
  - Do NOT use bash heredocs (`<<EOF`), `export`, `/tmp`, or POSIX-only syntax.
- **Worktree Safety**: Preserve unrelated dirty worktree changes. Stage named files only; never use broad destructive git resets or staging.

---

## 6. Validation Commands

Preferred viewer checks:

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Focused test checks (e.g. for Rosetta or Core):

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Rosetta"
```

Never claim runtime, visual, FPS, GPU, audible, or real-client proof from compilation or unit tests alone.

---

## 7. Documentation & Continuity

- **Spec Kit / Specify (`wow-viewer/specs/<spec-id>/`)**: All feature specs, technical design plans, and actionable task breakdowns are authored exclusively in Spec Kit format:
  - `spec.md`: User requirements, user stories, acceptance criteria, and constraints.
  - `plan.md`: Technical design, architectural decisions, file changes, and phase roadmap.
  - `tasks.md`: Fine-grained task checklist ordered by phase with verification gates.
  - `specs/STATUS.md`: Central registry and status ledger of all active and completed specs.
- **Memory Bank Role (Dashboard & History Only — NOT for Implementation Plans)**:
  - `wow-viewer/memory-bank/activeContext.md`: Compact live operational dashboard (active lanes, next task, proof owner, main gap, out-of-scope items).
  - `wow-viewer/memory-bank/progress.md`: Newest-first historical ledger (one compact entry per completed session or phase). Move durable findings to a research or workstream file.
- Update the relevant spec (`tasks.md`) and continuity dashboard (`activeContext.md`) in the same pass as non-trivial code changes.
- Archive superseded detail under `wow-viewer/memory-bank/archive/` and index it in README.

---

## 8. Communication Style

- **Lead with the result**: State what changed, what was validated, what remains user-owned, and the exact next bounded step.
- Do not bury unresolved proof gaps under long retrospectives.
- Format file and symbol references as clickable markdown links (`file:///...`).

---

## 9. Governance — Scope Fidelity, Receipts & Spec Hygiene (Spec 224; added 2026-09-06)

Owner spec: `wow-viewer/specs/224-speckit-governance/spec.md`. These rules are **binding on every
agent and harness**. Violating them is a process failure, not a style choice.

### 9.1 Scope freeze
- The named spec/task text is the complete implementation contract. Implement exactly what it states.
- Believed omissions or "improvements" are **proposed** as a dated spec amendment and await operator
  approval **before any code is written**. Never self-authorize additional scope.
- Nothing enters a spec document without operator-originated wording or explicit operator approval.
  Discovered fabricated scope is corrected in the same session with a dated operator-correction note.

### 9.2 Receipts required
A `tasks.md` checkbox may be set to `[x]` only with a receipt (inline or in the owning spec's
`evidence/` directory) containing:
1. Files changed, with paths.
2. Exact verification commands run and their exit status.
3. A criterion→evidence table mapping each spec acceptance criterion to **real output** (real data,
   real client, real file). Build/test output alone never satisfies a criterion naming runtime,
   visual, FPS, audible, or data behavior.

### 9.3 Write containment
All generated files — captures, exports, builds, datasets, temporaries — are written inside the
repository, a project-managed output root, or an operator-supplied path. Never write to desktops,
drive roots, OS temp directories, or other ad-hoc locations.

### 9.4 Spec-sync
When implementation diverges from its spec — including when the implementation is "better" — update
the spec in the same change or add a dated amendment note. Code and spec agree at session end.

### 9.5 Context discipline & monthly cleanup
- **The code talks for itself.** Fully implemented and operator-closed specs are archived out of the
  active registry; implemented-spec detail is not reloaded into session context.
- Run the `speckit-cleanup` skill on the first session of each month (or on operator request): audit
  tasks against real code, un-check receipt-less tasks, archive closed specs, compress memory banks,
  update the ledger below, and write a dated report under `specs/224-speckit-governance/evidence/`.

**Cleanup ledger**
- Last cleanup: 2026-09-11 (224-T201 receipt/symbol audit **COMPLETE** — all 5 active specs carrying
  checked tasks audited). 227 clean (2026-09-10); 232 had 8 stale checks corrected to `[ ]`
  (2026-09-10); 233 clean (3/3 receipts, 14 checks verified); 231 clean (0 un-checked — T001
  contradiction resolved to PASS via `inventory-v3-baseline.md`, T074 flagged as
  "receipt not in evidence/"); **223 had 16 receipt-less pre-§9.2 checks un-checked** (T101–T107,
  Gate A, T201, T202, T301, T302, Gate B, T401, T501, T502) and listed for operator decision.
  See `specs/224-speckit-governance/evidence/cleanup-2026-09-11.md`.
- Next cleanup due: 2026-10-01 (monthly cadence)

---

## 10. Source Decomposition — God-Class Freeze (Spec 228; added 2026-09-06)

Owner spec: `wow-viewer/specs/228-source-decomposition/spec.md`. The `ViewerApp_*` partial-class
split FAILED to contain growth: partial classes share one state space, so every session still loads
~33k lines of god-class context. Binding rules:

- **No new members in `WorldScene` or `ViewerApp`.** New features live in owned service classes
  that receive state via constructors/parameters — never in a new `ViewerApp_*.cs` partial file.
- **File budget:** no source file grows past ~2,000 lines; a change that would exceed it splits the
  file in the same change or opens a spec task for the split.
- **Extraction pattern:** move a cohesive feature (state + methods) into a service class; the
  god-class keeps one field + delegation; extracted classes never reach back into god-class
  internals.
- Extraction is behavior-preserving; receipts (§9.2) apply to every extraction phase.

---

## 11. UI Standardization (Spec 227; added 2026-09-06)

- Every sidebar/workbench surface uses the `SharedUiWidgets` primitives — no bespoke section
  styling. A styling mismatch (e.g., Archaeology) is a defect, not a theme.
- Every data surface has exactly one authoritative home (Spec 223 FR-3); the
  [UI re-audit](wow-viewer/specs/227-ui-reaudit/spec.md) inventory v2 is the enforcement artifact.
- New UI features must register a row in the inventory (Spec 223 FR-9) AND follow §10 (owned
  service class) in the same change.
