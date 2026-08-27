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
