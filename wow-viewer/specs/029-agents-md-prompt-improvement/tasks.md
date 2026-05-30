# Tasks: AGENTS.md Prompt Engineering Improvement

**Feature Branch**: `029-agents-md-prompt-improvement` | **Date**: 2026-05-30 | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

## Dependency Graph

```
Phase 1: Research ──► Phase 2: Priority System ──► Phase 3: Uncertainty Protocol
                    │                              ├── Phase 4: Evidence Protocol
                    │                              ├── Phase 5: Subagent Rules
                    │                              ├── Phase 6: Testing & Change Constraints
                    │                              ├── Phase 7: Response Format & Boundaries
                    │                              └── Phase 8: GEPA/ASI Pattern Reference
                    │
                    └──────────────────────────────► Phase 9: Final Review & Validation
```

**All Phase 3-8 tasks are [P]arallelizable** — each adds a new independent section to AGENTS.md.

## Phase 1: Research & Setup

- [ ] T001 Research gepa, superx/caveman, Anbeeld/AGENTS.md, awesome-agent-md for prompt engineering patterns → see [research.md](./research.md)
- [ ] T002 Create feature spec at `wow-viewer/specs/029-agents-md-prompt-improvement/spec.md`
- [ ] T003 Create feature plan at `wow-viewer/specs/029-agents-md-prompt-improvement/plan.md`
- [ ] T004 Create requirements checklist at `wow-viewer/specs/029-agents-md-prompt-improvement/checklists/requirements.md`
- [ ] T005 Create consolidated research document at `wow-viewer/specs/029-agents-md-prompt-improvement/research.md`

## Phase 2: Priority-Based Rule System [US1]

- [ ] T006 [US1] Add Priority system section to `AGENTS.md` after RULE 11A — numbered levels (1=Correctness, 2=Evidence, 3=Safety, 4=Minimal changes, 5=Consistency, 6=Performance)

## Phase 3: Uncertainty Protocol [US2]

- [ ] T007 [P] [US2] Add Uncertainty section to `AGENTS.md` specifying when to ask questions vs proceed with assumptions

## Phase 4: Evidence Protocol [US3]

- [ ] T008 [P] [US3] Add Evidence section to `AGENTS.md` with graduated evidence-gathering rules tied to change risk level

## Phase 5: Subagent & Workflow Rules [US4]

- [ ] T009 [P] [US4] Add subagent orchestration rules (2+ or none, concrete return formats, no handoff of in-context data) to `AGENTS.md`

## Phase 6: Testing & Change Constraints [US5]

- [ ] T010 [P] [US5] Add testing requirements section to `AGENTS.md` (preserve tests, scope validation proportionally, do not silently change behavior)
- [ ] T011 [P] [US5] Add change constraints section to `AGENTS.md` (smallest viable change, reuse abstractions, no scope creep without reason)

## Phase 7: Response Format & Boundaries

- [ ] T012 [P] Add Boundaries section to `AGENTS.md` with explicit NEVER directives (fabrication, verification gaming, secrets, destructive commands)
- [ ] T013 [P] Add Response Format directive to `AGENTS.md` (concise, no filler, answer questions directly)

## Phase 8: GEPA/ASI Pattern Reference

- [ ] T014 [P] Add GEPA-style Actionable Side Information (ASI) pattern note to Codex Skill Registry or a new "Prompt Evolution" section in `AGENTS.md`

## Phase 9: Final Review & Validation

- [ ] T015 Re-read complete `AGENTS.md` — verify all existing rules (1-11, 11A) and sections are preserved intact
- [ ] T016 Verify no H:\CLIENTS references were introduced in new sections
- [ ] T017 Verify no gillijimproject_refactor violations in new content
- [ ] T018 Restore `wow-viewer/.specify/feature.json` to original value (`specs/011-v16-2-patched-signal-expansion`)

## Implementation Strategy

**MVP scope**: T006, T007, T008 (Priority System + Uncertainty + Evidence — the three highest-impact patterns from Anbeeld/AGENTS.md).

**Parallel execution**: T007-T014 are all independent and can be done in parallel.

**Validation**: Each phase produces a self-contained section addition. Read the complete file after all edits to verify no regressions.
