---
name: speckit-checklist
description: Run a compliance checklist against a feature spec, plan, or implementation. Use when the user says "checklist", "$speckit-checklist", or wants to verify spec compliance.
---

# Speckit Checklist

Verify that a feature's spec, plan, and implementation are aligned and compliant with project conventions.

## When to Use

- User says "checklist", "$speckit-checklist", or "verify compliance"
- After completing a phase of implementation
- Before committing a batch of changes
- When switching between features

## Checklist Items

### Spec Completeness

- [ ] `spec.md` exists in the feature directory
- [ ] At least one P1 user story exists
- [ ] All user stories have acceptance scenarios (Given/When/Then)
- [ ] All functional requirements are specific and testable
- [ ] Success criteria are measurable

### Plan Alignment

- [ ] `plan.md` exists and references the correct spec
- [ ] Every requirement in spec.md has a corresponding phase in plan.md
- [ ] File paths in plan.md are real and exist in the repo
- [ ] Constitution check is complete (no violations or justified)

### Task Coverage

- [ ] `tasks.md` exists and references the correct plan
- [ ] Every requirement from spec.md has at least one task
- [ ] Every file in plan.md has at least one task
- [ ] Tasks are marked `[P]` for parallel where applicable
- [ ] Checkpoints exist after each user story

### Implementation Match

- [ ] All tasks in the current phase are checked off
- [ ] Code builds without errors
- [ ] Tests pass
- [ ] No new files reference paths outside `wow-viewer/`
- [ ] No constitution violations introduced

### Doc Hygiene

- [ ] Related architecture docs in `wow-viewer/docs/architecture/` are updated if behavior changed
- [ ] Memory bank (`activeContext.md`, `progress.md`) is updated if this was a significant session
- [ ] No stale references to `H:\CLIENTS` in any new or modified files

## How to Run

Read the feature's `spec.md`, `plan.md`, and `tasks.md`. Then check each item above against the actual code. Report pass/fail for each category.

## Output

- Checklist results (pass/fail per item)
- List of any failures with remediation steps
