---
name: speckit-plan
description: Generate an implementation plan from a feature spec using Spec Kit. Use when the user says "plan", "$speckit-plan", or wants to turn a spec into actionable phases.
---

# Speckit Plan

Generate an implementation plan (plan.md) from an existing feature specification.

## When to Use

- User says "plan", "$speckit-plan", or "create implementation plan"
- A spec.md exists for the current feature
- Ready to break the spec into phases and tasks

## Workflow

### 1. Locate the Spec

Find the active feature directory:
- Check `wow-viewer/specs/` for the most recent or user-specified feature directory
- Verify `spec.md` exists in that directory

If no spec exists, run `$speckit-specify` first.

### 2. Read Context

- Read `wow-viewer/.specify/memory/constitution.md` — check for constraint violations
- Read `spec.md` — understand all user stories and requirements
- Read related architecture docs in `wow-viewer/docs/architecture/`
- Check existing code in `wow-viewer/src/`, `wow-viewer/tools/`, `wow-viewer/data-harvester/`

### 3. Research Phase

For each technical decision:
- Check if the code already exists (glob/grep for related files)
- Identify which existing libraries and patterns to reuse
- Note any dependencies or sequencing constraints

### 4. Write plan.md

Use the template at `wow-viewer/.specify/templates/plan-template.md` as the structure. Fill in:

- **Summary** — extract from spec: primary requirement + technical approach
- **Technical Context** — language, deps, storage, testing, platform
- **Constitution Check** — verify no violations
- **Project Structure** — real file paths in the repo
- **Implementation Phases** — ordered, each with a clear goal and approach
- **Complexity Tracking** — only if constitution violations need justification

Write the file to `$featureDir/plan.md`.

### 5. Validate

- [ ] Each phase has a clear, testable goal
- [ ] Phases are in dependency order
- [ ] File paths are real and exist in the repo
- [ ] No phase violates the constitution
- [ ] The plan references the correct spec

## Output

- `wow-viewer/specs/<NNN>-<feature>/plan.md` — the implementation plan
- Report phase count and estimated complexity to the user
