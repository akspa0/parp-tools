---
name: speckit-specify
description: Write or refine a feature specification using Spec Kit. Use when the user says "specify", "write a spec", "$speckit-specify", or wants to define a new feature before implementing it.
---

# Speckit Specify

Write a feature specification (spec.md) for a new feature or refine an existing one.

## When to Use

- User says "specify", "write a spec", "$speckit-specify", or "define a feature"
- Starting any non-trivial new feature
- Refining an existing spec before planning

## Workflow

### 1. Gather Context

Read these before writing anything:
- `wow-viewer/.specify/memory/constitution.md` — project principles and constraints
- `AGENTS.md` — workspace guardrails and rules
- Any existing architecture docs in `wow-viewer/docs/architecture/` related to the feature area

### 2. Get Feature Description

Ask the user (if not already provided):
- What should this feature do?
- Who is the primary user?
- What problem does it solve?

### 3. Create Feature Directory

```powershell
$featureNum = "NNN"  # next sequential number from wow-viewer/specs/
$branchName = "$featureNum-<short-name>"
$featureDir = "wow-viewer/specs/$branchName"
New-Item -ItemType Directory -Path $featureDir -Force
```

Or run the setup script:
```powershell
pwsh wow-viewer/.specify/scripts/powershell/create-new-feature.ps1 -ShortName "<short-name>" "<feature description>"
```

### 4. Write spec.md

Use the template at `wow-viewer/.specify/templates/spec-template.md` as the structure. Fill in:

- **User Stories** with priorities (P1, P2, P3) — each independently testable
- **Acceptance Scenarios** — Given/When/Then format
- **Functional Requirements** — FR-001, FR-002, etc.
- **Success Criteria** — measurable outcomes
- **Assumptions** — what we're taking for granted

Write the file to `$featureDir/spec.md`.

### 5. Validate

- [ ] At least one P1 user story exists
- [ ] Each user story has acceptance scenarios
- [ ] Requirements are specific and testable
- [ ] No requirements contradict the constitution

## Output

- `wow-viewer/specs/<NNN>-<feature>/spec.md` — the feature specification
- Report the feature number and path to the user
