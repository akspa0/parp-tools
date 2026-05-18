---
name: speckit-tasks
description: Break an implementation plan into concrete, actionable tasks using Spec Kit. Use when the user says "tasks", "$speckit-tasks", or wants to see the work broken into individual steps.
---

# Speckit Tasks

Break an implementation plan into concrete, individually completable tasks.

## When to Use

- User says "tasks", "$speckit-tasks", or "break down the plan"
- A plan.md exists for the current feature
- Ready to produce the actual work items

## Workflow

### 1. Locate the Feature

Find the active feature directory:
- Check `wow-viewer/specs/` for the most recent or user-specified feature directory
- Verify both `spec.md` and `plan.md` exist

If either is missing, run the appropriate prior step first.

### 2. Read Context

- Read `spec.md` — user stories, priorities, acceptance scenarios
- Read `plan.md` — phases, technical approach, file structure
- Read `wow-viewer/.specify/memory/constitution.md` — constraints

### 3. Break Down Tasks

For each phase in the plan:

1. List the concrete files that need to be created or modified
2. For each file, write a specific task: what to add/change and where
3. Mark tasks that can run in parallel with `[P]`
4. Mark tasks with their user story: `[US1]`, `[US2]`, etc.
5. Add a checkpoint after each user story is complete

Rules:
- **One concern per task.** If a task touches multiple files, there should be one clear reason.
- **Each task is independently completable.** You should be able to build/test after each task.
- **No task depends on unstated context.** Reference specific files, functions, and line numbers.
- **Max 10 tasks per phase.** If more, split the phase.
- **Include test tasks** when the spec requests them.

### 4. Write tasks.md

Use the template at `wow-viewer/.specify/templates/tasks-template.md` as the structure. Fill in:

- **Phase N: [Name]** — group tasks by user story or concern
- **[ID] [P?] [Story] Description** — format for each task
- **Dependencies & Execution Order** — which phases block which
- **Parallel Opportunities** — tasks that can run concurrently

Write the file to `$featureDir/tasks.md`.

### 5. Validate

- [ ] Every requirement from spec.md has at least one task
- [ ] Every file in plan.md's structure has at least one task
- [ ] Tasks are specific enough to implement without ambiguity
- [ ] Parallel tasks don't touch the same files
- [ ] Checkpoints exist after each user story

## Output

- `wow-viewer/specs/<NNN>-<feature>/tasks.md` — the task breakdown
- Report task count, phase count, and parallel opportunities to the user
