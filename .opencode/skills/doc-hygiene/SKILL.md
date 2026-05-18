---
name: doc-hygiene
description: Use when starting a new task, finishing a task, or when documentation, plans, or memory bank may be stale. Triggers doc sync, plan chunking, and memory-bank compression checks.
---

# Documentation Hygiene Skill

## When to Use

- At the **start** of any non-trivial implementation task
- At the **end** of any task that changed code, data, or workflow
- When the user mentions plans, specs, or memory bank
- When switching between tasks or sessions

## Rules

### 1. Plans Must Be Bite-Sized

Every plan must be broken down into steps small enough for ANY LLM model to implement in a single focused pass. This means:

- **One concern per step.** A step that touches files A, B, and C should have a clear single reason (e.g., "add liquid mask head to model, dataset, and training script" — one concern, three files).
- **Each step is independently validatable.** You should be able to build/test/verify after each step without needing the next step.
- **No step depends on unstated context.** If a step requires reading a doc or understanding a prior decision, state it in the step.
- **Max 10 steps per phase.** If a phase has more than 10 steps, split it into sub-phases.

When writing new plans, put them in `docs/architecture/` or `docs/plans/` as markdown files. Use this structure:

```
docs/
  architecture/
    <feature-name>-plan-<date>.md    # top-level plan with phases
    <feature-name>-phase-N.md         # phase-level detail (if needed)
  plans/                               # if project uses this convention
```

### 2. Spec Docs Are the Source of Truth

- Every model, dataset, training run, pipeline, and public interface has a spec doc.
- Spec docs live in `docs/architecture/` as markdown.
- **When you change code that a spec doc describes, you MUST update the spec doc in the same commit or immediately after.**
- If no spec exists for something you're building, create one before implementing.

### 3. Memory Bank Must Stay Compressed and Current

- The memory bank (`memory-bank/`) is the session continuity source. If it rots, every new session starts blind.
- **At the end of every non-trivial implementation session**, update the relevant memory bank files:
  - `activeContext.md` — what's in progress, what's blocked, what changed
  - `progress.md` — what was completed, what's next
- **Compress aggressively.** Memory bank files should be concise. Remove stale entries. Prefer a 20-line accurate summary over a 200-line chronological log.
- **Known bug: memory bank does NOT auto-update when code edits are made.** This is a manual discipline. If you made code changes, you must explicitly update the memory bank before ending the session. No exceptions.

### 4. Commit Hygiene for Docs

- Doc updates (spec, memory-bank, plans) can be bundled with their corresponding code changes.
- Do NOT commit code changes without updating the relevant spec or spec section if one exists.
- If a spec doc doesn't exist yet and you're adding new behavior, create it.

## Checklist (Run Mentally Before Every Task)

1. **Before starting:** Load `speckit-checklist` (or `speckit-analyze` if no spec exists). Read `wow-viewer/specs/` for the active feature. Read the relevant spec/doc/memory-bank file for the area you're about to touch.
2. **During implementation:** If the scope expands beyond what the spec covers, note the delta. Update the spec after or ask the user.
3. **After completing:** Update the relevant memory-bank files. Compress if they've grown beyond 200 lines. Update the spec doc if behavior changed.
4. **Before committing:** Run `speckit-checklist` to verify spec/plan/implementation alignment. Verify memory-bank reflects what actually happened.