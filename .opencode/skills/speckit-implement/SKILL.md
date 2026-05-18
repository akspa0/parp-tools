---
name: speckit-implement
description: Execute tasks from a Spec Kit task breakdown with validation. Use when the user says "implement", "$speckit-implement", or wants to start coding from the tasks.md.
---

# Speckit Implement

Execute tasks from an existing tasks.md file, one phase at a time, with validation.

## When to Use

- User says "implement", "$speckit-implement", or "start implementing"
- A tasks.md exists for the current feature
- Ready to write code

## Rules

### ONE PHASE AT A TIME

- Only work on the current phase. Do not start the next phase early.
- After completing a phase, run the phase checkpoint before moving on.
- If validation fails, fix it before continuing. Do not skip validation.

### EVIDENCE-BASED COMPLETION

- Every task is complete when the code is written AND validated.
- "I wrote the code" is not complete. "I wrote the code and it builds/tests/passes" is complete.

## Workflow

### 1. Locate the Feature

- Find `wow-viewer/specs/<NNN>-<feature>/tasks.md`
- Read `spec.md` and `plan.md` for context

### 2. Read the Current Phase

- Identify the next incomplete phase in tasks.md
- Read all tasks in that phase
- Check which tasks are already done (files exist, code is written)

### 3. Execute Tasks

For each task in the current phase:

1. **Before**: Read the files the task will touch. Understand existing code conventions.
2. **Implement**: Write the code. Follow existing patterns in the codebase.
3. **After**: Verify the task is done:
   - Code compiles: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
   - Tests pass: `dotnet test wow-viewer/WowViewer.slnx -c Debug`
   - For Python: `cd wow-viewer/data-harvester && uv run python -c "import ..."`
4. **Update**: Mark the task as done in tasks.md (check the box `[x]`)

### 4. Phase Checkpoint

After all tasks in a phase are complete:
- Run the full build
- Run the full test suite
- Verify the phase goal from plan.md is met
- Report completion to the user

### 5. Move to Next Phase

Only after the checkpoint passes:
- Mark the phase as complete
- Start the next phase

## Build Commands

```powershell
# C# build
dotnet build wow-viewer/WowViewer.slnx -c Debug

# C# tests
dotnet test wow-viewer/WowViewer.slnx -c Debug

# Python validation
cd wow-viewer/data-harvester
uv run python scripts/validate_v16_training_ready.py --build <build>
```

## Output

- Code changes in `wow-viewer/src/`, `wow-viewer/tools/`, `wow-viewer/data-harvester/`
- Updated `tasks.md` with completed checkboxes
- Build/test evidence after each phase
