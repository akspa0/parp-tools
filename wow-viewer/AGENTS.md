<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan.
Spec Kit skills are available as opencode skills under `.opencode/skills/speckit-*`.
<!-- SPECKIT END -->

# wow-viewer Guardrails

- Treat `../AGENTS.md` as the authoritative workspace policy for scope, safety, and repo boundaries.
- Keep all new implementation work in `wow-viewer/`; do not write new code in `gillijimproject_refactor/` unless explicitly requested.
- Use Spec Kit skills for non-trivial feature slices:
  - `speckit-specify` -> write or refine spec (`wow-viewer/specs/<NNN>-<feature>/spec.md`)
  - `speckit-plan` -> implementation plan (`wow-viewer/specs/<NNN>-<feature>/plan.md`)
  - `speckit-tasks` -> concrete task breakdown (`wow-viewer/specs/<NNN>-<feature>/tasks.md`)
  - `speckit-implement` -> execute tasks with validation
  - `speckit-analyze` -> analyze spec for completeness and risks
  - `speckit-checklist` -> verify spec/plan/implementation alignment
- Specs live in `wow-viewer/specs/<NNN>-<feature>/`
- Architecture docs live in `wow-viewer/docs/architecture/` and must stay aligned with code changes.
- The Spec Kit constitution is at `wow-viewer/.specify/memory/constitution.md`.

## Mandatory Review & Rework Process

After EVERY implementation pass of a Speckit task, plan slice, or feature, the following steps MUST be completed before moving to the next task:

### Step 1: Review for Bugs and Regressions
- Read the diff of every changed file (`git diff`)
- Check for: null reference risks, missing enum cases in switch statements, file-locking build issues, incorrect coordinate transforms, stale cache versions
- Run `dotnet build` on all affected projects
- Run `dotnet test` on affected test projects
- Check that the implementation doesn't violate the spec's "explicitly out of scope" section
- Verify that no hardcoded paths use `H:\CLIENTS`

### Step 2: Update Documentation
- Update `tasks.md` — mark completed tasks `[x]`, add new tasks discovered during implementation, update task counts
- Update `memory-bank/activeContext.md` — task counts, what's done, what's pending, biggest gaps
- Update `memory-bank/progress.md` — add a dated entry describing what landed, the proof, and remaining gaps
- Update `spec.md` if new discoveries change the spec's assumptions or requirements

### Step 3: Commit
- Stage only intended files (`git add` specific paths, not wildcards)
- Write a concise commit message describing the "why" not just the "what"
- Verify no secrets or private paths are committed

### Step 4: Multi-Pass Self-Review
Before declaring a task complete, simulate at least one rework pass:
1. Read the implementation as if you're a new developer seeing it for the first time
2. Identify at least one thing that could be done more cleanly, safely, or efficiently
3. Either fix it now or create a follow-up task
4. Verify the fix compiles and tests still pass

This process prevents the "weeks of debugging" cycle by catching issues at implementation time rather than after they've accumulated.
