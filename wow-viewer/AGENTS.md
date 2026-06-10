<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan.
Spec Kit skills are available as opencode skills under `.opencode/skills/speckit-*`.
<!-- SPECKIT END -->

# wow-viewer Guardrails

The rules that keep new work in this repository shippable, findable, and safe to extend.

For the day-to-day how-to (C#/Python conventions, project layout, tests, commits), see `wow-viewer/memory-bank/coding_standards.md`. For paths and override environment variables, see `wow-viewer/memory-bank/data-paths.md`.

## Scope and Boundaries

- All new implementation work lives in `wow-viewer/`. Do not add new code in `gillijimproject_refactor/` unless explicitly requested.
- The top-level `../AGENTS.md` is the authoritative workspace policy for scope, safety, and repo boundaries. When guidance here and there conflict, the top-level file wins.
- Game client data is read only from `output/tmp/wowarchive-clients/`. Any reference to `H:\CLIENTS` in code, scripts, tests, or documentation is a bug. Replace it with a staged path or remove the reference.

## Spec Kit

Every non-trivial feature slice starts with a spec, then a plan, then tasks. Use the Spec Kit skills under `.opencode/skills/speckit-*`:

- `speckit-specify` — write or refine a spec (`wow-viewer/specs/<NNN>-<feature>/spec.md`)
- `speckit-plan` — implementation plan (`plan.md`)
- `speckit-tasks` — concrete task breakdown (`tasks.md`)
- `speckit-implement` — execute the task list with validation
- `speckit-analyze` — stress-test a spec before committing to it
- `speckit-checklist` — verify spec/plan/implementation alignment

Specs live in `wow-viewer/specs/<NNN>-<feature>/`. The Spec Kit constitution is at `wow-viewer/.specify/memory/constitution.md`.

Architecture docs live in `wow-viewer/docs/architecture/` and must stay aligned with the code that implements them. When a spec changes behavior, update the relevant architecture doc in the same change.

## Phasing and Validation

A spec is a sequence of phases. Do not start phase N+1 until phase N is validated against real data. "Validated" means a real-data proof has been recorded, not just that the code compiles.

Each phase produces a small, independently committable diff. Big-bang commits and big-bang rewrites are out.

## Implementation Pass Checklist

A complete pass on a spec task, plan slice, or feature lands four artifacts: code, documentation, a commit, and a self-review.

**Code and tests**

- The diff addresses the task, the plan, and the spec. Nothing extra.
- Null references, missing enum cases, file-locking, coordinate-transform, and cache-version regressions are caught before the commit, not after.
- `dotnet build` and `dotnet test` pass on the affected projects.
- No path in the diff resolves to `H:\CLIENTS` or any other untrusted client root.

**Documentation**

- `tasks.md` is current: completed items are checked off, new discoveries are added, counts are accurate.
- `memory-bank/activeContext.md` and `memory-bank/progress.md` reflect what landed, what is open, and the biggest unproven gap. Compress aggressively — a 20-line accurate summary beats a 200-line log.
- `spec.md` is updated when a discovery changes an assumption or requirement.

**Commit**

- One logical change per commit. A bug fix, a refactor, and a feature do not belong together.
- Staged by name. `git add .` is a smell.
- Commit message describes the "why," not the "what." The diff already shows the what.
- No secrets, no private paths, no `H:\CLIENTS` references.

**Self-review**

- Re-read the diff as a new contributor would. Would they understand the change without a conversation?
- If a cleaner, safer, or faster version is obvious, either make it now or add it as a follow-up task. Do not leave a known-uglier version in place to ship on schedule.
- The final state compiles and tests pass.

This four-part checklist is the difference between a one-day change and a one-week debug session. Skipping any of the four parts is how small issues become big ones.
