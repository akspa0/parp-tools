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
