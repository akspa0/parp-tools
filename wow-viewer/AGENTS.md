<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

# wow-viewer Guardrails

- Treat `../AGENTS.md` as the authoritative workspace policy for scope, safety, and repo boundaries.
- Keep all new implementation work in `wow-viewer/`; do not write new code in `gillijimproject_refactor/` unless explicitly requested.
- Use Spec Kit skills for non-trivial feature slices:
  - `$speckit-specify` -> write or refine spec
  - `$speckit-plan` -> implementation plan
  - `$speckit-tasks` -> concrete task breakdown
  - `$speckit-implement` -> execute with validation
- Keep architecture docs in `wow-viewer/docs/architecture/` aligned with code changes.
