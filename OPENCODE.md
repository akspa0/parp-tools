# OpenCode Workspace Instructions

This workspace is set up so `Codex`, `GitHub Copilot`, and `OpenCode` can share one instruction surface instead of drifting into separate workflows.

## Canonical Files

- Use `AGENTS.md` as the primary always-on repo instruction file.
- Use `.codex/prompts/` as the reusable prompt library.
- Use `.codex/skills/` as the reusable skill or playbook library.
- Use `.codex/README.md`, `.codex/prompts/README.md`, and any skill index files as the registry for those assets.

## Compatibility Rule

- If a tool understands `AGENTS.md`, treat it as the source of truth.
- If a tool wants prompt or skill libraries, prefer the `.codex/` copies because they were normalized from the original Copilot assets for cross-tool use.
- Keep `AGENTS.md`, `.codex/*`, and any future tool-specific wrappers aligned when workflow guidance changes.

## Source Provenance

- Original Copilot-specific assets remain under `.github/`.
- Cross-tool normalized assets live under `.codex/`.
- OpenCode should treat `.codex/` as the compatible shared workflow surface unless a future `.opencode/` file explicitly overrides it.
