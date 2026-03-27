# Codex Workflow Surface

This directory is a Codex-oriented conversion of the workspace's Cursor memory-bank rules and GitHub Copilot workflow assets, rewritten for Codex.

## What maps cleanly

- Root instructions become `AGENTS.md`.
- Reusable workflow playbooks become `.codex/skills/*/SKILL.md`.
- Focused reusable prompt templates become `.codex/prompts/*.md`.

## What does not map 1:1

- In this workspace, Codex auto-discovers `AGENTS.md`, but repo-local prompts and skills are still documentation assets unless the host explicitly wires them into tools.
- The original Cursor `memory-bank.mdc` behavior is converted into written guidance, not a separate runtime rule engine.
- The original prompt frontmatter was retained where useful, but names and references were rewritten so the Codex copies read naturally in this workspace.

## Conversion sources

- `.github/copilot-instructions.md`
- `.github/prompts/*`
- `.github/skills/*/SKILL.md`
- `gillijimproject_refactor/memory-bank/*`
- `parpToolbox/.rules/memory-bank.mdc`
- `PM4Tool/.cursor/rules/memory-bank.mdc`

## Working rule

- Keep `.codex` docs, `AGENTS.md`, `wow-viewer/README.md`, the relevant memory-bank files, and the relevant plan files synchronized when the workflow materially changes.
