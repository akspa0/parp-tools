---
name: speckit-cleanup
description: Monthly audit of wow-viewer Spec Kit artifacts against real code. Verifies checked tasks have receipts and named symbols exist in code, archives implemented/closed specs, compresses memory banks, and updates the cleanup ledger in AGENTS.md. Use when the operator invokes "speckit cleanup", at the first session of each month, or when session context is clogged with stale spec detail.
---

# speckit-cleanup

Audit Spec Kit artifacts against the real code and compress context. Mechanical checks only —
never rationalize a failed check. All work happens in `wow-viewer/`.

## When to run

- First session of each month (check the ledger in `AGENTS.md` §9.5).
- On explicit operator request.
- The ledger says "Last cleanup" is more than ~5 weeks old.

## Procedure

1. **Read the ledger.** Open repo-root `AGENTS.md`, find `## 9. Governance` → `Cleanup ledger`.
   Record `Last cleanup` for the report.

2. **Audit every active spec** listed in `wow-viewer/specs/STATUS.md`:
   For each `tasks.md` checkbox marked `[x]`:
   - **Receipt check**: does the task (inline or the spec's `evidence/` dir) state files changed,
     verification commands + exit status, and a criterion→evidence table with real output? If not,
     un-check it (`[ ]`) and note it.
   - **Symbol check**: do the task's named files/classes/methods exist in the tree? Use
     `Get-ChildItem -Recurse -Filter` / `Select-String`. A checked task whose symbols are absent
     is un-checked and noted as "phantom".
   - **Staleness check**: if code in the spec's area was modified more recently than the spec's
     files without a dated amendment note, flag it for spec-sync (do not silently rewrite history).

3. **Archive closed specs.** A spec is closed when all tasks are checked with receipts AND the
   operator gate passed (or the operator closed it). Move its directory to
   `wow-viewer/specs/archive/<spec-id>/` and move its `STATUS.md` row into an `## Archived` table.

4. **Compress memory banks.**
   - `memory-bank/activeContext.md`: keep only the current lane, next bounded action, proof owner,
     constraints, and handoff. Move anything older to `memory-bank/archive/` and index it in
     `archive/README.md`.
   - `memory-bank/progress.md`: keep entries from the current month; move older entries to
     `archive/<date>-progress-detail.md`.

5. **Update the ledger** in `AGENTS.md` §9.5:
   - `Last cleanup: <today's date> (<n> specs audited, <n> tasks un-checked, <n> specs archived)`
   - `Next cleanup due: <first of next month>`

6. **Write the report** to `wow-viewer/specs/224-speckit-governance/evidence/cleanup-<date>.md`
   containing: specs audited, tasks un-checked (with reason), specs archived, memory-bank lines
   moved, and any spec-sync flags. This report is the cleanup's own receipt (Spec 224 FR-2).

## Rules

- Never delete history: archive, don't destroy.
- Never mark anything complete during cleanup; only un-check or flag.
- Preserve unrelated dirty worktree changes; stage nothing unless the operator asked for a commit.
- If a spec's state is ambiguous, leave it active and list it under "needs operator decision" in
  the report.
