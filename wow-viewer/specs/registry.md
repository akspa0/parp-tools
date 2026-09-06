# Spec Routing Registry

This is the cold classification companion to [STATUS.md](STATUS.md). `STATUS.md` stays small and
is the only fresh-chat status entry. This registry prevents an old directory from becoming active
merely because it still exists.

## Default lanes

| Class | Specs | How to use |
|---|---|---|
| Current owner | 224, 227, 228, and 223 residual operator gate | Load only when the matching task is selected. |
| Near-term planned | 226, 229, 230 | Select one from `STATUS.md`; do not preload all three. |
| Queued active | 197, 221, 222 | Preserve their evidence; reopen only by explicit task selection. |
| Cold watch list | Every non-archived spec not named above, including all epic members | Not default context and not implicitly complete or canceled. A user request or a current-owner link is required to revive it. |

## Superseded archive

| Spec | Location | Active successor |
|---|---|---|
| 080 UI consolidation | `archived/superseded/080-wow-ui-consolidation/` | 227 and 229 |
| 145 UI overhaul | `archived/superseded/145-wow-ui-overhaul/` | 227 and 229 |
| 195 overhead chunk manipulator | `archived/superseded/195-overhead-chunk-manipulator/` | 219 and 222 |

## Rules

- A cold or archived record is preserved history, not implementation authority.
- Do not archive a cold spec as complete without task receipts and the required operator gate.
- When a new plan supersedes a cold record, add one forward pointer here and retain the old record
  under `archived/` only after a bounded link audit.
- Detailed epic membership remains in `epics/active-epics.md`; it is an on-demand watch list, not
  the default handoff.
