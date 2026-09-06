# Documentation Router — wow-viewer

This is the canonical entry point for written documentation. It is intentionally a routing page,
not a project history or second task registry.

## Fresh-chat implementation route

Repository rules in [AGENTS.md](../../AGENTS.md) are the harness policy. For wow-viewer work, then
follow the required small route:

1. [Current spec status](../specs/STATUS.md) — choose one active owner only.
2. [Active context](../memory-bank/activeContext.md) — the compact current handoff.
3. The selected spec's `spec.md`, `plan.md`, and `tasks.md`, then only its linked evidence.

`specs/STATUS.md` and the selected SpecKit directory are the implementation source of truth.
`memory-bank/activeContext.md` is the only default continuity document. This router is consulted
for operator/reference documentation, not inserted into the required implementation route. Do not
load an epic, architecture note, archive, or historical plan unless the selected spec links it.

## Operator documentation

- [Viewer user guide](WoWViewer/USERGUIDE.md) — current documented app routes; real-client
  acceptance remains separately recorded by its owning spec.
- [Viewer guide index](WoWViewer/README.md)
- [CLI tools](CLI-TOOLS.md)
- [Data harvester](../data-harvester/README.md)
- [Release notes](releases/)

## Reference only — never default context

- `architecture/` — durable technical research. A file is consulted only through a live spec or
  direct investigation question.
- `research/` and `wowdev-wiki/` — source research and working notes, not implementation plans.
- `audits/` and `validation/` — dated evidence, not current status.
- `archive/` — superseded plans, historical audits, and legacy payloads retained intact for
  provenance. It does not define current behavior or scope.

## Canonical ownership

| Need | Authoritative location |
|---|---|
| Repository policy and safety | `AGENTS.md` |
| Current work selection | `specs/STATUS.md` |
| Feature requirements, design, tasks, receipts | `specs/<owner>/` |
| Fresh-chat handoff | `memory-bank/activeContext.md` |
| Session ledger | `memory-bank/progress.md` |
| Operator usage | `docs/WoWViewer/USERGUIDE.md` |
| Historical evidence | `docs/archive/`, `specs/archived/`, `memory-bank/archive/` |

Legacy route pages [DOCUMENTATION-STATUS.md](DOCUMENTATION-STATUS.md) and
[PLANS-OVERVIEW.md](PLANS-OVERVIEW.md) remain as short redirects so old links continue to work.
