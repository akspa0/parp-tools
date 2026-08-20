# Epic: Editor Platform — plugin host, bridge, and the first editors

**Status**: specs complete, implementation not started
**Created**: 2026-08-19
**Member specs**: [166](../166-editor-plugin-host/spec.md) · [167](../167-editor-runtime-bridge/spec.md) ·
[168](../168-editor-session-undo/spec.md) · [169](../169-chunk-clipboard-plugin/spec.md) ·
[170](../170-dbc-table-browser/spec.md) · [171](../171-dbc-table-editing/spec.md) ·
[172](../172-editor-edit-journal/spec.md) · [173](../173-asset-integrity-gate/spec.md) ·
[174](../174-asset-repair-patterns/spec.md) · [175](../175-placement-authoring/spec.md) ·
[176](../176-object-transfer/spec.md) · [177](../177-adt-tile-creation/spec.md) ·
[178](../178-mcp-automation-surface/spec.md)

**Background** (superseded drafts, kept for rationale — **not** requirements):
[`background/`](background/) holds the four large specs this epic replaces. Their Context sections
carry the evidence; their requirements now live in the member specs above.

---

## Read this first if you are starting cold

The goal is to turn the viewer into an **Editor**: a plugin host where each editing capability is a
self-contained plugin, rather than another partial-class page bolted onto one god object.

**Four measured facts that shaped every spec here.** Do not re-derive these; they were verified
against the working tree on 2026-08-19.

### 1. The god object is the binding constraint

`ViewerApp.cs` is **15,670 lines** across 24 partial files. `WorldScene.cs` is **15,733**. Adding an
editor today means a 25th partial file and another block of `_prefixed` fields. That cost — not
format knowledge — is what blocks "a full-featured, all-versions WoW editor."

### 2. The authoring code already exists and has no UI caller

**Do not write ADT/WDT writers. They exist, they are correct, and they are library-first.**

| Capability | Lines | Production callers |
|---|---|---|
| `LkAdtWriter` | 620 | CLI only |
| `AlphaWdtWriter` | 1,314 | CLI only |
| `AlphaToLkConverter` | 667 | CLI only |
| `LkToAlphaConverter` | 784 | CLI only |
| **`AdtPlacementWriter`** | 200 | **None.** Sole caller is its own unit test |

3,585 lines of correct authoring code reachable only from a command line. Meanwhile `ViewerApp.cs`
carries **112 references** to `_stagedPlacementEdits`/`_selectedPlacement*` — a second,
translation-only staging implementation, built on the app object because nothing connected a live
scene selection to the core writer.

**The gap is the bridge, not the write code.** The work is wiring, not relocation.

### 3. The DBC path is already plumbed

DBCD and WoWDBDefs (**1,320 definitions**) are vendored, referenced by both `WowViewer.Core.IO` and
the viewer, and copied to build output. `ArchiveReaderDbcProvider` already bridges DBCD to the repo's
archive boundary. `DBCDStorage.Save(string)` exists. **The DBC editor is not blocked on parsing.**

### 4. The same mistake has been made three times

Three separate capabilities in this repo are implemented twice because nothing owned them:

| Surface | Implementation A | Implementation B |
|---|---|---|
| Placement staging | `AdtPlacementWriter` (unused) | `ViewerApp._stagedPlacementEdits` (112 refs) |
| MPQ patch priority | `MpqArchiveCatalog` (complete) | `NativeMpqService` (weaker, used by the builder) |
| Zarr codec default | `zarr_io.py` zstd-5 | `v25/dataset.py` lz4-1 |

This is what a missing owner looks like. Constitution II exists for it. **Every member spec here
either establishes an owner or consumes one — none may add a third implementation of anything.**

### The mistake that must not be repeated

The first draft of the MCP spec made MCP a *peer* of the UI — "one operation contract, three drivers"
— and required every Editor operation to be MCP-invocable. That put a standing obligation on every
future plugin author, and it had already leaked backwards: the host spec justified modeling
operations as data with *"so the MCP server can drive them."*

**External consumers never motivate core design decisions.** Operations are data because a shared
cross-plugin undo history needs the host to reverse an operation without knowing which plugin made
it. That is the whole reason. Spec 178 is a strictly downstream adapter and gets no vote.

---

## Dependency chain

```
166 host ──┬── 167 bridge ──┬── 169 chunk clipboard
           │                ├── 175 placement authoring ── 176 object transfer ── 177 tile creation
           │                └── 172 edit journal
           ├── 168 session/undo ── 172 edit journal
           ├── 170 DBC browser ── 171 DBC editing
           └── 173 integrity gate ── 174 repair patterns

178 MCP (external, optional) ── consumes whatever exists; requires nothing
```

## Recommended order

| # | Spec | Why here |
|---|---|---|
| 1 | **166** host | Nothing else exists without it |
| 2 | **167** bridge | The half that makes it an *editor*; 169/175 both need it |
| 3 | **169** chunk clipboard | **The contract's only real validation** — see below |
| 4 | **168** session/undo | Before plugin #3 grows its own undo stack |
| 5 | **170** → **171** DBC | The visible payoff; independent of the world-editing chain |
| 6 | **173** → **172** | Integrity before journaling: don't durably persist a path to corrupt output |
| 7 | **175** → **176** → **177** | World authoring, in increasing blast radius |
| 8 | **174** repair | Driven by the census from 173, not by guesses |
| 9 | **178** MCP | Last. Optional. Delete-able. |

**Why chunk clipboard is spec #3 and not later**: a host contract validated only against code written
to fit it is not validated. The chunk clipboard predates any contract — app-object state, direct
keyboard reads, renderer-owned terrain mutation, its own dirty set. If the contract absorbs that with
byte-identical output, it will absorb editor #7. If it cannot, the contract is wrong and we learn it
in spec 169 rather than in spec 190.

## Measured baselines

These are the numbers member specs assert against. Re-measure before claiming a change.

| Metric | Today | Target |
|---|---|---|
| `_chunkClipboard*`/`_selectedChunks` refs in `ViewerApp.cs` | 124 (18 fields) | 0 |
| `_stagedPlacementEdits`/`_selectedPlacement*` refs | 112 | 0 |
| `AdtPlacementWriter` production callers | 0 | 1 (the only placement write path) |
| ADT/WDT/placement serializers | *n* | *n* — this epic adds **zero** |
| Assets in `H:\CLIENTS\WoW335\modernwow\` crashing the viewer | many (census required) | 0 crashes; every one a *named verdict* |

## Validation corpus

`H:\CLIENTS\WoW335\modernwow\` — a 3.3.5 client full of **"fuckported"** assets (re-fitted backwards
from a later client; lossy by necessity) that crash the viewer today. Its root contains
`modernwow.noggitproj`: it is literally a Noggit project, made by the tool whose failure modes this
epic exists to avoid.

**The bar is a named verdict per asset, not survival.** A crash converted into a silent skip is a
regression, not a fix.

## Hard constraints (violating these fails the spec, not just review)

1. **Runtime never references Editor.** One-way dependency, enforced by a build/test check, not
   convention. Removing every Editor project must leave a working viewer.
2. **No Blizzard containers as output** (Constitution VII). MPQ/CASC are read-only inputs, forever.
   Client *content* formats (ADT/WMO/M2/BLP/DBC) are written directly as loose files — that is the
   intended output and not a violation.
3. **Never write into a game install.** Output directory only.
4. **Library-first** (Constitution II). Host and plugin logic in `src/core/`; the viewer app is a
   thin host shell.
5. **Real-data validation** (Constitution III) against `H:\CLIENTS`, with commands, build identity,
   and hashes recorded.

## Tracking

| # | Spec | Status | Gate |
|---|---|---|---|
| 166 | Editor plugin host | Draft | A reference plugin appears via one registration line, zero other edits |
| 167 | Editor↔Runtime bridge | Draft | Viewer builds and runs with Editor removed; move-object round-trips |
| 168 | Session, undo, dirty state | Draft | Undo reverses across two plugins in one history |
| 169 | Chunk clipboard plugin | Draft | Byte-identical output; 124 refs → 0 |
| 170 | DBC table browser | Draft | Typed columns from real defs on two build eras |
| 171 | DBC table editing | Draft | Unmodified load→save is byte-identical |
| 172 | Edit journal (Zarr) | Draft | 10 process kills lose no completed operation |
| 173 | Asset integrity gate | Draft | modernwow census published; zero crashes |
| 174 | Asset repair patterns | Draft | Every repair passes the validation its input failed |
| 175 | Placement authoring | Draft | Unedited chunks byte-identical after save |
| 176 | Object transfer | Draft | Cross-era transfer round-trips or is refused |
| 177 | ADT tile creation | Draft | New tile loads in an independent tool |
| 178 | MCP automation surface | Draft | Deleting it leaves the Editor complete |
