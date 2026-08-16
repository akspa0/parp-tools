# Quickstart: Asset Reference Inventory

Substitute your own configured client root — no client path belongs in source or portable
configuration.

## What exists today

Verified against the real tool surface on 2026-08-16. These are the readers and commands the feature
builds on; none of them is new work.

```powershell
$ROOT053 = "<your client library>\Vanilla\0.x\0_5_3_3368\World of Warcraft"

# World object inspection — currently narrow, no doodad or texture dump, no batch mode
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  wmo inspect --archive-root $ROOT053 --virtual-path "<world/...wmo>"

# Model inspection and full export
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  mdx inspect --archive-root $ROOT053 --virtual-path "Creature\HighElf\HighElfMale_Warrior.mdx"

# Corpus-wide scanning already has a working precedent — note --path-filter and --limit
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  mdx chunk-carriers --chunks TEXS --archive-root $ROOT053 --path-filter world --limit 200

# The catalogued set (one of the three sets being compared)
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  archive build-listfile-cache --archive-root $ROOT053 --cache-key 0.5.3.3368
```

**Point `--archive-root` at the directory containing `Data\`.** One level too high builds a cache with
zero entries and reports success.

## The trap this feature is built around

The listfile index is the **catalogued** set. It is *not* the corpus.

For the earliest staged build it names **one** world object; the build contains **532**, stored as
per-asset containers under the loose `World` tree. Enumerating the corpus from that index would sweep
one object, find almost no references, and report almost nothing missing — while looking like a clean
result.

Corpus enumeration comes from the archive access layer. The listfile stays, in its correct role, as one
side of the comparison.

## Reference kinds swept

| Source | Reference kind |
|---|---|
| World object | Placed doodads |
| World object | Material textures |
| Model | Textures |

## Sanity read, not a target

The engine paints untextured geometry neon green in at least every pre-alpha and beta Vanilla build, so
a missing texture shows up in-world. The Mt. Hyjal effect objects are the known example — green smoke
on the mountainside, confirmed by explorers after Classic launched in 2018.

They are useful **after** a sweep, as a spot-check that coverage is sane. They are not a target and no
phase is gated on them. The whole point is the population nobody has ever counted; a sweep's headline
number is how many references resolve to nothing.

## Model route readability

Model sweeping depends on the build's format route:

| Route | Status |
|---|---|
| Alpha `MDLX` | Reads today — 5,545 models in the earliest build |
| `MD20 0x100`–`0x107` | **Blocked** pending Spec 154 |
| `MD20 0x108` | Reads today |

A blocked build must be reported as blocked. `modelsExamined: 0` without a block record beside it reads
as "no missing textures" and means "never checked".

## Regression check

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test  I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

The core suite carries **9 known pre-existing failures** unrelated to this work. Compare the failure
**set**, not the count. To baseline for diffing, stash the working tree **including untracked files** —
otherwise new tests are left behind and the baseline build fails.

## Scope

Sound and other asset classes are out. `uniqueId` is out as a chronology source — it dates placements,
not assets. Repair never runs unless explicitly requested, and a missing asset is a finding, not
necessarily a defect: some absences are deliberate, which is exactly why the control objects are
famous.
