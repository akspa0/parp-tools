# Quickstart: M2 Reader Era Parity

Commands verified against the real tool surface on 2026-08-15. Substitute your own configured client
root — no client path belongs in source or portable configuration.

## Reproduce the baseline

These four reads produce the measured table in `research.md`. Run them before changing anything; they
are the before-state every phase is diffed against.

```powershell
$ROOT053 = "<your client library>\Vanilla\0.x\0_5_3_3368\World of Warcraft"
$ROOT200 = "<your client library>\TBC\2.X_Pre-Release_Windows_enUS_2.0.0.5610\World of Warcraft"
$ROOT301 = "<your client library>\Wrath\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft"
$ROOT330 = "<your client library>\Wrath\3.X_Retail_Windows_enUS_3.3.0.10958\World of Warcraft"

# Works today: 54 bones, 106 sequences
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  mdx inspect --archive-root $ROOT053 --virtual-path "Creature\HighElf\HighElfMale_Warrior.mdx"

# Broken: bones=0, geometry fails at bone index 10
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  m2 inspect --archive-root $ROOT200 --virtual-path "CHARACTER\BloodElf\Male\BloodElfMale.m2"

# Refused: unhandled "2.x TBC era, not yet supported"
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  m2 inspect --archive-root $ROOT301 --virtual-path "CHARACTER\BloodElf\Male\BloodElfMale.M2"

# The reference: 151 bones, 155 sequences, geometry available
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  m2 inspect --archive-root $ROOT330 --virtual-path "CHARACTER\BloodElf\Male\BloodElfMale.M2"
```

## Find models in a build

Enumeration goes through the archive tooling, not the filesystem.

```powershell
# Build the listfile cache once per build
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  archive build-listfile-cache --archive-root $ROOT330 --cache-key 3.3.0.10958

# Then query the cache under output/cache/archive-listfiles/<cache-key>.json
```

**Point `--archive-root` at the directory containing `Data\`.** Pointing one level too high builds a
cache with zero entries and reports success — a silent failure worth knowing about.

For `MDLX` builds, models can also be enumerated directly:

```powershell
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  mdx chunk-carriers --chunks SEQS,BONE --archive-root $ROOT053 --path-filter elf --limit 200
```

## Model set

The same models across builds, so results are comparable:

| Purpose | 0.5.3 | Later builds |
|---|---|---|
| Subject | `Creature\HighElf\HighElfMale_Warrior.mdx` | `CHARACTER\BloodElf\Male\BloodElfMale.m2` |
| Control | `Character\NightElf\Male\NightElfMale.mdx` | `Character\NightElf\Male\NightElfMale.m2` |
| Control | — | `CHARACTER\HUMAN\MALE\HumanMale.m2` |

0.5.3 also carries seven more High Elf models — `HighElf{Male,Female}_{Hunter,Mage,Priest,Warrior}`.

**Controls are not optional.** If Blood Elf matches High Elf but also matches Night Elf and Human,
what has been shown is that humanoid rigs share a base — not that Blood Elves descend from High Elves.

## Regression check

Run at every phase exit gate.

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test  I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

The core suite carries **9 known pre-existing failures** unrelated to this work. Compare the failure
**set**, not the count — a changed set with an unchanged count is still a regression. To capture a
baseline for diffing, stash the working tree first (**include untracked files**, or new tests are left
behind and the baseline build fails), run, then restore.

## Scope ceiling

Nothing at or beyond **4.0.1** is read, surveyed, or referenced by this work.
