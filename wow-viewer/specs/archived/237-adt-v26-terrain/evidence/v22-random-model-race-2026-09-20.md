# Spec 237 Evidence — v22 Objects Load a Different Random Model Each Run: a Data Race

Date: 2026-09-20

## Operator report

> "there's a bug with the objects that load from v22 files. it loads a random different model every
> time, implying that an index is not being used right"

## The decode is correct — ruled out first

`ACDO` +0x00 `ModelIndex` is a **per-tile** index into that tile's `ADOO` list. Measured on all four
Expansion01 v22 files:

| File | ADOO | ACDO | ModelIndex range | Out of range? |
|---|---|---|---|---|
| `area_24_38.dat` | 21 | 105 | 0…20 | no |
| `area_25_37.dat` | 15 | 685 | 0…14 | no |
| `area_25_38.dat` | 11 | 59 | 0…10 | no |
| `area_26_38.dat` | 0 | 0 | — | n/a |

Every index is in range for its own tile. **The lists genuinely differ between tiles** — `ADOO[1]` is
`TerokkarTreeStump` in `area_24_38` but `TerokkarBush01` in `area_25_37` — so a per-tile index is
mandatory, and every call site uses one:

- `AhdrTerrainAdapter.cs:220,236` — `tile.ModelNames[obj.ModelIndex]` ✔
- `AdtAhdrCommandSupport.cs:74,80` — same ✔
- `DatToLkAdtConverter.cs:292,305` — same ✔

`adt-ahdr objects --list` resolves them correctly and **deterministically**: real names
(`TerokkarTreeLarge.mdx`, `TerokkarBush01.mdx`), plausible positions, stable across runs. So the
reader, the index and the export path are all fine.

## Root cause: unsynchronized read of the shared name table

Tiles load on the **ThreadPool, up to 4 concurrently**
([`TerrainManager.cs:875`](../../../../src/viewer/WoWViewer/Terrain/TerrainManager.cs), gated by
`MaxConcurrentMpqReads = 4`). The adapter *call* is serialized by `_adapterLoadLock`, and
`AhdrTerrainAdapter` serializes its own writes with `_placementLock`.

The **reads were not serialized against those writes.**

`AhdrTerrainAdapter` is unusual: it has no WDT, so it cannot know its model names up front. It builds
the shared table **lazily, during tile loads**, interning each name with `GetOrAddName` and handing
the placement the resulting index. `MdxModelNames` returned the live `List<string>`.

Meanwhile `WorldScene.OnTileLoaded` runs on the main thread (it uploads GPU state), captures
`var mdxNames = adapter.MdxModelNames;`, then does:

```csharp
if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;
string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
```

`List<T>.Add` grows by allocating a new array, copying, assigning `_items`, then writing the element
and incrementing `_size`. A concurrent reader can observe **the new `_size` against the old `_items`**,
or the new array before the element lands — and resolve the placement to a stale name, a different
name, or null. Which one depends on thread interleaving, so it is **a different wrong model on every
run**. That is the reported symptom exactly.

Why v22 shows it and other maps mostly do not: the WDT-backed adapters have their name tables
populated before streaming starts, so there is no concurrent growth to race against. The AHDR adapter
grows its table for the entire duration of the load.

The operator's instinct — "an index is not being used right" — was correct in substance. The index is
computed right; the **table it points into** was being read while it changed shape.

## Fix

[`AhdrTerrainAdapter`](../../../../src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs) now publishes
immutable snapshots:

```csharp
private volatile string[] _mdxNamesPublished = [];
public IReadOnlyList<string> MdxModelNames => _mdxNamesPublished;
```

`GetOrAddName` still appends to the builder list under `_placementLock`, then swaps in a fresh array
**before the new index is returned**, so no reader can ever hold an index past the end of the array it
can see. Readers are lock-free and always see one internally consistent array — either the old one or
the new one. `GetOrAddName` changed from `static` to an instance method to reach the fields.

## Verification

| Action | Result |
|---|---|
| ADOO/ACDO index census, 4 v22 files | every `ModelIndex` in range for its own tile |
| Cross-tile `ADOO[1]` comparison | differs per tile — per-tile indexing confirmed mandatory |
| Every `ModelIndex` call site | all three resolve against `tile.ModelNames` ✔ |
| `adt-ahdr objects --list` | correct, deterministic names and positions |
| Concurrency | `ThreadPool.QueueUserWorkItem`, 4-way semaphore, confirmed in `TerrainManager` |
| `dotnet build WoWViewer.csproj -c Debug` | 0 errors |

## NOT proven

**The visual symptom has not been confirmed fixed** — that needs a viewer run, which is operator-owned.
This is a diagnosis from code inspection plus a confirmed concurrency model, not an observed repro:
the race is real and matches the description, but a second, independent cause cannot be excluded until
someone loads the tiles and watches.

A race is also, by nature, not proven absent by a single clean run. If the wrong model persists after
this change, the next place to look is `WorldAssetManager.NormalizeKey` collisions and the MDX cache,
not the index.

## Related latent issue, not addressed

`AhdrTerrainAdapter` also exposes `MddfPlacements` / `ModfPlacements` as live `List<T>` and dedups by
`_placedUniqueIds`, which is never cleared. On a tile re-entry after eviction, `firstSighting` is false
so the placement is not re-added to the adapter-wide list. `WorldScene.BuildInstances` iterates that
list. Not the reported bug and not touched here.
