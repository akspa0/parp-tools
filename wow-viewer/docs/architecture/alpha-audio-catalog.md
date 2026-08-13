# Alpha audio catalog

## One-sentence definition

The Alpha audio catalog is a build-aware lookup that answers: **“For this Alpha
area, which day/night and underwater MIDI ambience records does the client
reference, and can the referenced MIDI sequences and DLS instrument bank be
found?”**

It is an inspection and asset-resolution layer. It is not an audio player and
it is not a catalog of every sound file in the client.

## Why it exists

The earliest clients do not use the later `SoundKit`/FMOD-style lookup chain
for this part of world audio. Their area ambience is split across two DBC
tables:

| Source | What the catalog reads |
|---|---|
| `AreaTable.dbc` | Area ID, continent/parent area, display name, `MIDIAmbience`, and `MIDIAmbienceUnderwater` references. It also preserves `ZoneMusic`, `IntroSound`, and `IntroPriority` when those fields exist in the selected build schema. |
| `AreaMIDIAmbiences.dbc` | The ambience record keyed by those references: day sequence path, night sequence path, DLS bank path, and volume. |

The catalog joins the integer reference in `AreaTable.dbc` to the matching
record in `AreaMIDIAmbiences.dbc`. The integer is not itself a filename.

For example, the conceptual lookup is:

```text
area  ->  MIDIAmbience ID  ->  day/night .mid sequence + .dls bank + volume
      ->  MIDIAmbienceUnderwater ID  ->  underwater day/night .mid + .dls
```

The exact field layout is selected through the build-specific WoWDBDefs/DBD
definitions. The reader does not use a universal hardcoded byte layout.

## What “catalog” means here

There are three separate layers. Keeping them separate prevents a metadata
report from being mistaken for working playback:

1. **Catalog** — `AlphaAreaAudioCatalog` holds typed area and ambience records
   and joins the two DBC tables.
2. **Asset resolver** — `AlphaAreaAudioAssetResolver` checks the referenced
   virtual paths against loose files and the active MPQ/archive source. Each
   result says whether the asset is on disk, in an archive, missing, or not
   referenced.
3. **Playback runtime** — a future audio backend must decode/play MIDI and
   provide the DLS/DirectMusic instrument-bank behavior. That backend is not
   implemented merely because the catalog can resolve a `.mid` or `.dls` path.

The current inspect command proves layers 1 and 2. It does not play audio.

## What it is not

- **Not MCSE sound emitters.** `MCSE` records are positional sounds placed in
  ADT chunks. The Alpha catalog describes area-wide ambience. They will later
  feed different source types in the world-audio runtime.
- **Not a complete music system.** `ZoneMusic`, intro sounds, later sound tables,
  positional emitters, MIDI synthesis, DLS playback, mixing, and capture audio
  are separate work.
- **Not a filename guesser.** A similarly named WAV/MP3/OGG file is not treated
  as the soundtrack unless client metadata or an explicit authored binding says
  so.
- **Not proof that playback works.** “Resolved” means the client data source
  contains the referenced file. Decoder, bank compatibility, audible output,
  and video muxing still need separate proof.

## How data is found

`AlphaAreaAudioCatalogReader` accepts a configured client/data root and an
optional archive reader. It looks for `AreaTable` and `AreaMIDIAmbiences` as
loose tables first, then through the archive source. It also locates the
WoWDBDefs `definitions` directory and loads the requested build, defaulting to
`0.5.3.3368`.

Use `--build` for a different Alpha build. Do not assume the default schema is
correct for another client version.

`AlphaAreaAudioAssetResolver` checks both the client root and its `Data`
directory for loose assets, then checks the archive catalog. This is why the
inspect command needs the client root even when the DBCs are archived.

## Inspect it

From `wow-viewer/`, run:

```powershell
dotnet run --project tools/inspect -c Debug -- audio alpha-area --archive-root <client-root> --build 0.5.3.3368 --limit 20
```

Useful filters:

```powershell
# One area ID
dotnet run --project tools/inspect -c Debug -- audio alpha-area --archive-root <client-root> --build 0.5.3.3368 --area-id <area-id>

# Areas whose name or referenced asset paths contain text
dotnet run --project tools/inspect -c Debug -- audio alpha-area --archive-root <client-root> --build 0.5.3.3368 --search <text> --limit 50
```

The report includes:

- total `AreaTable` and `AreaMIDIAmbiences` records loaded;
- how many area references joined to an ambience record;
- referenced versus resolved asset counts;
- whether resolved assets came from disk or an archive;
- each selected area's name, ambience IDs, day/night paths, underwater paths,
  DLS paths, and volume.

An output such as `day=path.mid [missing]` means the DBC link was understood but
the file was not found. It does **not** mean that the DBC row failed to decode.
An output such as `day=path.mid [archive]` proves source discovery only; it does
not claim that the viewer can synthesize or play that MIDI sequence. Loose-file
results also include the resolved disk path after the `[disk:...]` marker.

## Code ownership

- [`AlphaAreaAudioCatalog.cs`](../../src/core/WowViewer.Core/Audio/AlphaAreaAudioCatalog.cs)
  — typed area/ambience records and the DBC-reference join.
- [`AlphaAreaAudioCatalogReader.cs`](../../src/core/WowViewer.Core.IO/Dbc/AlphaAreaAudioCatalogReader.cs)
  — build-aware DBC loading and record shaping.
- [`AlphaAreaAudioAssetResolver.cs`](../../src/core/WowViewer.Core.IO/Audio/AlphaAreaAudioAssetResolver.cs)
  — loose-file and archive asset probing.
- [`Program.cs`](../../tools/inspect/WowViewer.Tool.Inspect/Program.cs)
  — `audio alpha-area` inspection output.

For the future playback/runtime work, see
[`audio-engine-plan-2026-04-21.md`](audio-engine-plan-2026-04-21.md) and
[Spec 146](../../specs/146-audio-camera-playback/spec.md). The current
capability boundary is intentionally “metadata and source resolution proven;
playback not yet proven.”
