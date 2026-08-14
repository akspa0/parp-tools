# Research: Artifact World Simulator Runtime

## Local implementation audit

### Current MPQ and audio path

- `src/viewer/WoWViewer/DataSources/MpqDataSource.cs` indexes and reads loose files, the alpha MPQ
  cache, and the archive catalog. The catalog includes audio entries in the current client logs,
  but the count of `.wav` or `.mp3` files proves only catalog visibility, not that a DBC-resolved
  virtual path came from `sounds.mpq`, was read successfully, or decoded.
- `src/core/WowViewer.Core.IO/Dbc/AlphaSoundEntriesCatalogReader.cs` resolves SoundEntries rows and
  combines `DirectoryBase` with declared files. It intentionally does not guess alternate paths.
- `src/viewer/WoWViewer/Audio/ClientAudioDecoder.cs` currently handles WAV, OGG/Vorbis, and MP3;
  MIDI/DLS are explicitly rejected because a soundbank/backend is not yet proven.
- `WorldAudioRuntime` combines SoundEntries resolution, file reads, decoding, OpenAL buffers,
  distance admission, emitter playback, and area music. The Phase 1 slice now exposes a first-class
  resident-MCSE decision list; area music remains a compact active-area status until the later
  per-trigger area/WMO context slice.

### Current MCSE and area context

- Alpha and Standard terrain adapters own MCSE extraction and convert positions into renderer-world
  coordinates. The Phase 1 slice preserves both raw and transformed positions with an explicit
  profile label; it does not change the existing transform.
- Standard MCSE entries expose IDs and position; range fields are not populated by the common reader.
  Alpha 0.5.3 entries have additional min/max/cutoff/start/end/mode fields.
- `DBCTool.V2` establishes the Alpha area contract: `MCNK.Unknown3` is the packed
  `AreaNumber` `(zone << 16) | subzone`, and an Alpha child is parented through `ParentAreaNum`
  after the row is matched by `ContinentID`. The viewer's ZoneMusic catalog now follows that
  numeric contract before falling back to modern direct `ID`/`ParentAreaID` resolution; it does
  not treat the packed value as an unrelated single AreaTable ID.
- `WorldScene` obtains the audio area from the terrain chunk under the camera. WMO area context is
  not yet the same audio lookup path.
- The first diagnostic phase must make these gaps visible before changing a transform or assuming
  that an era uses the same MCSE layout.

### Failure gates to separate

1. Trigger is not present in the current residency set.
2. SoundEntries ID is unresolved or has ambiguous paths.
3. Virtual path is catalog-visible but not readable.
4. Bytes are readable but format/decoder fails.
5. OpenAL/native backend is unavailable or buffer/source creation fails.
6. Emitter is outside effective range, muted, or rejected by start/end/mode.
7. Area music/ambience is MIDI/DLS and the required soundbank backend is unsupported.

## External architecture comparisons

- [WoWee](https://github.com/Kelsidavis/WoWee) separates rendering, asset, world, UI, and audio
  concerns and describes asynchronous world streaming. Its README also extracts client data to
  loose files/manifest form, so it is a modularity reference, not evidence that this viewer should
  stop reading MPQs directly.
- [WowUnreal](https://github.com/Clancey/WowUnreal) documents MPQ-backed data access, world
  streaming, lazy DBC loading, M2 doodad instancing/frustum/LOD work, zone lighting, water/fog,
  and zone music/ambience. It targets 3.3.5a; those choices guide instrumentation and ownership,
  not a universal first-decade schema claim.
- [WebWowViewerCpp](https://github.com/Deamon87/WebWowViewerCpp) is a useful map-viewer comparison
  for map selection and minimap-oriented workflows, but it is not a source for this project's era
  schemas or audio behavior.
- The [Hugging Face World of Warcraft dataset search](https://huggingface.co/datasets?other=world-of-warcraft)
  did not expose an authoritative renderer/audio implementation relevant to this task. No dataset,
  model, or external asset is needed for the first phases.

## Decisions

- Prove the current pipeline through diagnostics before changing archive selection, MCSE transforms,
  or audio backend dependencies.
- Treat the camera actor as a shared state/lifecycle contract. Rendering a camera M2 is optional and
  is not considered a performance optimization by itself.
- Use leases and stage attribution to make fog/doodad performance changes reversible and measurable.
- Keep MIDI/DLS as an explicit capability boundary until a backend can be run and licensed safely on
  the target platforms.
- Keep the viewer a local, user-supplied-artifact museum tool. Future simulator/server/LLM work is
  represented by contracts and extension points, not bundled into the audio fix.
