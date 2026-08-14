# Research: World Audio and Camera Playback

## Decision: Reuse the existing audio data contracts as the first boundary

The repository already contains `AlphaAreaAudioCatalog`, `AlphaAreaAudioAssetResolver`,
`AlphaAreaAudioCatalogReader`, and `AdtMcseReader`. These establish useful data ownership:
area and MIDI-ambience metadata comes from build-aware DBC definitions, referenced assets can be
resolved through loose or archive-backed client data, and MCSE records can preserve raw emitter
bytes plus decoded standard-era positions. The new work should add runtime playback contracts on
top of these readers rather than duplicate or replace them.

## Decision: Make playback backend-neutral before selecting libraries

The requested format range crosses several eras and runtime concerns: PCM/WAV, compressed MP3 and
OGG, MIDI sequences, DLS/DirectSound banks, positional mixing, and audio capture/muxing. No single
library has been proven in this repository to cover all of those requirements on every target
platform. The first implementation phase therefore owns a small C# playback/mixing contract and a
capability probe. Format-specific backends can be added behind that boundary after focused
research and real-client samples identify the required decode and bank behavior.

Python or command-line tooling may assist with offline inspection or conversion, but interactive
viewer playback and the camera/capture transport remain C#-owned. This prevents the viewer from
depending on a Python process for basic world audio and keeps the future world/session boundary
usable by the eventual single-player client direction.

## Decision: Camera audio is an explicit binding, not filename inference

An M2/MDX camera track supplies camera motion, not necessarily a complete soundtrack declaration.
The viewer may use proven client metadata, a project sidecar, or an explicit user selection. It
must not infer a soundtrack merely because a similarly named MP3/OGG/WAV exists. `CinematicCamera`
and `CinematicSequences` remain camera placement/sequence metadata; the audio source must be
resolved from the client’s actual audio tables/assets or an authored project binding.

## Decision: One logical transport owns preview and capture

Camera preview, Play + Video, looping, pause, scrub, and stop need one timebase. The viewer should
not start an independent audio player beside the camera transport. The transport reports its
playhead, lifecycle, loop state, and capture relationship to both the camera evaluator and the
audio runtime. If the capture path cannot ingest the live mix, the system reports a separate audio
artifact or silent capture before finalization rather than implying muxed audio.

## Decision: Area ambience and positional emitters are separate source classes

Area ambience is a low-frequency state selected from the resolved area and time/underwater context.
MCSE records are positional candidates tied to resident ADT tile/chunk content. They share buses and
transport controls but have different lifetime, transition, attenuation, and streaming rules.
The emitter evaluator must use the camera/player-head world position and bounded resident content;
it must not load the whole map to produce sound.

## Decision: Preserve unresolved audio evidence and fail closed

Missing client files, archive failures, unsupported decoders, missing DLS banks, and DBC/DB2 schema
gaps must remain distinguishable. Raw MCSE bytes may be retained for later archaeology, but no
positional sound may be invented from an unproven field layout. A source failure must not disable
unrelated ambience, emitters, rendering, or video capture.

## Decision: Record the single-player client/server direction as a separate roadmap boundary

The viewer is moving toward a drop-in single-player client experience. The eventual system may
reuse Alpha-Core SQL data for NPCs and game objects and add a local server/session authority, while
terrain reconstruction models and the viewer remain important upstream capabilities. That is a
major architecture program, not a deliverable of the audio MVP. Spec 146 therefore exposes a
world/session event seam and records the direction, but explicitly excludes server, login, AI,
quest, and authoritative world-mutation implementation.

## Open research gates before runtime implementation

- Identify representative client-era samples and exact DBC/DB2 layouts for area music, sound IDs,
  emitter identities, and any camera-specific audio associations.
- Compare candidate C# playback backends for WAV/MP3/OGG, low-latency mixing, spatial attenuation,
  Windows support, and cross-platform viability.
- Determine whether MIDI/DLS playback is viable in-process, requires a platform bridge, or should
  initially be an offline/diagnostic capability.
- Determine whether the existing ffmpeg capture route can accept a live mixed audio stream without
  introducing a second clock or drift.
- Establish a focused fixture set and a user-run audible proof matrix before claiming format or
  client-era support.

## 2026-08-14 implementation checkpoint

- WorldAudioRuntime now loads AreaTable and optional AreaMIDIAmbiences from the exact active DBC
  build. The optional MIDI table matters because later clients can carry usable ZoneMusic IDs even
  when the historical MIDI ambience table is absent.
- For Alpha terrain, the area value from `MCNK.Unknown3` is the packed `AreaNumber`
  `(zone << 16) | subzone`; resolution is continent-qualified and follows `ParentAreaNum` before
  using modern direct `ID`/`ParentAreaID` fallback. The viewer decodes this as two `ushort`
  components through `AreaNumberParts`; the old high-word/low-word aliasing path is removed, and
  the audio runtime receives the same resolved Zone/SubZone context used by the status bar. Paths
  still come only from DBC metadata and the active client source.
- The runtime loops a resolvable ZoneMusic asset through the existing OpenAL path and exposes an
  area-music diagnostic in the Audio panel. MIDI/DLS selections remain explicit unsupported states;
  no fake PCM conversion or filename inference was added.
- Audible playback, area transition behavior, WMO-area selection, and camera/capture transport
  synchronization remain user-run or future gates.

## 2026-08-14 client reverse-engineering checkpoint

- In the open Alpha 0.5.3 client, `AreaMIDIAmbiences` is the authoritative MIDI/DLS pairing table:
  the row provides day sequence, night sequence, and one shared DLS bank. The client asynchronously
  loads both through `SFile`, connects the DLS collection to the MIDI segment through DirectMusic,
  and starts the segment with the standard audio path.
- `ZoneMusic` is separate from `SoundEntries`. A ZoneMusic row selects day/night SoundEntries IDs;
  the ordinary SoundEntries table then chooses among up to ten declared filename/frequency pairs.
  The observed missing SoundEntries ID 1 is genuine absence, not a MIDI mapping.
- MCSE Alpha 0.5.3 records are 0x34 bytes on disk. The prior 76-byte assumption was the client’s
  in-memory `CWSoundEmitter` size, not the file record size; the reader now decodes the proven
  0x34-byte fields and preserves scheduler metadata.
- The client has create/destroy callback slots for map sound emitters, but this executable clears
  the slots and exposes no in-process registration xref. The viewer must not claim that its local
  resident-emitter path is a native-equivalent client callback implementation.

## 2026-08-14 mute control checkpoint

- Added an explicit `AUDIO: ON` / `AUDIO: MUTED` button to the bottom status bar. The control is
  intentionally outside the crowded diagnostic tabs so audio output state is always visible.
- Mute is owned by the viewer audio runtime's master bus and applies to resident emitters, sound
  preview, and resolved ZoneMusic. The configured master gain is retained while muted.
- Source/build proof is the viewer Debug build; audible mute/unmute behavior remains user-owned.
