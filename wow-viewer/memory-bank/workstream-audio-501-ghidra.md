# 5.0.1 Audio (SE3) — Ghidra Evidence

Last verified: 2026-09-02 against Ghidra project `Mists of Pandaria 5.0.1.15464`, program `/Wow.exe`.
Read-only session; nothing in the program was edited.

Companion to [`workstream-audio-client-053-ghidra.md`](workstream-audio-client-053-ghidra.md), which
holds the **0.5.3** evidence. Read both: the two eras do not share a backend.

## The backend is era-split — this is the headline

| Era | Backend | Evidence |
|---|---|---|
| 0.5.3 | **DirectSound + DirectMusic** | `MIDI_Play @ 0x007b7370`, `GUID_ConnectToDLSCollection`, `PlaySegmentEx` (see the 0.5.3 note) |
| 5.0.1 | **FMOD** | `fmod_soundi.cpp`, `fmod_music.cpp`, `fmod_output_dsound.cpp`, `fmod_reverbi.cpp`, `fmod_soundgroupi.cpp`, `fmod_dsp_sfxreverb.cpp`, EAX2/3/4 outputs |

Anything written as "the client's audio system" without naming an era is wrong on one of them.

## SE3 — the 5.0.1 sound engine

WoW-side anchors: `SE3MemorySounds.cpp` `0x00d84668`, `SI3ZoneSounds.cpp` `0x00dabf58`,
`UnitSound_C.cpp` `0x00dab0b8`, `BPlayerSound_C.cpp` `0x00daaddb`,
`SI3VocalErrorSounds.cpp` `0x00e80874`, `ComSatSoundIOSoundEngine.cpp` `0x00e09d5c`.

RTTI confirms the object model: `SE3SoundKitDef`, `SE3CombinableSoundKitPosition`,
`SE3CreateSoundEmitterMessage`, `SE3DestroySoundEmitterMessage`, and explicit lists over
`SE3SOUNDKITOBJECT_LOOKUP` / `SOUNDKITLOOKUP`.

### A playing sound is a state machine over six lists

This is the central finding. Every sound is a `SoundKitObject` that lives on exactly one list and is
**explicitly unlinked from one and linked to another** at each transition. The log strings name every
edge:

| List | Meaning |
|---|---|
| `sm_SoundKitObjects_WaitingForDownloadList` | created for download, play ASAP |
| `sm_SoundKitObjects_Loading` | data not yet in the cache |
| `sm_SoundKitObjects_GoGoGo` | data ready, not yet playing |
| `sm_SoundKitObjects` | playing |
| `sm_SoundKitObjects_FadeList` | fading in or out, with an explicit fade time in ms |
| `sm_SoundKitObjects_DeleteList` | finished; reclaimed by `ProcessSoundKitObjectDeleteList` |

Observed transition strings include `"Data found in cache node %d, linking to SoundKitObjects_GoGoGo
list"`, `"Unlink from sm_SoundKitObjects_GoGoGo, Link to sm_SoundKitObjects"`, `"Fade out NOW (fade
time %d ms) - Link to sm_SoundKitObjects_FadeList"`, `"Done Fading Out, Linking to
sm_SoundKitObjects_DeleteList"`, and `"ProcessSoundKitObjectDeleteList: Delete!"`.

**A sound that never reaches the delete list never stops.** That is the mechanism behind a one-shot
trigger playing forever, and it is a *lifecycle* defect, not a playback defect. Any implementation
without these transitions will reproduce the bug regardless of how correct its decoding is.

### Sounds have three repeat modes, not two

`"StopSound: Periodic sound between tweets: Unlink from sm_SoundKitObjects, Link to
sm_SoundKitObjects_DeleteList"` establishes a distinct **periodic** mode — a sound that repeats with
silence between repetitions ("between tweets") — which is neither a one-shot nor a continuous loop,
and which is stopped by a different path. Conflating periodic with looping produces continuous noise
where the client produces occasional chirps.

### Entry point, categories and duplicate suppression

```text
PlaySoundKitID(ID, optional["SFX","Music","Ambience" or "Master"], optional[forceNoDuplicates])
```

- Four **category buses**: SFX, Music, Ambience, Master.
- **`forceNoDuplicates`** — the client has explicit duplicate suppression at the play call. Its
  absence is a plausible cause of the same trigger stacking on itself.
- Internal entry: `SE3::PlaySoundKitInternal`, logged as `PlaySoundKit(%d) - %s`.
- Errors are named: `ERROR_SOUNDKITISEMPTY`, `ERROR_INVALIDSOUNDKITID`, `ERROR_INVALIDSOUNDKITNAME`.

### Channels are finite and prioritised

`"No More Valid Channels, linking to SoundKitObjects_DeleteList"`, `"Failed to set priority (WTF??)"`
and `"Failed to play (FMOD_RESULT: %d)"` prove a bounded channel pool with per-sound priority, and
that failure to acquire a channel is a *normal, handled* outcome that routes to deletion rather than
an error state. An implementation with unbounded voices does not behave like the client under load.

### Variation selection is weighted, not uniform

Asserts `m_pSoundKitRec->m_Freq[i] >= 0` and `m_pSoundKitRec->m_Freq[i] < 255` show a per-variation
frequency array on the sound-kit record. This is the same shape the 0.5.3 note already measured —
`BuildSoundFilesRec @ 0x004a4890` proved ten filename pointers at `0x0c..0x30` paired with ten
frequency values at `0x34..0x58`. **The frequency array is a weight, and ignoring it makes selection
uniform**, which sounds wrong in a specific way: rare variations play as often as common ones.

`SE3SoundKitDef` is refcounted (`m_nRefCount >= 0`) and has an `m_pAdvancedSoundKitRec` alongside the
base record.

### Emitters are message-driven

`SE3CreateSoundEmitterMessage` / `SE3DestroySoundEmitterMessage`, with
`pEmitterLookup->m_pSoundEmitter->m_pParentSoundChunk` tying an emitter to a parent **sound chunk** —
the same relationship 0.5.3's MCSE has to its MCNK. Data: `SoundEmitters.dbc` `0x00e05bf4` and
`SoundEmitterPillPoints.dbc` `0x00e05c88` (a "pill" is a capsule — an emitter region, not a point).

## Era warning

None of this is evidence about 0.5.3. 0.5.3 is DirectMusic/DirectSound with MIDI+DLS ambience and
`SoundEntries`; SE3, FMOD, `SoundEmitters.dbc` and `SoundEmitterPillPoints.dbc` are later. The
*lifecycle discipline* (a sound must reach a delete path) and the *concepts* (one-shot vs periodic vs
loop, weighted variation, finite prioritised channels, duplicate suppression) are what transfer.
Whether 0.5.3 implements each of them is a question for the 0.5.3 binary, and
[[project_mcse_emitter_frame_unverified]] still stands: the MCSE chunk-local frame assumption has no
evidence and must be measured before `ConvertSoundPosition` is touched.

## What has NOT been done

No function here has been decompiled. These are strings, RTTI names and DBC filenames — enough to
explain the observed defect and aim the work, and nothing more. No record layout, list-transition
condition, channel-selection rule or timing value has been recovered. Do not cite this note as
evidence for any of those.

## Consumed by

- **Spec 217** — audio lifecycle and playback correctness.
