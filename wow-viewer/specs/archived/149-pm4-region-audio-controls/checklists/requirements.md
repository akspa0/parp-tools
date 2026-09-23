# Specification Quality Checklist: PM4 Region Navigation and Audio Trigger Controls

**Purpose**: Validate that the feature request is complete, internally consistent, and ready for
planning without silently expanding into game-mode movement or speculative audio decoding.
**Created**: 2026-08-14
**Feature**: [spec.md](../spec.md)

## Scope and user value

- [x] CHK001 The specification names the PM4 region-navigation outcome and the audio-trigger opt-in
  outcome in user-visible terms.
- [x] CHK002 Each current user story is independently testable and delivers value without requiring
  the later player/game-mode feature.
- [x] CHK003 The deferred player-height, walking/running, jumping, and collision behavior is explicitly
  separated from this feature's acceptance criteria.

## PM4 behavior

- [x] CHK004 Region rows have a deterministic identity and enough decoded geometry totals/context to
  distinguish regions without external asset matching.
- [x] CHK005 Double-click navigation defines selection, finite camera focus, and bounded residency behavior
  for both single-tile and multi-tile regions.
- [x] CHK006 The cleanup explicitly removes correlation/matching controls and tooltip fields while
  preserving proven decoded PM4 facts.
- [x] CHK007 Empty, malformed, stale, unavailable, and empty-stub PM4 states have defined behavior.

## Audio behavior

- [x] CHK008 The trigger list scope is explicit: resident MCNK environmental/liquid candidates, resident
  MCSE instances, and the applicable current-area trigger, without whole-map or whole-client loading;
  the 0.5.3 no-MCSE case is covered.
- [x] CHK009 Default-off initialization, individual enablement, master blocking, stopping, duplicate
  prevention, and diagnostic visibility are all specified.
- [x] CHK010 Audio mute/gain and deliberate SoundEntries preview are kept separate from world-trigger
  enablement.
- [x] CHK011 Unsupported decoder, missing file/bank, unresolved metadata, and unavailable backend states
  remain inspectable instead of causing guessed playback.

## Boundaries and proof

- [x] CHK012 Existing PM4, AreaNumber, MCNK/liquid, MCSE, decoder, coordinate, and provenance owners are
  named for reuse.
- [x] CHK013 The specification does not claim MIDI/DLS playback, native callback installation, audible
  proof, visual proof, or performance proof from source changes alone.
- [x] CHK014 Focused automated coverage, Debug build proof, and user-owned real-client/audio/runtime
  proof are distinguished.
- [x] CHK015 No unresolved `[NEEDS CLARIFICATION]` markers or template placeholders remain.
