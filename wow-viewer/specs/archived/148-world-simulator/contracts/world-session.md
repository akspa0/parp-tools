# Contract: World Session and Camera Actor

## WorldSession

A local session owns the configured client build/root, selected map, active `CameraActorState`, path
playback state, optional audio backend, residency leases, and diagnostic snapshots. It does not
require a Blizzard executable, server, or repository-contained client assets.

## CameraActor

Manual camera input and camera-path playback update one authoritative actor transform. Rendering,
audio listener placement, terrain/WMO area lookup, collision, and residency selection consume the
same snapshot. A visible camera asset is optional and must not be required for the contract.

## Context transitions

Entering/leaving a terrain chunk, WMO placement, WMO group, or camera-path segment publishes a new
context snapshot. Existing consumers may continue using their current paths during migration, but
the diagnostic snapshot must identify when two consumers disagree.

## Lifecycle

Session open creates the actor and baseline lease set. Manual movement refreshes actor/fog leases.
Path playback adds a swept warmup lease. Stop, map change, and close release leases according to
their owner and hold interval; stale GPU/audio handles must never be used after release.
