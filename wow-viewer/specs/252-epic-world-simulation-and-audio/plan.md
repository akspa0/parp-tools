# Plan — Epic 252 World Simulation, Audio & Interaction

**Status**: Implementation approach **not yet selected** (operator directive 2026-09-23: triage first).

## Dependencies between backlog items

```text
W-01 ZoneMusic ──> W-05 lifecycle ──> W-02 path audio ──> W-03 video muxing
W-11 head camera ──> W-13 game mode ──> W-20 physics (bounded)
W-20 physics ──> W-21 weather (wind consumer)
W-23 creature staging ──> W-22 cursor light (torch-in-hand reproduction)
W-24 server data ──> W-25 museum sim
```

Era split (do not conflate): 0.5.3 audio is DirectSound + DirectMusic; 5.0.1 audio is FMOD/SE3
(memory-bank workstream notes).

## Design documents adopted by reference

| Item | Adopted design |
|---|---|
| W-01–W-06 | [146 plan](../archived/146-audio-camera-playback/plan.md) · [148 plan](../archived/148-world-simulator/plan.md) · [217 spec](../archived/217-audio-lifecycle/spec.md) · [0.5.3 audio Ghidra](../../memory-bank/workstream-audio-client-053-ghidra.md) · [5.0.1 audio Ghidra](../../memory-bank/workstream-audio-501-ghidra.md) |
| W-10–W-12 | [143 plan](../archived/143-world-context-lighting/plan.md) |
| W-13 | [151 plan](../archived/151-portal-game-mode-surface/plan.md) |
| W-14 | [156 plan](../archived/156-precise-object-selection/plan.md) |
| W-15, W-16 | [158 plan](../archived/158-alpha-demo-restoration/plan.md) · [159 plan](../archived/159-wtf-command-inspection/plan.md) |
| W-20, W-21 | [214 plan](../archived/214-mop-physics-domino/plan.md) · [215 spec](../archived/215-mop-weather-system/spec.md) · [5.0.1 atmosphere Ghidra](../../memory-bank/workstream-atmosphere-501-ghidra.md) |
| W-22, W-23 | [216 spec](../archived/216-model-cursor-light-source/spec.md) · [218 spec](../archived/218-creature-staging/spec.md) |
| W-24, W-25 | [186 spec](../archived/186-server-data-transformer/spec.md) · [187 spec](../archived/187-museum-world-simulation/spec.md) |

## Standing constraints

Audible, visual and motion proof is operator-owned. The W-20 third-party solver dependency needs
explicit operator approval before any package reference is added (214 T021).
