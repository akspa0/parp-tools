# Receipt — T100/T101 amendment A: M2 animation cost (2026-09-27)

Operator: "yes, the animation is eating everything for breakfast. it's especially apparent when you move from an area
with no animated doodads to an area with animated models ..."

## Findings (T100, code reading; per animated native M2, per frame)
1. `M2TrackSampler.Sample` and `M2AnimatedRenderStateEvaluator.EvaluateTrack` decoded **every key frame of the
   track from the raw bytes into a new List on every call** — for every bone translation/rotation/scale track,
   every color/alpha/texture-weight/texture-transform track and every light track.
2. `M2BonePoseEvaluator.Evaluate` allocated a pose object per bone plus arrays and a LINQ copy.
3. `M2SkinnedRenderModelBuilder.ApplyPose` rebuilt every vertex into new Lists; `UploadAnimatedVertices` then built
   a float[] per section and re-uploaded the whole vertex buffer.
4. `ApplyAnimatedFrame` ran a LINQ GroupBy/ToDictionary and an OrderBy per section.
New animated models entering view each pay all of this at once, plus the garbage it leaves for the GC.

## Changes (T101)
| Step | Change | Output |
|---|---|---|
| A1 | `M2TrackKeyFrameCache` (core, new): decoded key frames cached per (track, payload, slot) in both samplers | identical (deterministic decode) |
| A2 | `M2BonePoseEvaluator.EvaluateWorldMatrices` (core, additive): same `SolveBone` math into caller arrays | identical matrices (test) |
| A3 | Native `M2Renderer` GPU skinning for models with 1–128 bones: bind pose + resolved bone indices/weights in the vertex buffer (resolved once with the now-public `M2SkinnedRenderModelBuilder.ResolveBoneIndex`), `uBones`/`uSkinned` uniforms, shader weighting identical to `ApplyVertex`; >128 bones keep the CPU path | same formula on the GPU (float math on GPU vs CPU: not bit-identical) |
| A4 | `ApplyAnimatedFrame` without LINQ (first pass per section; first texture with the lowest stage) | identical selection |

Files: `src/core/WowViewer.Core.Runtime/M2/M2TrackKeyFrameCache.cs` (new), `M2TrackSampler.cs`,
`M2AnimatedRenderStateEvaluator.cs`, `M2BonePoseEvaluator.cs`, `M2SkinnedRenderModelBuilder.cs`;
`src/viewer/WoWViewer/Rendering/M2Renderer.Skinning.cs` (new), `M2Renderer.cs`, `M2Renderer.Instancing.cs`;
`tests/WowViewer.Core.Tests/M2RuntimeTests.cs` (+1 test).

## Verification
| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 0 | 0 errors; compiler warnings identical to base |
| `dotnet test WowViewer.slnx -c Debug` | 1 | Core.Tests 1,600 passed (+1: `BonePoseEvaluator_WorldMatricesMatchEvaluateAndCachedKeyFramesSampleIdentically`); 26 failures identical to the pre-existing E4 set |
| `glslangValidator m2.vert m2.frag -l` (sources from `M2Renderer.cs`) | 0 | compile + link, `#version 330 core` |

## Criterion → evidence
| Criterion | Evidence |
|---|---|
| Cached sampling = fresh sampling; world matrices = `Evaluate` | Unit test above (exact equality over 7 sample times, cache warm and cold) |
| MdxAnimation stage drops; no hitch entering animated areas; animation looks the same | **Not proven.** Operator capture owed (T102) |
