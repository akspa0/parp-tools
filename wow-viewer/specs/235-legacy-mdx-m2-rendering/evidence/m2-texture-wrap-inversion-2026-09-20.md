# Spec 235 Evidence — M2 Texture Wrap Was Inverted: Cylinders Broken Across All Eras

Date: 2026-09-20

## Operator report

> "you broke m2 texturing at some point in the last 2 weeks, across all eras that use m2 format files"
> … "tubes like tree trunks or lighthouse cylinders are screwed up, that's all"

The "tubes only" detail is what identifies it: cylindrical meshes are the geometry whose UVs run past
1.0 all the way around, so they are the only shapes that visibly break when a repeating texture is
clamped instead.

## Root cause: the wrap bit was read as a clamp bit

[`M2Renderer.cs:1147`](../../../src/viewer/WoWViewer/Rendering/M2Renderer.cs) read:

```csharp
bool clampS = (candidate.TextureFlags & 0x1u) != 0;
bool clampT = (candidate.TextureFlags & 0x2u) != 0;
```

`TextureFlags` is genuinely `M2Texture.flags` — `M2Era100Constants.TextureFlagsOffset = 0x04`, which is
the `flags` field of

```c
struct M2Texture { uint32_t type; uint32_t flags; M2Array<char> filename; };
```

and in that field the documented meaning is:

| Bit | Meaning |
|---|---|
| `0x1` | Texture wrap X |
| `0x2` | Texture wrap Y |

**The bit set means REPEAT.** Clamping is the *absence* of the bit. So `!= 0` clamped precisely those
textures that declared they must wrap, and repeated the ones that asked to clamp — exactly inverted.

A flat surface with UVs inside `[0,1]` looks identical either way, which is why only tubes showed it.

## How it got there

Introduced by `64ce5aaa` (2026-09-12, *"resolve 0-vertex M2 emitter white box rendering and correct
texture clamping behavior"*). Its own receipt states the reasoning:

> `clampS` and `clampT` were computed as `(candidate.TextureFlags & 0x1u) == 0` … any texture with
> `TextureFlags == 0` … forced `TextureWrapMode.ClampToEdge`… When UV coordinates outside `[0.0, 1.0]`
> (e.g. `UV0.X = 8.741`) were clamped to edge, the edge border pixels smeared…

The observation was real; the diagnosis inverted the flag. The original `== 0` was **spec-correct**.
The `UV0.X = 8.741` garbage that motivated the change came from somewhere else: the same work stream
records fixing `M2Era100Constants.cs` vertex-layout offsets, *"corrected inverted offsets so Normal
(0x14), UV0 (0x20), UV1 (0x28)… fixing flat/stretched distorted textures and broken UVs."* UVs were
being read from the wrong offsets. Once that was fixed, the clamp inversion had no remaining
justification but stayed in, and from then on every wrapping texture clamped.

This is a **fix layered on a symptom whose real cause was fixed separately** — the inversion was never
removed.

## Fix

Restored the spec sense, with the history recorded in a comment so it is not "corrected" back:

```csharp
bool clampS = (candidate.TextureFlags & 0x1u) == 0;
bool clampT = (candidate.TextureFlags & 0x2u) == 0;
```

`ModelRenderer.NormalizeAdaptedM2TextureSampling` — the other half of `64ce5aaa`, which used to force
`ClampToEdge` on non-opaque textures — is already an empty no-op and needed no change.

## Not from this session

For the record, since the report arrived alongside two DAT-related ones: this session's changes touch
`AhdrTerrainAdapter`, `StandardTerrainAdapter`, `ViewerApp*`, `AdtAhdr*`, `DatToLk*`, `DatLayerSource`,
the path picker and tests. **No M2, MDX, texture, shader or asset-manager file was modified.** The
defect dates to 2026-09-12.

One earlier lead was chased and dropped: `MdxRenderer._uFlipTexU` and `_flipTextureUForCurrentDraw` are
dead fields, which looked like removed flip logic — but `git log -S` dates them to 2026-06-10, well
outside the window. They remain dead and unexplained; not this bug.

## Verification

| Action | Result |
|---|---|
| `M2Era100Constants.TextureFlagsOffset` | `0x04` — the real `M2Texture.flags` field |
| Live code before fix | `!= 0` (clamp when wrap bit set) |
| `git log -S` on the flip fields | 2026-06-10, outside the window — red herring |
| `dotnet build WoWViewer.csproj -c Debug` | 0 errors |

## NOT proven

**Not visually confirmed.** No model has been rendered with the fix; that is operator-owned. Two things
to watch when checking:

1. **Tubes** — tree trunks, lighthouse cylinders — should texture continuously around the seam.
2. **The models `64ce5aaa` was originally chasing** — `ballistaruined.m2`, wood, arrows, wheels. If
   those streak again, their UVs are still out of range and the real fault is upstream in UV reading
   for that era, not in the wrap mode. That would be the next thread, and it must not be "fixed" by
   re-inverting the flag.
