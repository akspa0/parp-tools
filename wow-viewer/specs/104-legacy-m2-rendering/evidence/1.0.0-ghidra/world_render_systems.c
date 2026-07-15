// ============================================================================
// WoW 1.0.0 (build 3980) — World render systems: M2 draw, terrain surface,
//   liquid/water, blend modes, render state. Decompiled via GhidraMCP 2026-07-15.
// ============================================================================


// ############################################################################
// SECTION A — M2 / DOODAD / CREATURE RENDER
// ############################################################################
//
// Per-model tick (world doodad path):  FUN_006bf060
//   - model alpha fade   = (byte @model+0xb) / 255       (FUN_0070c0a0)
//   - advance anim time  = now - lastTime                 (FUN_007169b0)
//   - build sorted draw list                              (FUN_00716c40)
//   - draw pass 0 (opaque) then pass 1 (transparent)      (FUN_00717cd0 x2)
//
// Draw-list builder FUN_00716c40 buckets each visible submesh/effect into a
//   0x38-byte "M2Batch" render entry with a type tag [0]:
//       0 = opaque geometry     1 = transparent geometry (needs sort)
//       2 = multi-region geometry (draws entry[0x1c] sub-entries)
//       3 = particle emitter     4 = ribbon emitter      5 = projected shadow
//   entry fields: [0]=type, [1]=model, [3]=sortKey(dist), [5]=dist,
//   [9]=submeshIndex, [0xa]=submeshPtr, [0xb]=materialPtr, [0xc]=isOptimized.
//   Transparent entries (type 1) are distance-sorted (FUN_00721080) back-to-front.
//
// Draw dispatch FUN_0071a150(pass, entries, sortedIdx[], count):
//   switch(entry.type){ 0:FUN_0071b550  1:FUN_0071b970  2:FUN_0071bd30
//                       3:FUN_0071c190  4:FUN_0071c220  5:FUN_0071c2d0 }
//
// Geometry draw FUN_0071b550 (opaque; 0071b970 = transparent, same shape):
//   FUN_0071a540()  // texture units + animated UV transforms  (see below)
//   FUN_0071a910()  // render flags -> lighting/fog/blend enable  (see below)
//   FUN_0071ae30()  // material color+alpha (from anim tracks) -> vertex color
//   <transform verts by bone matrices>   FUN_0058dd90()  // draw indexed
//
// ---- render flags (M2 renderFlag = { u8 flags, u16 blendingMode }) ----
// FUN_0071a910 reads the renderFlag pair (ptr @state+0x3260):
//   state+0x3238 = (~flags) & 1     -> bit 0x01 = UNLIT   (1 => do lighting)
//   fogEnable    : if (flags & 2) || batchAlpha<=0 -> 0   -> bit 0x02 = UNFOGGED
//   state+0x3248 : (blendMode==3 || blendMode==4)   -> the "additive" class
//   (other flags standard M2: 0x04 two-sided/no-cull, 0x08 depth-test off,
//    0x10 depth-write off, 0x40 shadow-batch — not all exercised here)
//
// blendingMode enum (client values; GL mapping is the standard M2 table):
//   0 Opaque         glDisable(BLEND)                     (src=ONE dst=ZERO)
//   1 AlphaKey       alpha-test ~0.5, no blend            (mask/cutout)
//   2 Alpha          BLEND src=SRC_ALPHA dst=INV_SRC_ALPHA
//   3 Add            BLEND src=SRC_ALPHA dst=ONE          (additive)  [flagged]
//   4 Add_Alpha      BLEND src=SRC_ALPHA dst=ONE (no vtx-alpha mod)   [flagged]
//   5 Mod            BLEND src=DST_COLOR dst=ZERO         (modulate)
//   6 Mod2x          BLEND src=DST_COLOR dst=SRC_COLOR    (modulate 2x)
//
// ---- textures FUN_0071a540 ----
//   assert m_batch->textureCount < 2   => MAX 2 TEXTURES / M2 BATCH.
//   for each tex unit: resolve texture (batch+0x12 lookup) + apply the animated
//     texture matrix (batch+0x16 = textureTransform index -> M2 tex-anim track).
//   -> M2 supports 2-texture batches (env-map / detail) with per-unit animated UV.
//
// ---- material color FUN_0071ae30 ----
//   RGB @material+0x190/0x194/0x198, alpha @+0x18c, emissive @+0x19c/0x1a0/0x1a4,
//   all sampled from the M2 color/alpha animation tracks and modulated by the
//   submesh color; if UNLIT the emissive is folded into the base color.


// ############################################################################
// SECTION B — TERRAIN SURFACE (MapChunkRender.cpp)
// ############################################################################
//
// Chunk render FUN_006c0bc0: picks a plain path (no shadow/fog) vs a shadow+fog
//   path, sets fog index, then fills the VBO and draws the LOD strip.
//
// Vertex fill FUN_006c0db0 -> VBO stride 0x18 (24 bytes) per vertex:
//     +0x00 C3Vector position   (from chunk vtx+0x848)
//     +0x0c C3Vector normal      (from chunk vtx+0x17c, the MCNR normal)
//   NO color, NO uv in the stream => terrain is HARDWARE FFP-LIT: normals are
//   uploaded and GL_LIGHT0 (sun) + ambient compute N·L on the GPU. (Resolves the
//   "FFP vs CPU bake" open question: it's FFP hardware lighting.) UVs are
//   generated (planar tiling of the layer textures over the chunk).
//
// Draw FUN_006c0e80 -> FUN_0058d0b0(indexCount<<1, 0): one indexed draw of the
//   neighbour-stitched LOD triangle-strip (index count from a per-LOD table).
//
// Texture layers: up to 4 MCLY layers; layer 0 opaque base, 1-3 alpha-blended by
//   the MCAL alpha maps, composited by the FFP multitexture / register combiners.
//   Shadow: MCSH 64x64 1-bit map -> shadowGxTexture multiplied over the lit result.


// ############################################################################
// SECTION C — LIQUID / WATER (MapWater.cpp)
// ############################################################################
//
// 12 liquid types (LIQUID_COUNT=0xC, LIQUID_NONE sentinel). Per-type texture base
//   names in liquidTexBaseName[type] @ 0x00834d4c, e.g.:
//     XTextures\river\lake_a.%d.blp   XTextures\river\fast_a.%d.blp
//     XTextures\ocean\ocean_h.%d.blp  XTextures\slime\slime.%d.blp   (+lava/splash/water)
//   The "%d" is the ANIMATED FRAME index — ~30 frames/type cycled over time
//   (frame advance FUN_00686d40). So water = flipbook-animated tiling texture.
//
// Tile grid: liquid stored as an 8x8 tile block per map-chunk (MD_LIQUID_NPOLY=8),
//   each tile has a type nibble + depth; FUN_00687460 finds the nearest liquid
//   tile to a point (for splash/sound/wading). Height/depth per tile drives the
//   surface Z and the shore alpha fade.
//   Terrain liquid chunk = MCLQ; WMO-internal liquid = MLIQ (see wmo doc §20.5).
//
// Shaders: ocean0_s.bls (terrain deep water), MapObjExtWater0.bls (WMO near water).
//   Ripple effect: Water0Ripple / WaterRadWave. Water is drawn as a transparent
//   surface (alpha-blended) after opaque terrain/WMO, with the animated texture,
//   optional specular, and depth-based shore transparency.
//   Cvars: waterParticulates / Ripples / Specular / Waves / MaxLOD / LOD, SetWaterDetail.


// ############################################################################
// SECTION D — RENDER STATE / CGx (CGxStateBom)
// ############################################################################
//
// Fixed-function state is set through a batched state stack (CGxStateBom):
//   FUN_0058cb30 = push a render-state token,  FUN_0058cb70 = push (variant),
//   FUN_0058cae0 = set texture/sampler,        FUN_0058ca90 = set constant color,
//   FUN_0058ccb0 / FUN_0058ccc0 = begin/end pass,   FUN_0058dd90 = draw indexed.
// State changes are queued and flushed as a "state bomb" to minimize GL/D3D calls.
// Backends: CGxDeviceOpenGl (NV register combiners / ARB fp / texture_shader) and
//   CGxDeviceD3d (ps_1_1..ps_2_0). Lighting/fog/material/blend are all FFP tokens;
//   the .bls pixel shaders only do the multi-texture COMBINE stage.
//
// Master frame order (CWorldScene::Render FUN_0067c460, see wmo doc §20.2):
//   camera-in-WMO test -> interior portal walk OR exterior pass -> OPAQUE world
//   (terrain + WMO opaque + opaque doodads) -> fog select (interior/exterior) ->
//   TRANSPARENT/effect passes (liquids, transparent doodads/WMO, particles,
//   ribbons) -> projected unit shadows / blobs -> debug overlays.
