// ============================================================================
// WoW 1.0.0 (build 3980, beta-3) — WMO Scene / Portal / BSP / MOPY / MLIQ
// Decompiled via GhidraMCP (/decompile_function) — 2026-07-15 follow-up pass.
// Resolves the "Open Follow-ups" items 2,5,6,7,8 from
//   docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md
//
// Program: WoW.exe  image_base 0x00400000  (H:/CLIENTS/1.X_Retail...1.0.0.3980)
// Source files (from asserts):
//   WoW\Source\WorldClient\WorldScene.cpp
//   WoW\Source\World\Map\MapObjGroup*.cpp  (WMO group parse)
//   WoW\Source\World\Map\MapObjRender.cpp  (portal + batch render)
//   WoW\Common\AaBsp.cpp                    (BSP tree)
// ============================================================================


// ############################################################################
// SECTION A — WMO GROUP CHUNK PARSERS  (MOPY / MOBA / MLIQ / MOBN+MOBR / ...)
// ############################################################################

// ----------------------------------------------------------------------------
// FUN_006c5380 — CMapObjGroup::Read  (group file reader, version 0x11)
//   Validates MVER==0x11 + MOGP token, copies MOGP header (0x58 bytes) into the
//   group object, then dispatches sub-chunks via FUN_006c55a0 (mandatory) which
//   in turn calls FUN_006c5810 (optional). MOGP header field map:
//     piVar1[5]  -> group name offset (+group[0xbc] = root MOGN base + this)
//     piVar1[7]  -> group[0x10]  = SMOGroup FLAGS  (drives optional-chunk gating)
//     piVar1[8..0xd] (6 dwords) -> group[0x14] = bounding box (min xyz, max xyz)
//     (u16)piVar1[0xe]        -> group[0x2c] = MOPR portalRefStart
//     (u16)piVar1[0x3a]       -> group[0x30] = MOPR portalRefCount
//     (i16)piVar1[0xf]        -> group[0x3c] = transBatchCount
//     (u16)piVar1[0x3e]       -> group[0x3e] = intBatchCount   (interior/opaque)
//     (i16)piVar1[0x10]       -> group[0x40] = extBatchCount / total batch count
//     piVar1[0x11]            -> group[0x34]
//     piVar1[0x12]            -> group[0x38]
//     piVar1[0x13]            -> group[0x148]
//   Then FUN_006c55a0(&piVar1[0x16])  ( &MOGP data = header+0x58 = first sub-chunk )
//   Then per-batch material warm-up loop: FUN_006c5080(batch[i].materialIndex@0x17)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// FUN_006c55a0 — mandatory MOGP sub-chunk parser (linear IFF walk).
//   this = CMapObjGroup*, param_2 = ptr to first sub-chunk.
//   ALL of these are required in fixed order (asserts on token mismatch):
//
//   MOPY  0x4d4f5059  group[0xc0]=data  group[0x120]=size>>1   (2 bytes / triangle)
//   MOVI  0x4d4f5649  group[0xc4]=data  group[0x124]=size>>1   (u16 vertex indices)
//   MOVT  0x4d4f5654  group[0xcc]=data  group[0x12c]=size/0xc  (C3Vector, 12 B)
//   MONR  0x4d4f4e52  group[0xd0]=data  group[0x130]=size/0xc  (C3Vector normals)
//   MOTV  0x4d4f5456  group[0xd4]=data  group[0x134]=size>>3   (2 floats, 8 B)
//   MOBA  0x4d4f4241  group[0xd8]=data  group[0x138]=size/0x18 (24-byte batches)
//
//   vertexBufSize (group[0xe8]):
//     if ((group[0x10] & 0x48) == 0)  group[0xe8] = vertCount * 0x24   (36 B/vtx)
//     else                            group[0xe8] = vertCount << 5     (32 B/vtx)
//     (0x48 = flags 0x40|0x08; the 4-byte delta = one MOCV BGRA colour slot.)
//
//   Then FUN_006c5810(afterMOBA)  -> optional chunks (see below).
//   Then indexBufSize (group[0xec]):
//     if (group[0xc8]!=0 && DAT_00a93790/*useStrips*/) group[0xec]=group[0x128]<<1
//     else                                             group[0xec]=group[0x124]<<1
// ----------------------------------------------------------------------------
void __thiscall FUN_006c55a0(int this, int *chunk)
{
  if (chunk[0] != 'MOPY') ASSERT("MapObjGroup..cpp",0x1e4,"pIffChunk->token=='MOPY'");
  *(int**)(this+0xc0)  = chunk+2;          // MOPY data
  *(uint*)(this+0x120) = (uint)chunk[1]>>1; // MOPY count = size / 2  (2 B / tri)
  // ... MOVI, MOVT, MONR, MOTV identical pattern ...
  // MOBA:
  //   *(int*)(this+0xd8)  = &MOBA.data;
  //   *(uint*)(this+0x138)= MOBA.size / 0x18;   // 24-byte SMOBatch
  FUN_006c5810(afterMOBA_ptr);
}

// ----------------------------------------------------------------------------
// FUN_006c5810 — OPTIONAL MOGP sub-chunk parser.  Each block is gated by a bit
//   in the group SMOGroup flags (group[0x10]).  Order is fixed:
//
//   flag 0x00200  MOLR 0x4d4f4c52  group[0xe0]=data group[0x13c]=size>>1 (u16 light refs)
//   flag 0x00800  MODR 0x4d4f4452  group[0xe4]=data group[0x140]=size>>1 (u16 doodad refs)
//   flag 0x00001  MOBN 0x4d4f424e + MOBR 0x4d4f4252  (BSP — see SECTION C)
//                   nodes = &MOBN.data, nodeCount = MOBN.size>>4  (16-byte nodes)
//                   refs  = &MOBR.data, refCount  = MOBR.size>>1  (u16 face refs)
//                   FUN_00695b80(nodes,nodeCount,refs,refCount, &group[0x14]);
//   flag 0x00400  MPBV/MPBP/MPBI/MPBG  (map-object particle-batch chunks; skipped)
//   flag 0x00004  MOCV 0x4d4f4356  group[0xf0]=data group[0x144]=size>>2 (BGRA, 4 B)
//                   (if DAT_00aade64==0) FUN_006c57b0()  // vertex-colour fixup
//   flag 0x01000  MLIQ 0x4d4c4951  (liquid — see SECTION D)
//   flag 0x20000  MORI 0x4d4f5249 + MORB 0x4d4f5242  (triangle-strip idx + render batch)
//                   group[0xc8]=&MORI.data group[0x128]=MORI.size>>1
//                   group[0xdc]=&MORB.data
//                   if (useStrips) copy each MORB{u32 baseVtx, u16 startIdx} into
//                     batch[i].+0x0c / +0x10  (overrides MOBA base/start for strips)
// ----------------------------------------------------------------------------
void __thiscall FUN_006c5810(int this, int *p)
{
  if (*(uint*)(this+0x10) & 0x200) { /*MOLR*/ *(int**)(this+0xe0)=p+2; *(uint*)(this+0x13c)=(uint)p[1]>>1; p=(int*)((int)(p+2)+p[1]); }
  if (*(uint*)(this+0x10) & 0x800) { /*MODR*/ *(int**)(this+0xe4)=p+2; *(uint*)(this+0x140)=(uint)p[1]>>1; p=(int*)((int)(p+2)+p[1]); }
  if (*(byte*)(this+0x10) & 1) {    /*MOBN+MOBR*/
      uint sz = p[1]; int *nodes = p+2; int *mobr = (int*)((int)nodes+sz);
      // ... assert mobr token == 'MOBR' ...
      FUN_00695b80(nodes, sz>>4, mobr+2, (uint)mobr[1]>>1, this+0x14);
      p = (int*)((int)(mobr+2)+mobr[1]);
  }
  // ... MPBV group (flag 0x400) ...
  if (*(byte*)(this+0x10) & 4) {    /*MOCV*/ *(int**)(this+0xf0)=p+2; *(uint*)(this+0x144)=(uint)p[1]>>2; p=(int*)((int)(p+2)+p[1]); if(DAT_00aade64==0) FUN_006c57b0(); }
  if (*(uint*)(this+0x10) & 0x1000){ /*MLIQ — SECTION D*/ }
  if (*(uint*)(this+0x10) & 0x20000){/*MORI+MORB — strip batches*/ }
}


// ############################################################################
// SECTION B — MOPY  (per-triangle material/collision flags)
// ############################################################################
//
// On-disk MOPY entry = 2 bytes, one per triangle (MOVI face):
//     +0x00  u8  flags        (material/collision flag bits)
//     +0x01  u8  materialId    (index into MOMT; 0xFF = collision-only, no render)
//   count = MOPY.size / 2 = triangle count = MOVI.size/3.  (group[0x120])
//
// RUNTIME USE (recovered):
//   * MOPY.materialId is NOT consulted by the runtime renderer — draw batches
//     come from MOBA (SECTION E); each batch carries its own materialIndex@0x17.
//     materialId=0xFF collision-only triangles simply never appear in any MOBA.
//   * MOPY.flags IS consulted, as a per-face COLLISION FILTER MASK.  During BSP
//     node-cache build (FUN_00696bf0) each face's flags byte is cached as:
//         cachedFaceFlags = MOPY.flags[faceIdx] & 0x7f      // bit 0x80 stripped
//     stored in the cache node at +0x1fae[i].  Collision queries
//     (FUN_006a2840 line, FUN_006a2c60 box) then test:
//         if ((cachedFaceFlags & queryMask) == 0) skip face;   // filter
//     i.e. a face participates in a query only if its flags intersect the
//     query's mask.  A second, mutable per-face byte (temp "visited", OR 0x80)
//     is distinct from the MOPY flags.
//
// Bit meanings are NOT named in the beta strings.  Cross-referenced to the
// documented WMO SMOPoly flag set (stable across the 1.x–3.x format):
//     0x01 F_UNK_01     0x02 F_NOCAMCOLLIDE  0x04 F_DETAIL   0x08 F_COLLISION
//     0x10 F_HINT       0x20 F_RENDER        0x40 F_UNK_40   0x80 F_COLLIDE_HIT
// The 0x80 strip at cache time matches F_COLLIDE_HIT being a runtime-only result
// bit, not a static classification bit.
//
// see FUN_00696bf0 (SECTION C) for the read site, and FUN_006a2c60 for the mask.


// ############################################################################
// SECTION C — WMO BSP  (AaBsp.cpp: MOBN nodes + MOBR face refs, node cache)
// ############################################################################

// ----------------------------------------------------------------------------
// MOBN node = 16 bytes (CAaBspNode). Recovered from the two traversals below:
//     +0x00  u16 flags     bit 0x04 = LEAF; low 2 bits = split plane axis
//                          (0 = X / YZ-plane, 1 = Y / XZ-plane, 2 = Z / XY-plane)
//     +0x02  i16 negChild  (child on negative side; 0xFFFF = none)
//     +0x04  i16 posChild  (child on positive side; 0xFFFF = none)
//     +0x06  u16 nFaces    (leaf: # of MOBR face refs)
//     +0x08  u32 faceStart (leaf: offset into MOBR ref array)
//     +0x0c  f32 planeDist (split plane distance from origin along axis)
// MOBR = u16 array; each entry indexes a MOVI triangle (== MOPY face index).
// ----------------------------------------------------------------------------

// FUN_00695b80 — CAaBspTree::Init (stores prebuilt MOBN/MOBR + group bbox).
//   tree[0]=tree[1]=nodeList; tree[2]=faceCount; tree[3]=faceIndices(MOBR);
//   tree[0x13..0x18] = group bbox (6 floats copied from &group[0x14]).
//   NB: 1.0.0 uses the *prebuilt* tree from the file; the AaBsp builder
//   helpers (FUN_00695800/00695980 + buildFaceIndicesNext/nodeFaceIndicesNext)
//   exist but are only used when constructing a tree from scratch.

// FUN_006965f0 — RAY/SEGMENT traversal (recursive). node = nodeList + idx*0x10.
//   Internal node: classify segment endpoints vs. split plane (axis = flags&3,
//   dist = node+0xc); if it straddles, split at the plane and recurse both
//   children (near/far ordered by sign); else recurse the single side.
//   Leaf (flags&4): FUN_00696560(node) -> gather faces.
void __thiscall FUN_006965f0(int tree, int nodeIdx, float *seg /*[p0.xyz,p1.xyz]*/)
{
  ushort *n = (ushort*)(nodeIdx*0x10 + *(int*)(tree+4));
  ushort f = *n;
  if ((f & 4) == 0) {
     // ... plane classify (axis = f&3, planeDist=*(float*)(n+6)), split, recurse
     //     n[1]=negChild n[2]=posChild ...
  } else {
     FUN_00696560(n);   // leaf -> collect
  }
}

// FUN_00696820 — AABB/BOX traversal (recursive). box = [min.xyz, max.xyz].
//   axis=flags&3, planeDist=node+0xc.  If box straddles plane -> recurse both
//   children (clamping the child box on the split axis to planeDist); else
//   descend the side the box lies on.  Leaf -> FUN_00696560.
void __thiscall FUN_00696820(int tree, int nodeIdx, int *box)
{
  int base = *(int*)(tree+4);
  ushort *n = (ushort*)(nodeIdx*0x10 + base);
  while ((*n & 4) == 0) {
     uint axis = *n & 3;
     if ((float)box[axis] <= *(float*)(n+6)) {
         if (*(float*)(n+6) <= (float)box[axis+3]) {   // straddle -> both
             if (n[2]!=0xffff){ int cb[6]; memcpy(cb,box,24); ((float*)cb)[axis]=*(float*)(n+6); FUN_00696820(n[2],cb); }
             if (n[1]!=0xffff){ int cb[6]; memcpy(cb,box,24); ((float*)cb)[axis+3]=*(float*)(n+6); FUN_00696820(n[1],cb); }
             return;
         }
         if (n[1]==0xffff) return; n = (ushort*)(n[1]*0x10 + base);
     } else {
         if (n[2]==0xffff) return; n = (ushort*)(n[2]*0x10 + base);
     }
  }
  FUN_00696560(n);   // leaf
}

// FUN_00696560 — leaf face gather.  Adds each MOBR-referenced face index to a
//   global visible-face list (DAT_00a9b840[DAT_00a9b848++]) using a per-face
//   "already added" bitmask (DAT_00a9b84c, tested via (&DAT_00835bc4)[idx&7]
//   (1<<(idx&7)) & mask[idx>>3]).  node+6=nFaces, node+8=faceStart into MOBR.
void __thiscall FUN_00696560(int tree, int node)
{
  int refs = *(int*)(tree+8);            // MOBR
  int start = *(int*)(node+8);           // faceStart
  for (uint i=0; i < *(ushort*)(node+6); i++) {
     ushort face = *(ushort*)(refs + start*2 + i*2);
     if ((mask_bit(face)) == 0) { DAT_00a9b840[DAT_00a9b848++] = face; set_mask(face); }
  }
}

// ---- BSP node CACHE ("BSP node caching" / bspcache) ----
// FUN_00696ab0 — 8-way set-associative cache keyed on node ptr.  Entry stride
//   0x2460 bytes.  On miss -> FUN_00696bf0 builds a compact per-node buffer.
//   Returns 0 if the built node overflowed (entry[+4]!=0).
// FUN_00696bf0 — build cached leaf node.  For each MOBR face of the leaf:
//     * dedups the 3 MOVI vertex indices into a local hash (local_818, 0x400)
//       building a compact vertex list (MOVT xyz, 0xc each) at cache+0x8,
//       an index triple at cache+0x18a6, count-limited to <=0x1c1 (449) verts /
//       <=0x12d (301) faces (else entry[+4]=1 = "too big, don't cache").
//     * READS MOPY FLAGS:  cachedFaceFlags[i] = MOPY.flags[face]*  &  0xff7f
//       (param_4 = group MOPY base; face*2 because MOPY entry=2 B; &~0x80).
//       stored at cache+0x1fae + i*2.
//   params: (cacheNode, srcNode, MOPY, MOVI/*+6/face*/, MOVT, ...)
void FUN_00696bf0(ushort *cache, ushort *srcNode, int node,
                  int mopy, int movt, int movi)
{
  // ... per face ...
  //   cache->faceFlags[i] = *(byte*)(mopy + faceIdx*2) & 0xff7f;   // MOPY.flags & ~0x80
}

// FUN_006a2c60 — BOX face test against a cached BSP leaf.  This is the site
//   that consumes MOPY flags as a collision filter mask:
//       queryMask = *(u16*)(this + 0x14);            // param_1[5]
//       if ((cache->faceFlags[i] & queryMask) == 0)  continue;   // FILTER
//       if ((this->tempVisited[face] & queryMask)!=0) continue;   // dedup
//       ... mark tempVisited |= 0x80 ; do the triangle/box overlap test ...
//   FUN_006a2840 is the LINE/RAY counterpart (queryMask at this+0x50).
//   Both are reached from box/line collision walks FUN_006a6580 / FUN_006a6730.


// ############################################################################
// SECTION D — MLIQ  (WMO-internal liquid)
// ############################################################################
//
// Parsed inside FUN_006c5810 under group flag 0x1000. Header = 30 bytes (0x1e):
//     data+0x00  u32 xVerts   -> group[0xf4]
//     data+0x04  u32 yVerts   -> group[0xf8]
//     data+0x08  u32 xTiles   -> group[0xfc]
//     data+0x0c  u32 yTiles   -> group[0x100]
//     data+0x10  f32 baseX    -> group[0x104]     (SMOLiquid tile-corner origin)
//     data+0x14  f32 baseY    -> group[0x108]
//     data+0x18  f32 baseZ    -> group[0x10c]
//     data+0x1c  u16 materialId-> group[0x110] (i16)   (index into MOMT)
//
//   vertexArray = data + 0x26  ( = data+8chunkhdr... actually (int)chunk+0x26 )
//     group[0x114] = &vertexArray
//     vertex stride = 8 bytes, grid = xVerts * yVerts vertices  (SMOLVert, 8 B).
//   tileFlags   = vertexArray + xVerts*yVerts*8
//     group[0x118] = &tileFlags
//     1 byte per tile, grid = xTiles * yTiles.
//   after MLIQ: FUN_006a6070() + FUN_006a4cb0()  (liquid render-vert setup;
//     FUN_006a6070 grows a CMapObjGroup VertArray to xVerts*yVerts, 0xc/vert).
//
//   NB xTiles == xVerts-1, yTiles == yVerts-1 (grid convention). This is the
//   classic pre-MH2O WMO liquid layout: (xVerts*yVerts) height/uv verts + a
//   (xTiles*yTiles) tile-flag grid, single materialId, single base corner.
// ----------------------------------------------------------------------------
void __thiscall FUN_006c5810_MLIQ(int this, int *chunk)
{
  // if (chunk[0] != 'MLIQ') ASSERT(..0x2a4,"pIffChunk->token=='MLIQ'");
  *(int*)(this+0xf4)  = chunk[2];  // xVerts
  *(int*)(this+0xf8)  = chunk[3];  // yVerts
  *(int*)(this+0xfc)  = chunk[4];  // xTiles
  *(int*)(this+0x100) = chunk[5];  // yTiles
  *(int*)(this+0x104) = chunk[6];  // baseX
  *(int*)(this+0x108) = chunk[7];  // baseY
  *(int*)(this+0x10c) = chunk[8];  // baseZ
  int matAndVerts = chunk[9];      // low u16 = materialId
  *(int*)(this+0x114) = (int)chunk + 0x26;                                  // vertexArray
  int tiles = (int)chunk + 0x26 + this->yVerts * this->xVerts * 8;          // tileFlags
  *(int*)(this+0x118) = tiles;
  *(short*)(this+0x110) = (short)matAndVerts;                               // materialId
  FUN_006a6070(); FUN_006a4cb0();
}


// ############################################################################
// SECTION E — WORLD SCENE render order  (WorldScene.cpp)
// ############################################################################
//
// Frustum STACK: 32 slots x 0xfc bytes at DAT_00a7a758, top index DAT_00a7a428.
//   FUN_0067d760 PUSH  (assert idx<31; copies parent frustum into new slot)
//   FUN_0067e390 POP   (assert idx>0; --idx)
//   FUN_0067dd30 BUILD a narrowed frustum from a 2D portal screen-rect
//                (projects the rect corners through the inverse view-proj to
//                 make the 4 side planes; writes them into the current slot).
//
// FUN_0067c460 — CWorldScene::Render (MASTER render-order entry).  Called by
//   the map-render top FUN_006742e0.  Sequence:
//     1. FUN_0058ccb0                  begin render pass
//     2. reset per-frame vis state (frustumIndex=0, vis counters)
//     3. FUN_0067d4f0                  determine camera-inside-WMO
//                                      (-> DAT_00a78e74 = camMapObj, 0 if outside;
//                                       finds camMapObjGroup via FUN_006ad600)
//     4. near-plane / fov fixups
//     5. BRANCH on DAT_00a78e74:
//          camera OUTSIDE (==0):   FUN_00681d60(); FUN_0067e3c0();     // exterior pass
//          camera INSIDE  (!=0):   FUN_00681d60(); FUN_00681690();     // interior vis
//                                  then for each CExtView (exterior seen out
//                                  through a portal; DAT_00ac4030, max 16):
//                                      FUN_00681d60(); FUN_0067e3c0();  // render exterior
//     6. opaque/world passes: FUN_0067b460, FUN_006b3170, FUN_00689420,
//        FUN_006abed0, FUN_006b6cc0
//     7. fog/env select: DAT_00aadec8 chooses interior(iVar4+0xe4) vs
//        exterior(iVar4+0xf8) fog params
//     8. transparent/effect passes: FUN_0067fa70, FUN_0067fd40, FUN_0067fff0,
//        FUN_00681030, FUN_0067f870, FUN_0067f500
//     9. FUN_0058ccc0                  end render pass
//    10. if (_DAT_00a78b1c & 0x200000) FUN_006bd620()  // portal debug overlay (TogglePortals)
//
// FUN_00681250 — exterior WMO visibility pass (called by FUN_0067e3c0).
//   Iterates ALL CMapObjDef instances (linked list at scene+0x2c); for each:
//     * frustum-cull the instance bbox (FUN_0067e340 -> FUN_006827e0, + FUN_00681fd0)
//     * set WMO world transform (FUN_005fe960 with instance matrix @+0x98..0xc0)
//     * compute camera pos+dir in WMO-local space -> globals
//         _DAT_00aadcec/f0/f4 (cam pos), _DAT_00aadcfc/d00/d04 (cam dir)
//       (these feed the portal back-face test in FUN_006b9d30)
//     * FUN_006b9350 (reset per-WMO vis) ; FUN_006b9900 (camera-OUTSIDE portal seed)
//
// FUN_00681690 — interior WMO visibility pass (called by FUN_0067c460 when the
//   camera is inside mapObjDef param_1).  Same local-space setup, then
//   FUN_006b9600 (camera-INSIDE portal seed) with the inGroups list.


// ############################################################################
// SECTION F — PORTAL VISIBILITY PROPAGATION  (screen-rect portal culling)
// ############################################################################
//
// Root object (CMapObj) portal arrays:
//     scene[0x134]  MOPV portal vertices  (C3Vector, 12 B each)
//     scene[0x138]  MOPT portals          (SMOPortal, 20 B: startVtx u16,
//                                           count u16, plane C4Plane[4 f])
//     scene[0x13c]  MOPR portal refs      (SMOPortalRef, 8 B: portalIdx u16,
//                                           groupIdx u16, side i16, filler u16)
// Per-group: group[0x2c]=portalRefStart, group[0x30]=portalRefCount.
// SPortalExt (per-portal, 0x1c bytes, array &DAT_00ab5d6c, indexed portalIdx):
//     +0x00 u16 flags (bit0 = fully behind/degenerate)
//     +0x04 f32 minX  +0x08 f32 minY  +0x0c f32 maxX  +0x10 f32 maxY  (screen rect)
//     +0x18 i32 frameStamp (DAT_00aade18)   // per-frame projection cache
//
// FUN_006ba230 — project one portal to a screen-space rect (fills SPortalExt).
//   * transform up to 12 (assert portal->count<=12) MOPV verts by view-proj
//     (FUN_0078df60), near-clip the polygon (FUN_006ba6b0 -> clipped verts+count)
//   * if clipped count==0 -> set SPortalExt.flags|=1 (portal not visible)
//   * else compute polygon plane, and the min/max screen X/Y over the clipped,
//     perspective-divided verts -> SPortalExt rect (init +FLT_MAX / -FLT_MAX).
//
// FUN_006b9d30 — CMapObj::PropagatePortalVis (THE recursion).
//   args: (scene, groupId, entryPortalId, screenRect[minX,minY,maxX,maxY], depth)
//   * depth guard: depth <= DAT_00ab5d5c (max portal recursion)
//   * group = FUN_006ad600(groupId); skip if flags&0x10000 (exterior sentinel)
//   * flags&0x1000 -> mark scene "has liquid"; flags&0x48|0x100 -> DAT_00aadec8=1 (in-WMO fog)
//   * for each MOPR ref of the group:
//       - targetGroup = ref.groupIdx; skip if == entryPortalId (no backtrack)
//       - project portal (FUN_006ba230, cached by frame stamp)
//       - skip if SPortalExt.flags & 1
//       - BACK-FACE cull: d = dot(camLocal, portalPlane.n) + portalPlane.d;
//                         if (ref.side < 0) d = -d;  if (d < 0) skip
//       - INTERSECT SPortalExt rect with incoming screenRect (clamp min/max);
//         if resulting w/h < 0.001 -> skip (portal not visible through frustum)
//       - if neighbour is exterior-connected (flags&8) and not yet queued:
//             push a CExtView (DAT_00ac4034, max 16): rect (remapped *0.5+0.5)
//             + a depth key (FUN_006ba1c0)  -> deferred exterior render
//       - FUN_0067d760(); FUN_0067dd30(intersectRect);           // push+build sub-frustum
//         FUN_006b9d30(targetGroup, groupId, intersectRect, depth+1);  // RECURSE
//         FUN_0067e390();                                         // pop
//
// FUN_006b9600 — camera-INSIDE portal seed.
//   ++frameStamp; push initial frustum; for each seed group in camera inGroups
//   list: FUN_006b9d30(group, 0xffff, fullRect{-1,-1,1,1}, 0).  pop.  Then drain
//   the CExtView list (push frustum per ext-view, recurse into exterior groups
//   flag 0x8), then mark exterior groups (flag 0x10000) via FUN_006b9cd0.
//
// FUN_006b9900 — camera-OUTSIDE portal seed.
//   ++frameStamp; for every group of the WMO:
//     flags&0x10000 (exterior) -> FUN_006b9cd0 (mark visible directly)
//     flags&0x8     (interior, reachable) -> FUN_006b9d30(group,0xffff,rect,0)
//   (all gated by an instance-bbox frustum test FUN_0067e360).
//
// FUN_006b9cd0 — mark a group visible (sets scene liquid flag; fires the
//   per-group visibility callback DAT_00ae42b4).  Actual "is visible this frame"
//   is encoded by the frame stamp (DAT_00aade18) written during the walk.
//
// FUN_0067d400 — portal-reachable group WORKLIST (dedup queue at DAT_00a7a088,
//   count DAT_00a7a084) used to avoid re-visiting a group.
