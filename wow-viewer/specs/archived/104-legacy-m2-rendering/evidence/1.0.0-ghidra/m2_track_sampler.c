/*
 * FUN_0070f6d0 — CM2Model track key resolver ("the track sampler")
 * WoW.exe 1.0.0.3980, image base 0x00400000. Decompiled via GhidraMCP 2026-07-15.
 *
 * WHY THIS FILE EXISTS: the 2026-07-15 session recorded this algorithm in
 * memory-bank/activeContext.md as prose only — no raw evidence was saved, unlike the
 * other ~40 traces. This file is the verification pass. The prose was CONFIRMED
 * accurate on every point; see NOTES for the one detail it glossed over.
 *
 * Called 58x from CM2Model::Update (FUN_0070f960).
 *
 * SIGNATURE (recovered):
 *   void FUN_0070f6d0(CM2Model *model,   // param_1
 *                     uint      time,    // param_2 — ms
 *                     int       animIndex, // param_3 — ANIMATION index, not sequence id
 *                     M2Track  *track,   // param_4 — OLD 0x1C-byte track
 *                     uint      out[3])  // param_5 — IN/OUT: see NOTES[1]
 *   out = { key0, key1, lerpT_as_float_bits }
 *
 * OLD M2Track = 0x1C (1.0.0 / 1.x). Field offsets as used below:
 *   +0x00 interpType     u16
 *   +0x02 globalSequence u16      (0xFFFF = none)
 *   +0x04 interpRanges.count  u32 \ M2Array over 8-byte M2Range{u32 start, u32 end}
 *   +0x08 interpRanges.offset u32 /
 *   +0x0C timestamps.count  u32   \ M2Array<u32>
 *   +0x10 timestamps.offset u32   /
 *   +0x14 values.count  u32       \ M2Array<T>
 *   +0x18 values.offset u32       /
 * Wrath+ drops interpRanges and nests M2Array<M2Array<T>> per sequence => track 0x1C -> 0x14.
 * THAT IS THE ENTIRE INCOMPATIBILITY. See NOTES[3].
 *
 * model+0x5c = global-sequence timer array (u32 ms per global sequence).
 */

void __thiscall FUN_0070f6d0(int param_1,uint param_2,int param_3,int param_4,uint *param_5)

{
  int iVar1;
  uint uVar2;
  uint uVar3;
  uint uVar4;
  uint *puVar5;
  uint uVar6;

  /* ---- range select: interpRanges[animIndex], or the whole flat array ---- */
  if (*(int *)(param_4 + 4) == 0) {              /* interpRanges.count == 0     */
    uVar4 = 0;                                   /*   start = 0                 */
    uVar6 = *(int *)(param_4 + 0xc) - 1;         /*   end   = timestamps.count-1*/
  }
  else {
    /* interpRanges.offset + animIndex*8 -> {start@+0, end@+4}, INCLUSIVE bounds
       into the FLAT timestamps/values arrays. */
    uVar6 = *(uint *)(*(int *)(param_4 + 8) + 4 + param_3 * 8);   /* end   */
    uVar4 = *(uint *)(*(int *)(param_4 + 8) + param_3 * 8);       /* start */
  }
  if (uVar6 <= uVar4) {                          /* single key => no interp     */
    *param_5 = uVar4;
    param_5[1] = uVar4;
    param_5[2] = 0;
    return;
  }

  /* ---- global sequence overrides the caller's time ---- */
  if (*(ushort *)(param_4 + 2) != 0xffff) {
    param_2 = *(uint *)(*(int *)(param_1 + 0x5c) + (uint)*(ushort *)(param_4 + 2) * 4);
  }

  /* ---- bracket `time` in timestamps[start..end] ---- */
  iVar1 = *(int *)(param_4 + 0x10);              /* timestamps.offset           */
  uVar3 = *param_5;                              /* NOTES[1]: cached key seed   */
  uVar2 = param_2 - *(int *)(iVar1 + uVar3 * 4); /* time - ts[cachedKey]        */
  if (uVar2 < 500) {
    /* delta in [0,500) => cache is just behind: linear forward scan */
    if (uVar3 < uVar6) {
      puVar5 = (uint *)(iVar1 + 4 + uVar3 * 4);
      do {
        if (param_2 < *puVar5) break;
        uVar3 = uVar3 + 1;
        puVar5 = puVar5 + 1;
      } while (uVar3 < uVar6);
    }
  }
  else if (uVar2 < 0xfffffe0c) {                 /* 0xfffffe0c == (uint)-500     */
    /* delta far from cache in either direction => scan from start, else bsearch */
    if (param_2 - *(int *)(iVar1 + uVar4 * 4) < 500) {
      puVar5 = (uint *)(iVar1 + 4 + uVar4 * 4);
      do {
        uVar3 = uVar4;
        if (param_2 < *puVar5) break;
        uVar4 = uVar4 + 1;
        puVar5 = puVar5 + 1;
        uVar3 = uVar4;
      } while (uVar4 < uVar6);
    }
    else {
      do {                                       /* standard bracketing bsearch  */
        uVar3 = uVar6 + uVar4 >> 1;
        if (param_2 < *(uint *)(iVar1 + uVar3 * 4)) {
          uVar6 = uVar3 - 1;
        }
        else {
          uVar4 = uVar3 + 1;
          if (param_2 < *(uint *)(iVar1 + 4 + uVar3 * 4)) break;
        }
        uVar3 = uVar4;
      } while (uVar4 < uVar6);
    }
  }
  else if (uVar4 < uVar3) {
    /* delta in [-500,0) => cache is just ahead: linear backward scan */
    puVar5 = (uint *)(iVar1 + uVar3 * 4);
    do {
      if (*puVar5 <= param_2) break;
      uVar3 = uVar3 - 1;
      puVar5 = puVar5 + -1;
    } while (uVar4 < uVar3);
  }

  /* ---- emit {k0, k1, t} ---- */
  uVar6 = uVar3 + 1;
  if (*(uint *)(param_4 + 0xc) <= uVar6) {       /* NOTES[2]: TOTAL count, not `end` */
    param_5[1] = uVar3;
    *param_5 = uVar3;
    param_5[2] = 0;
    return;
  }
  *param_5 = uVar3;
  param_5[1] = uVar6;
  iVar1 = *(int *)(*(int *)(param_4 + 0x10) + uVar3 * 4);
  param_5[2] = (uint)((float)(param_2 - iVar1) /
                     (float)(*(int *)(*(int *)(param_4 + 0x10) + uVar6 * 4) - iVar1));
  return;
}

/*
 * =============================== NOTES ===================================
 *
 * [1] out[0] IS AN IN/OUT PARAMETER — the prose missed this.
 *     On entry, param_5[0] is read as a *cached previous key index* and seeds a
 *     three-way search heuristic, selected on the UNSIGNED delta
 *     `uVar2 = time - ts[cachedKey]`:
 *         uVar2 in [0, 500)              -> forward linear scan from cache
 *         uVar2 in [500, (uint)-500)     -> far: scan from `start`, else binary search
 *         uVar2 in [(uint)-500, 0)       -> backward linear scan from cache
 *     This is PURELY a performance cache. All three branches converge on the same
 *     bracketing key. IMPLEMENTATION CONSEQUENCE: a stateless bracketing search over
 *     [start..end] is behaviourally equivalent — we do NOT need to port the cache,
 *     and we do NOT need to thread per-track mutable state through our sampler.
 *
 * [2] THE FINAL CLAMP TESTS timestamps.count (TOTAL), NOT `end`.
 *     The prose flagged this ("NB: clamps on TOTAL count") and it is confirmed here.
 *     Consequence: when k0 == end (last key of an animation range) and end+1 < total,
 *     the clamp does NOT fire — the client lerps ts[end] -> ts[end+1], i.e. across the
 *     boundary into the NEXT animation's first key. This is reachable only when
 *     time > ts[end]; normally time is clamped to the sequence duration and the last
 *     key sits at the duration, giving lerpT == 0. Reproduce the client's exact
 *     expression rather than "fixing" it to clamp at `end` — but expect it to be a
 *     no-op on well-formed data. Do not treat a diff here as a bug without a repro.
 *
 * [3] WHY 3.x CANNOT EXPRESS THIS (the actual porting problem):
 *     1.0.0 : ONE flat timestamps[] + values[] per track, sliced per animation by
 *             interpRanges[animIndex] = {firstKey, lastKey} INCLUSIVE.
 *     Wrath+: timestamps/values are M2Array<M2Array<T>> — an outer array indexed by
 *             sequence, each element its own {count,offset} pair. interpRanges GONE.
 *     Our M2TrackDefinition<T> (Core/M2/M2AnimationBlocks.cs) models ONLY the Wrath
 *     form: it carries TimestampArray/ValueArray as M2TrackArrayReference and
 *     M2TrackSampler.TryReadSequenceSlice indexes them as an array of 8-byte refs
 *     (ArrayReferenceSize = 0x08). Pointed at a 1.0.0 track, that reads the FIRST
 *     TIMESTAMP VALUE as if it were a {count,offset} pair. It is not a stride bug and
 *     not fixable by adjusting offsets — the two eras need different addressing modes.
 *
 * [4] `animIndex` (param_3) indexes interpRanges directly. RESOLVED 2026-07-15 by tracing the
 *     caller FUN_0070f960 (2082 lines; 58 call sites to this function). See NOTES[6].
 *     ANSWER: animIndex == the index into the M2Sequence array (stride 0x44) == exactly what our
 *     sampler already calls `sequenceIndex`. The mapping is DIRECT; no alias translation sits
 *     between them at this layer.
 *
 * [5] Cross-refs: bone layout M2CompBone = 0x6C from relocator FUN_0071f440
 *     (translation M2Track@0x0C, rotation@0x28, scale@0x44, pivot C3Vector@0x60).
 *     Quaternion track values decoded via FUN_00720d30.
 *
 * =================== CALLER TRACE: FUN_0070f960 (2026-07-15) ===================
 *
 * void __thiscall FUN_0070f960(int model, float *p2, float *boneIndex, float *p4, int *p5)
 * Iterates bones; `param_3` is the BONE INDEX loop counter. Per-bone animation state lives at
 * model+0x80 with stride 0x118; the bone record itself at data+0x38 with stride 0x6c.
 *
 *   iVar20  = boneIndex * 0x6c + *(int*)(data + 0x38);   // &bone[i]        <-- CONFIRMS 0x6C
 *   piVar18 = boneIndex * 0x118 + *(int*)(model + 0x80); // &boneAnimState[i], stride 0x118
 *   iVar10  = *(int*)(boneIndex*0x118 + 0xa4 + model[0x80]); // state[0x29] = current anim id
 *
 *   if (iVar10 == -1) {            // -1 => INHERIT the parent bone's animation state
 *       parent = *(ushort*)(iVar20 + 8);                 // bone.parentBone @0x08  <-- CONFIRMS
 *       piVar18[0x26] = *(int*)(&state[parent] + 0x98);  // time
 *       piVar18[0x27] = *(int*)(&state[parent] + 0x9c);  // animIndex
 *   } else {
 *       seq = iVar10 * 0x44 + *(int*)(data + 0x20);      // &sequence[animId]  <-- M2Sequence 0x44
 *       start = *(int*)(seq + 4);  end = *(int*)(seq + 8);
 *       if ((*(byte*)(seq + 0x10) & 1) == 0) {           // flags@0x10 bit0 => NON-looping
 *           t = start + ((elapsed + state[0x2e]) % (end - start));   // LOOP
 *       } else { ...clamp into [start, end]... }         // ONE-SHOT
 *       piVar18[0x27] = piVar18[0x29];                   // animIndex = the anim id
 *       piVar18[0x26] = t;                               // time
 *   }
 *
 * Then: FUN_0070f6d0(piVar18[0x26], piVar18[0x27], iVar20 + 0x28, piVar18 + 0xc);
 *       // __thiscall => model is implicit in ECX; the 4 visible args are param_2..param_5.
 *       // iVar20+0x28 = bone.rotation track, iVar20+0x44 = bone.scale track  <-- CONFIRMS layout
 *
 * [6] animIndex IS THE SEQUENCE INDEX. state[0x27] is assigned from state[0x29], which is the
 *     same value used as `animId * 0x44` to index the M2Sequence array. Therefore interpRanges
 *     is indexed in lockstep with the sequences array, i.e. interpRanges[i] describes the key
 *     span of sequences[i]. Our sampler's existing `sequenceIndex` is the correct value to feed.
 *     This retires the "largest known correctness risk" recorded in spec 105.
 *
 * [7] *** 1.0.0 SEQUENCE TIME IS A GLOBAL TIMELINE, NOT SEQUENCE-LOCAL. ***
 *     M2Sequence (0x44) carries {id u16 @0x00, variationIndex u16 @0x02, START u32 @0x04,
 *     END u32 @0x08, movespeed f32 @0x0C, flags u32 @0x10, ...}. The client computes
 *     `time = start + ((elapsed) % (end - start))` and hands that ABSOLUTE time to the sampler,
 *     which brackets it against a flat timestamp array shared by every animation. This is the
 *     other half of why interpRanges exists: one timeline, one key array, sliced per sequence.
 *     Wrath+ replaced start/end with a single `duration` @0x04 and made each sequence's keys
 *     their own array — which is why interpRanges could be dropped.
 *     IMPLEMENTATION CONSEQUENCE: our M2TrackSampler.ResolveSampleTime does `timeMs % duration`.
 *     That is the WRATH rule and is WRONG for 1.0.0, which needs `start + (timeMs % (end-start))`.
 *     Sequence time-base is therefore a THIRD era-dependent behaviour, alongside track addressing
 *     and the header layout. Verify our M2SequenceDefinition actually carries start/end for
 *     era-100 (it may only model `duration`) before implementing.
 *
 * [8] Per-bone animation state is INDEPENDENT: each bone has its own (time, animIndex), and -1
 *     means "inherit the parent bone's". state[0x31]/[0x32] hold a SECOND (time, animIndex) pair
 *     used for cross-animation BLENDING — FUN_0070f6d0 is called again with that pair and the
 *     two results are blended. Our runtime models one sequence for the whole model, so blending
 *     and per-bone animation divergence are BOTH out of scope for a first pass; note them rather
 *     than silently flattening them away.
 */
