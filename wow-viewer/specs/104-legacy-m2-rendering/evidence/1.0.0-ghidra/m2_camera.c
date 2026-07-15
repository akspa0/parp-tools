// ============================================================================
// WoW 1.0.0 (build 3980) — M2 model cameras (camera tracks / cinematic cameras)
// Decompiled via GhidraMCP. 2026-07-15.
//   M2 header: cameras @0x124 (count+ofs), cameraLookup @0x12C (int16[]).
//   Camera record = 0x7c (124) bytes. Runtime instance = 0x84 (132) bytes,
//   array at CM2Model instance +0x398, +0x80 = ptr back to source record.
// ============================================================================


// ----------------------------------------------------------------------------
// FUN_00720450 — M2Camera block relocator (stride 0x7c). Confirms the record
//   layout below. For each of `count` records it relocates the sub-arrays of the
//   3 embedded M2Tracks (OLD/v256 track format: interp,gseq + 3 M2Arrays each):
//     +0x14 ranges  (FUN_007203d0)  \
//     +0x1c timestamps(FUN_0071f5f0) |  positions track (base offset 0x10)
//     +0x24 values  (FUN_00720f30)  /
//     +0x3c/0x40 ranges (8B elems)  \
//     +0x44 timestamps(FUN_0071f5f0) |  target_position track (base offset 0x38)
//     +0x4c values  (FUN_00720f30)  /
//     +0x64/0x68 ranges (8B elems)  \
//     +0x6c/0x70 timestamps (4B)     |  roll track (base offset 0x60)
//     +0x74/0x78 values (0xC elems)  /   (0xC = M2SplineKey<float>{val,in,out})
// ----------------------------------------------------------------------------

// M2Camera (v256 / format 0x100), 0x7c = 124 bytes — matches canonical layout:
//   0x00  uint32   type;          // -1 = portrait; >=0 = camera id / kind
//   0x04  float    fov;           // DIAGONAL field of view, RADIANS  (see note)
//   0x08  float    farClip;
//   0x0c  float    nearClip;
//   0x10  M2Track  positions;     // C3Vector spline — camera eye path (model space)
//   0x2c  C3Vector positionBase;  // eye pivot / fallback when track empty
//   0x38  M2Track  targetPosition;// C3Vector spline — look-at path (model space)
//   0x54  C3Vector targetBase;    // target fallback when track empty
//   0x60  M2Track  roll;          // float spline — camera roll (radians)
//
// OLD M2Track (v256 / 0x100), 0x1c = 28 bytes  (NOTE: has interp ranges!):
//   +0x00 uint16 interpolationType;  // 0 none,1 linear,2 hermite/bezier
//   +0x02 uint16 globalSequence;     // 0xffff = none
//   +0x04 M2Array interpRanges;      // count+ofs, elem = M2Range{u32 start,u32 end} (8B)
//   +0x0c M2Array timestamps;        // count+ofs, elem = uint32 ms
//   +0x14 M2Array values;            // count+ofs, elem = M2SplineKey<T>
//   M2SplineKey<T> = { T value, T inTan, T outTan }  (tangents present only if hermite;
//     storage is 3*sizeof(T): float->0xC, C3Vector->0x24. For linear only `value` is used.)
// >>> This OLD format (with the +0x04 interpRanges array) is the load-bearing
//     difference for the viewer's per-version M2 reader: Wrath+ (rev>=264) DROPS
//     interpRanges (track shrinks to 0x14) and moves timestamps/values up. A 0x100
//     reader MUST parse the 0x1c track with ranges, or every camera/anim offset is wrong.


// ----------------------------------------------------------------------------
// Camera lookup + accessors  (CM2Model / CM2Shared).
// ----------------------------------------------------------------------------

// FUN_0070b6c0 — cameraLookup: id -> record index.
//   cameraLookup is M2Array<int16> at M2 header 0x12C. Returns 0xffff if id out of range.
undefined2 FUN_0070b6c0(uint id, uint *cameraLookupArray /*{count,ofs}*/)
{
  if (cameraLookupArray[0] <= id) return 0xffff;
  return *(int16*)(cameraLookupArray[1] + id*2);
}

// FUN_0070edc0 — HasCamera(id): true if cameraLookup[id] != 0xffff and in range.
// FUN_0070ee30 — GetCameraById(id):  idx=cameraLookup[id]; return instance[idx]  (ptr @ +0x80)
// FUN_0070eeb0 — GetCameraByIndex(i):                       return instance[i]    (ptr @ +0x80)
//   both return: *(int*)(model->cameraInstances/*+0x398*/ + idx*0x84 + 0x80)
// FUN_0070ed90 — GetCameraCount() = data->cameras.count (M2 header 0x124).

// Consumers (how the game uses model cameras):
//   FUN_0053b6d0 — character PORTRAIT render: GetCameraById(0) -> render model to
//                  "Portrait1" texture through the model's own camera.
//   FUN_00743630 / FUN_007435f0 — set the active camera on a model-view UI widget
//                  (stored as a smart-ptr at widget+0x2d8; the widget renders through it).
//   FUN_00474880 — render a creature/model preview via GetCameraByIndex(0).
//
// The world/cinematic system (strings: InCinematic / OpeningCinematic / StopCinematic,
//   m_cinematic.cameraMusic) drives the main CGCamera from a model camera the same way —
//   a camera M2 is placed in the world and the view glides along its authored spline.


// ----------------------------------------------------------------------------
// EVALUATION (standard M2 track sampling — the viewer already does this for anim/taxi):
//
//   t = current time within the camera's chosen animation sequence (ms), looped.
//   eye    = sampleTrack(positions, t, positionBase)      // base used iff track empty
//   target = sampleTrack(targetPosition, t, targetBase)
//   roll   = sampleTrack(roll, t, 0)                      // radians
//   fov    = record.fov                                   // static in 1.0.0
//
//   // both eye & target are in the camera-model's LOCAL space; transform by the
//   // model's world/placement matrix M:
//   eyeW = M * eye;  targetW = M * target;
//   up   = rollAroundAxis( worldUp, (targetW-eyeW), roll );
//   view = lookAt(eyeW, targetW, up);
//   proj = perspective(vfov, aspect, nearClip, farClip);
//
//   FOV NOTE: `fov` is the DIAGONAL fov in radians. To a vertical fov for a given
//     aspect a = w/h:  vfov = 2*atan( tan(fov/2) / sqrt(1+a*a) ) * sqrt(... )  — in
//     practice the community-correct conversion for WoW is:
//         verticalFov = fov * (h / diag)  is WRONG; use:
//         hfov based; simplest faithful match: treat `fov` as the vertical fov in
//         radians * 0.6..1.0 and tune against a reference cinematic. (The client feeds
//         `fov` straight into its projection; exact aspect handling not traced here.)
// ----------------------------------------------------------------------------
