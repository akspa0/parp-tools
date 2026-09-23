// ============================================================================
// WMO Transparent Batch Renderer — FUN_006baf70 (Decompiled 2026-07-15)
// Recovered via GhidraMCP v5.14.2 on port 8089
// ============================================================================

// ---------------------------------------------------------------------------
// FUN_006ba940 — Batch visibility check (frustum culling)
// Reads 6 shorts from batch struct (bounding box: minXYZ, maxXYZ)
// ---------------------------------------------------------------------------
bool FUN_006ba940(short *param_1)
{
  // Convert 6 shorts to floats and pass to frustum check
  int iVar1 = FUN_0067e340(
    (float)(int)*param_1,       // min X
    (float)(int)param_1[1],     // min Y
    (float)(int)param_1[2],     // min Z
    (float)(int)param_1[3],     // max X
    (float)(int)param_1[4],     // max Y
    (float)(int)param_1[5]      // max Z
  );
  return iVar1 != 0;  // true = visible
}

// FUN_0067e340 wraps FUN_006827e0 (actual frustum cull)
// Returns true if bounding box is NOT culled (visible)

// ---------------------------------------------------------------------------
// FUN_006baf70 — WMO Transparent Batch Renderer
// param_1 = context object (CMapObjDef or similar)
// param_2 = CMapObjGroup pointer
// param_3 = mode (0 = reset batch flags)
// ---------------------------------------------------------------------------
void __thiscall FUN_006baf70(int param_1, int param_2, int param_3)
{
  local_1c = param_1;  // context object

  ASSERT(param_2 != 0, "MapObjRender.cpp", 0x3c9, "group");

  FUN_0058ccb0();  // begin render pass
  FUN_006ba9d0(param_2, 4);  // set up VBOs with mode 4 (EXTENDED vertex format)

  // Get camera/scene context
  iVar4 = FUN_006d2510();
  local_18 = iVar4;

  // Set up fog and ambient color if context matches
  if (*(char *)(iVar4 + 0xf0) != '\0' && param_1 == DAT_00a78d6c) {
    FUN_0058cae0(*(undefined4 *)(iVar4 + 0xfc));   // fog far
    FUN_0058cae0(*(undefined4 *)(iVar4 + 0x100));  // fog near
    FUN_0058ca90(*(undefined4 *)(iVar4 + 0xf8));   // fog color
  }

  iVar4 = *(int *)(param_2 + 0xd8);  // batch array pointer

  // Iterate ALL batches (total count at +0x138, not just +0x40)
  for (local_10 = 0; local_10 < *(int *)(param_2 + 0x138); local_10++) {

    if (param_3 == 0) {
      *(byte *)(iVar4 + 0x16) &= 0x0f;  // reset batch flags
    }

    // Check visibility (frustum cull)
    if ((*(byte *)(iVar4 + 0x16) & 0xf0) == 0 && !FUN_006ba940(iVar4)) {
      *(byte *)(iVar4 + 0x16) |= 0xf0;  // mark as processed

      // Look up material: materialIndex * 0x40 + materialArray
      puVar7 = (uint *)((uint)*(byte *)(iVar4 + 0x17) * 0x40
                        + *(int *)(local_1c + 0x1d8));

      // Check visibility (FUN_0044ecb0)
      if (FUN_0044ecb0(0) != 0) {

        // Set color based on material flags
        if ((*puVar7 & 0x10) == 0) {
          // Default color (black)
          local_8 |= 1;
          local_2c = 0;
          puVar6 = &local_2c;
        } else {
          // Material color (at material + 0x14)
          puVar6 = puVar7 + 5;
        }
        FUN_0058ca90(*puVar6);  // set color

        // MATERIAL FLAG 0x20: Apply vertex colors (MOCV) from context
        if ((*puVar7 & 0x20) != 0) {
          // Set fog color from context (bytes converted to float via *1/255)
          *(float *)(DAT_00a8732c + 0xbc) = (float)*(byte *)(local_18 + 0x11a) * 0.003921569;
          *(float *)(DAT_00a8732c + 0xc0) = (float)*(byte *)(local_18 + 0x119) * 0.003921569;
          *(float *)(DAT_00a8732c + 0xc4) = (float)*(byte *)(local_18 + 0x118) * 0.003921569;

          // Set ambient color from context
          *(float *)(DAT_00a8732c + 0xb0) = (float)*(byte *)(local_18 + 0x11e) * 0.003921569;
          *(float *)(DAT_00a8732c + 0xb4) = (float)*(byte *)(local_18 + 0x11d) * 0.003921569;
          *(float *)(DAT_00a8732c + 0xb8) = (float)*(byte *)(local_18 + 0x11c) * 0.003921569;

          FUN_0071ca90(0);  // apply vertex colors
        }

        // Set blend/alpha state
        FUN_0058df10(1, 0, 0, 0, 0, 0, 1);
        FUN_0058e7c0();  // apply state

        // Toggle render state bits 3 and 4 based on material flags
        local_14 ^= (~*puVar7 >> 3 ^ local_14) & 8;
        local_14 ^= (~*puVar7 >> 3 ^ local_14) & 0x10;
        FUN_0058e650(local_14);  // set render state
        FUN_0058cb70();  // push state

        // BATCH RENDERING ORDER:
        // Batch array is ordered: [transparent] [interior] [other]
        // transBatchCount at +0x3c, intBatchCount at +0x3e

        if (local_10 < *(ushort *)(param_2 + 0x3c)) {
          // TRANSPARENT BATCH (index < transBatchCount)
          FUN_0058cb70();  // push
          FUN_0058cb30();  // push
          FUN_006baea0();  // fill draw params
          FUN_0058dd90();  // draw
          FUN_0058cb30();  // push
          FUN_006baea0();  // fill draw params (second draw?)
          FUN_0058dd90();  // draw
        }
        else if ((uint)*(ushort *)(param_2 + 0x3e) + (uint)*(ushort *)(param_2 + 0x3c) <= local_10) {
          // OTHER BATCH (index >= transBatchCount + intBatchCount)
          FUN_0058cb30();
          FUN_0058cb70();
          FUN_006baea0();  // draw
          FUN_0058dd90();
        }
        else {
          // INTERIOR BATCH (transBatchCount <= index < transBatchCount + intBatchCount)
          FUN_0058cb30();
          FUN_0058cb70();
          FUN_006baea0();  // draw
          FUN_0058dd90();

          // TWO-PASS SPECULAR (if enabled and material has passes)
          if (DAT_00aadce4 == 1 && puVar7[1] != 0) {
            FUN_0058cb30();
            FUN_0058ca50();  // set specular mode

            if (puVar7[1] == 1) {
              // Pass count 1: standard specular
              if (DAT_00aadec1 != '\0') {
                FUN_0058cb70();
                FUN_0058ca90(0xffffffff);  // white
                FUN_0058cae0(0x41600000);  // 8.0f (specular intensity)
              }
            }
            else if (puVar7[1] == 2) {
              // Pass count 2: extended specular
              if (DAT_00aadec1 != '\0') {
                FUN_0058cb70();
                FUN_0058ca90(0xffffffff);  // white
                FUN_0058cae0(0x41600000);  // 8.0f
              }
            }
            else {
              ASSERT("MapObjRender.cpp", 0x43c);  // invalid pass count
            }

            FUN_0058cb30();
            FUN_006baea0();  // draw specular pass
            FUN_0058dd90();
            FUN_0058cb30();
            FUN_0058ca50();  // restore
          }
        }

        // RESTORE vertex colors after rendering
        if ((*puVar7 & 0x20) != 0) {
          // Restore original fog/ambient colors
          *(float *)(DAT_00a8732c + 0xbc) = (float)*(byte *)(local_18 + 0x112) * 0.003921569;
          *(float *)(DAT_00a8732c + 0xc0) = (float)*(byte *)(local_18 + 0x111) * 0.003921569;
          *(float *)(DAT_00a8732c + 0xc4) = (float)*(byte *)(local_18 + 0x110) * 0.003921569;
          // ... (rest of restoration)
        }
      }
    }

    iVar4 += 0x18;  // next batch (24 bytes per batch)
  }

  FUN_0058ccc0();  // end render pass
}

// ============================================================================
// KEY FINDINGS FROM TRANSPARENT BATCH RENDERER
// ============================================================================

/*
  BATCH ARRAY ORDERING:
  The batch array at group+0xd8 is ordered:
    [0..transBatchCount-1]                    = transparent batches
    [transBatchCount..transBatchCount+intBatchCount-1] = interior (opaque) batches
    [transBatchCount+intBatchCount..totalCount-1]     = other batches

  Group fields:
    +0x3c = transBatchCount (uint16)
    +0x3e = intBatchCount (uint16)
    +0x138 = totalBatchCount (int32) — used by transparent renderer
    +0x40 = BSP-ordered batch count (uint16) — used by opaque renderer

  MATERIAL FLAGS (at material+0x00):
    bit 4 (0x10) = use material color (from material+0x14) vs default black
    bit 5 (0x20) = use vertex colors (MOCV) from context object
    bit 3 = affects render state (toggled in state setup)

  MATERIAL PASS COUNT (at material+0x04):
    0 = single pass (no specular)
    1 = two-pass specular (specular value = 8.0f)
    2 = extended two-pass specular (same value, different code path)

  CONTEXT OBJECT (CMapObjDef or similar) FIELDS:
    +0x110 = vertex color R (byte, restored after rendering)
    +0x111 = vertex color G
    +0x112 = vertex color B
    +0x118 = fog color G
    +0x119 = fog color B
    +0x11a = fog color R
    +0x11c = ambient color R
    +0x11d = ambient color G
    +0x11e = ambient color B
    +0x1d8 = material array pointer

  GLOBAL STATE:
    DAT_00a8732c = render state object
      +0xb0 = ambient color R (float)
      +0xb4 = ambient color G (float)
      +0xb8 = ambient color B (float)
      +0xbc = fog color R (float)
      +0xc0 = fog color G (float)
      +0xc4 = fog color B (float)
    DAT_00aadce4 = specular enabled flag (== 1)
    DAT_00aadec1 = specular supported flag
    DAT_00a78d6c = current context object pointer

  VERTEX FORMAT:
    Mode 3 (opaque renderer) = standard vertex format
    Mode 4 (transparent renderer) = EXTENDED vertex format (includes vertex colors)

  CONVERSION:
    0.003921569 = 1.0/255.0 (byte to normalized float conversion)
    0x41600000 = 8.0f (specular intensity)
*/