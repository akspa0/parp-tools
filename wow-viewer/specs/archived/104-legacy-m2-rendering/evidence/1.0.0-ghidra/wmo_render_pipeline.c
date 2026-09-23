// ============================================================================
// WMO Rendering Pipeline — Decompiled Functions (2026-07-15)
// Recovered via GhidraMCP /decompile_function?address=0x...
// ============================================================================

// ---------------------------------------------------------------------------
// FUN_006abab0 — WMO shader loader
// Loads all 6 WMO .bls shaders by calling FUN_0058ee90 6 times
// ---------------------------------------------------------------------------
void FUN_006abab0(void)
{
  DAT_00aade18 = 0;
  DAT_00ab5d68 = 0x800;
  FUN_0058ee90();  // Called 6 times (once per shader)
  // The decompiler collapsed 6 calls into 1, but xrefs show 6 shader strings:
  //   MapObjTransSpecular.bls, MapObjTransDiffuse.bls, MapObjExtWater0.bls,
  //   MapObjMetal.bls, MapObjSpecular.bls, MapObjOverbright.bls
  _DAT_00aade84 = 1;
  return;
}

// ---------------------------------------------------------------------------
// FUN_0058ee90 — Shader loading function
// Calls virtual method on CGx device (vtable offset 0xb4) to load shader
// ---------------------------------------------------------------------------
void __fastcall FUN_0058ee90(undefined4 param_1, int param_2)
{
  if (param_2 == 0) {
    // Error: shader name is null — log error with source file path
    FUN_0063d520(0x85100000,
                 "C:\\build\\buildWoW\\ENGINE\\Source\\...",  // s_C__build_buildWoW_ENGINE_Source__0081fa3c
                 0x7cb,  // line 1995
                 "filename",  // s_filename_007fabb4
                 0, 1);
  }
  // Virtual call: (*DAT_00a1ce58)->vtable[0xb4](param_1, param_2, 1)
  // DAT_00a1ce58 = global CGx device pointer
  (**(code **)(*DAT_00a1ce58 + 0xb4))(param_1, param_2, 1);
  return;
}

// ---------------------------------------------------------------------------
// FUN_006ba9d0 — WMO group VBO setup
// Creates vertex and index buffers for a WMO group
// param_1 = CMapObjGroup*, param_2 = vertex format mode (3 or 4)
// ---------------------------------------------------------------------------
void FUN_006ba9d0(int param_1, int param_2)
{
  // Part 1: Vertex Buffer Setup (param_1 + 4 = vertexVB)
  if (*(int *)(param_1 + 4) == 0) {  // vertexVB not yet created
    uVar5 = *(param_1 + 300);  // 0x12c — vertex data size/format
    uVar1 = FUN_0058c490(uVar5);  // create vertex buffer
    FUN_006a47e0((int *)(param_1 + 4), uVar1, uVar5);  // store VB
  }

  iVar2 = *(int *)(param_1 + 4);  // vertexVB pointer
  if (iVar2 == 0) {  // VB creation failed — use immediate mode
    if (param_2 == 3) {
      // Mode 3: standard vertex format
      uVar5 = *(param_1 + 0xd4);  // vertex count
      uVar3 = *(param_1 + 0xd0);  // vertex format
    } else if (param_2 == 4) {
      // Mode 4: extended vertex format
      uVar5 = *(param_1 + 0xd4);  // vertex count
      uVar1 = *(param_1 + 0xf0);  // extended vertex format
      uVar3 = *(param_1 + 0xd0);  // vertex format
    } else {
      // Invalid mode — assert at MapObjRender.cpp:843
      ASSERT("MapObjRender.cpp", 0x34b);
      goto LAB_006baadc;
    }
    FUN_0058d6e0(0xc, uVar3, 0xc, uVar1, 0, uVar5, 8, 0, 0);  // draw vertices
  } else {
    // VB exists — set up for rendering
    *(undefined4 *)(iVar2 + 8) = DAT_007dd528;  // set global state
    iVar2 = FUN_0058cf30();  // begin pass
    if (iVar2 == 0) FUN_006a4a20();  // error handler
    FUN_0058dc40();  // set vertex stream
    iVar2 = FUN_0058ce70();  // get buffer size
    if (iVar2 != *(int *)(param_1 + 0xe8))  // validate vertex buffer size
      ASSERT("MapObjRender.cpp", 0x32e, "GxBufSize(group->vertexVB->buf)");
  }

  // Part 2: Index Buffer Setup (param_1 + 8 = indexVB)
  if (*(int *)(param_1 + 8) == 0) {  // indexVB not yet created
    if (*(int *)(param_1 + 200) == 0 || DAT_00a93790 == '\0')
      uVar5 = *(param_1 + 0x124);  // standard index format
    else
      uVar5 = *(param_1 + 0x128);  // alternate index format (32-bit?)
    FUN_006a47e0((int *)(param_1 + 8), 2, uVar5);  // create index buffer
  }

  iVar2 = *(int *)(param_1 + 8);  // indexVB pointer
  if (iVar2 == 0) {  // IB creation failed — use immediate mode
    FUN_0058db00();  // draw indices
  } else {
    // IB exists — set up for rendering
    *(undefined4 *)(iVar2 + 8) = DAT_007dd528;
    iVar2 = FUN_0058cf30();
    if (iVar2 == 0) FUN_006a4bd0();
    FUN_0058dcc0();  // set index stream
    iVar2 = FUN_0058ce70();
    if (iVar2 != *(int *)(param_1 + 0xec))  // validate index buffer size
      ASSERT("MapObjRender.cpp", 0x360, "GxBufSize(group->indexVB->buf)");
  }
}

// ---------------------------------------------------------------------------
// FUN_006babc0 — WMO batch renderer
// Iterates batches, looks up materials, sets render state, draws
// param_1 = CMapObjGroup*, param_2 = mode (0 = reset batch flags)
// ---------------------------------------------------------------------------
void FUN_006babc0(int param_1, int param_2)
{
  ASSERT(param_1 != 0, "MapObjRender.cpp", 0x372, "group");

  FUN_0058ccb0();  // begin render pass
  FUN_006ba9d0(param_1, 3);  // set up VBOs with mode 3

  iVar4 = *(int *)(param_1 + 0xd8);  // batch array pointer

  // Assert batch counts are zero (this function handles BSP-ordered batches)
  if (*(short *)(param_1 + 0x3c) != 0)  // transBatchCount
    ASSERT("MapObjRender.cpp", 0x379, "group->transBatchCount == 0");
  if (*(short *)(param_1 + 0x3e) != 0)  // intBatchCount
    ASSERT("MapObjRender.cpp", 0x37a, "group->intBatchCount == 0");

  // Iterate batches
  for (local_14 = 0; local_14 < *(ushort *)(param_1 + 0x40); local_14++) {
    // batch = iVar4 + local_14 * 0x18  (each batch is 0x18 = 24 bytes)

    if (param_2 == 0) {
      // Reset mode: clear upper nibble of batch flags
      *(byte *)(iVar4 + 0x16) &= 0x0f;
    }

    // Check if batch should be rendered
    if ((*(byte *)(iVar4 + 0x16) & 0xf0) == 0 && !FUN_006ba940(iVar4)) {
      // Mark batch as processed
      *(byte *)(iVar4 + 0x16) |= 0xf0;

      // Look up material: materialIndex * 0x40 + materialArray
      puVar5 = (uint *)((uint)*(byte *)(iVar4 + 0x17) * 0x40
                        + *(int *)(local_10 + 0x1d8));

      // Check visibility (FUN_0044ecb0)
      if (FUN_0044ecb0(0) != 0) {
        // Set up render state based on material flags
        if ((*puVar5 & 0x10) == 0) {
          // No special flag — use default color
          local_8 |= 1;
          local_18 = 0;
          puVar3 = &local_18;
        } else {
          // Special flag — use material color (puVar5 + 5)
          puVar3 = puVar5 + 5;
        }

        FUN_0058ca90(*puVar3);  // set color
        FUN_0058cb30();  // push render state
        FUN_0058df10(1, 0, 0, 0, 0, 0, 1);  // set blend/alpha
        FUN_0058e7c0();  // apply state

        // Toggle render state flags based on material flags
        local_c ^= (~*puVar5 >> 3 ^ local_c) & 8;   // bit 3
        local_c ^= (~*puVar5 >> 3 ^ local_c) & 0x10; // bit 4
        FUN_0058e650(local_c);  // set render state
        FUN_0058cb70();  // push state

        // Two-pass rendering (if material has second pass)
        if (puVar5[1] != 0) {  // passCount
          if (puVar5[1] == 1) {
            if (DAT_00aadec1 != '\0') {  // global flag (specular enabled?)
              FUN_0058cb70();
              FUN_0058ca90(0xffffffff);  // white
              FUN_0058cae0(DAT_007dd288);  // set specular state
            }
          } else {
            ASSERT("MapObjRender.cpp", 0x3a8);  // invalid pass count
          }
        }

        // Draw the batch
        FUN_006baea0();  // fill draw parameters
        FUN_0058dd90();  // execute draw

        // Restore state after second pass
        if (puVar5[1] != 0) {
          if (puVar5[1] == 1) {
            if (DAT_00aadec1 != '\0') {
              FUN_0058cb70();
              FUN_0058ca90(0);  // black
              FUN_0058cae0(0);  // reset specular
            }
          } else {
            ASSERT("MapObjRender.cpp", 0x3bd);
          }
        }
      }
    }

    iVar4 += 0x18;  // next batch (24 bytes per batch)
  }

  FUN_0058ccc0();  // end render pass
}

// ---------------------------------------------------------------------------
// FUN_006baea0 — Batch draw parameter setup
// Fills draw parameters from batch struct
// param_1 = draw params output, param_2 = batch struct
// ---------------------------------------------------------------------------
undefined4 * __fastcall FUN_006baea0(undefined4 *param_1, int param_2)
{
  // Determine primitive type
  uVar5 = 3;  // GL_TRIANGLES
  if (DAT_00a93790 != '\0' && (*(byte *)(param_2 + 0x16) & 1) != 0) {
    uVar5 = 4;  // GL_TRIANGLE_STRIP
  }

  // Read batch fields and fill draw parameters
  uVar1 = *(undefined2 *)(param_2 + 0x10);  // startIndex
  uVar2 = *(undefined2 *)(param_2 + 0x14);  // primCount
  uVar3 = *(undefined2 *)(param_2 + 0x12);  // count
  uVar4 = *(undefined4 *)(param_2 + 0x0c);   // baseVertex

  *param_1 = uVar5;                    // [0x00] primitiveType
  param_1[1] = uVar4;                  // [0x04] baseVertex
  *(undefined2 *)(param_1 + 2) = uVar1;   // [0x08] startIndex
  *(undefined2 *)((int)param_1 + 10) = uVar3; // [0x0a] count
  *(undefined2 *)(param_1 + 3) = uVar2;   // [0x0c] primCount

  return param_1;
}

// ============================================================================
// RECOVERED STRUCT LAYOUTS
// ============================================================================

/*
  WMO Batch Struct (0x18 = 24 bytes):
  Offset  Size    Field
  0x00    ?       (unknown — possibly BSP node reference)
  0x04    ?       (unknown)
  0x08    ?       (unknown)
  0x0c    uint32  baseVertex (starting vertex in VBO)
  0x10    uint16  startIndex (starting index in IBO)
  0x12    uint16  count (number of vertices or indices)
  0x14    uint16  primCount (number of primitives)
  0x16    byte    flags (bit 0 = use triangle strip, upper nibble = render state)
  0x17    byte    materialIndex (index into material array)

  WMO Material Struct (0x40 = 64 bytes):
  Offset  Size    Field
  0x00    uint32  flags (bit 3 = ?, bit 4 = 0x10 = use material color)
  0x04    uint32  passCount (0 = single pass, 1 = two-pass with specular)
  0x05+   uint32* color (at offset 0x14 = puVar5 + 5, when flag 0x10 is set)

  WMO Group (CMapObjGroup) Fields:
  Offset  Size    Field
  0x04    ptr     vertexVB (vertex buffer object)
  0x08    ptr     indexVB (index buffer object)
  0x3c    int16   transBatchCount (transparent batch count)
  0x3e    int16   intBatchCount (interior/opaque batch count)
  0x40    uint16  batchCount (total BSP-ordered batch count)
  0xc8    int32   flag (use alternate index format?)
  0xd0    uint32  vertexFormat
  0xd4    uint32  vertexCount
  0xd8    ptr     batchArray (array of 0x18-byte batch structs)
  0xe8    int32   vertexBufferSize (expected VBO size)
  0xec    int32   indexBufferSize (expected IBO size)
  0xf0    uint32  extendedVertexFormat (mode 4)
  0x124   uint32  standardIndexFormat
  0x128   uint32  alternateIndexFormat (32-bit indices?)
  0x12c   uint32  vertexDataSize/Format
  0x1d8   ptr     materialArray (array of 0x40-byte material structs)

  Draw Parameters Struct:
  Offset  Size    Field
  0x00    uint32  primitiveType (3 = GL_TRIANGLES, 4 = GL_TRIANGLE_STRIP)
  0x04    uint32  baseVertex
  0x08    uint16  startIndex
  0x0a    uint16  count
  0x0c    uint16  primCount

  Global Variables:
  DAT_00a93790   — bool: use triangle strips / 32-bit indices
  DAT_00aadec1   — bool: specular enabled
  DAT_007dd528   — global render state value
  DAT_007dd288   — specular render state
  DAT_00a1ce58   — ptr: CGx device pointer (vtable at offset 0xb4 = shader load)
*/