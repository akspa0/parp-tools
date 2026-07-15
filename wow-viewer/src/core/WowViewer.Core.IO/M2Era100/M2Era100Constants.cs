namespace WowViewer.Core.IO.M2Era100;

/// <summary>
/// Header field offsets and element sizes for the WoW 1.0.0 (build 3980, beta-3) M2 format.
/// Recovered via Ghidra static trace of FUN_0071e190 (MD20 parser/relocator).
/// See: specs/104-legacy-m2-rendering/research-1.0.0-ghidra-trace.md §4.
///
/// CRITICAL: 1.0.0 uses version 0x100 — the SAME version field as 1.12.1 — but the header
/// layout is completely different. 1.0.0 uses the "classic" M2 layout (M2Vertex records +
/// M2Division embedded skin profiles), while 1.12.1 uses flat parallel arrays.
/// </summary>
public static class M2Era100Constants
{
    public const uint Md20Magic = 0x3032444Du;

    public const int SignatureSizeBytes = 4;
    public const int VersionFieldSizeBytes = 4;
    public const int DispatchHeaderSizeBytes = SignatureSizeBytes + VersionFieldSizeBytes;

    /// <summary>Header extends to at least 0x144 (particles at 0x13C + 8).</summary>
    public const int MinimumHeaderSizeBytes = 0x144;

    // --- Fixed header fields ---

    public const int VersionOffset = 0x04;
    public const int NameCountOffset = 0x08;
    public const int NameOffsetOffset = 0x0C;
    public const int FlagsOffset = 0x10;

    // --- M2Array<T> fields ({count: u32, offset: u32}) ---

    public const int GlobalLoopCountOffset = 0x14;
    public const int GlobalLoopOffsetOffset = 0x18;

    public const int SequenceCountOffset = 0x1C;
    public const int SequenceOffsetOffset = 0x20;

    public const int SequenceLookupCountOffset = 0x24;
    public const int SequenceLookupOffsetOffset = 0x28;

    /// <summary>uint32[] lookup at 0x2C — likely keyBoneLookup (per Ghidra §4).</summary>
    public const int KeyBoneLookupUint32CountOffset = 0x2C;
    public const int KeyBoneLookupUint32OffsetOffset = 0x30;

    public const int BoneCountOffset = 0x34;
    public const int BoneOffsetOffset = 0x38;

    /// <summary>int16[] keyBoneLookup at 0x3C (second keyBoneLookup array).</summary>
    public const int KeyBoneLookupInt16CountOffset = 0x3C;
    public const int KeyBoneLookupInt16OffsetOffset = 0x40;

    /// <summary>M2Vertex[vertices.count] — position, normal, UV, bone weights/indices.</summary>
    public const int VertexCountOffset = 0x44;
    public const int VertexOffsetOffset = 0x48;

    /// <summary>M2Division[divisions.count] — embedded skin profiles (no external .skin).</summary>
    public const int DivisionCountOffset = 0x4C;
    public const int DivisionOffsetOffset = 0x50;

    public const int ColorCountOffset = 0x54;
    public const int ColorOffsetOffset = 0x58;

    /// <summary>M2Texture {type, flags, nameLen, nameOfs} — texture definitions.</summary>
    public const int TextureCountOffset = 0x5C;
    public const int TextureOffsetOffset = 0x60;

    public const int TextureWeightCountOffset = 0x64;
    public const int TextureWeightOffsetOffset = 0x68;

    public const int TextureTransformCountOffset = 0x6C;
    public const int TextureTransformOffsetOffset = 0x70;

    /// <summary>replaceableTexLookup — 0x54 (84) bytes per element.</summary>
    public const int ReplaceableTexLookupCountOffset = 0x74;
    public const int ReplaceableTexLookupOffsetOffset = 0x78;

    // --- Unnamed lookups (int16[] / uint32[]) at 0x7C–0xB0 ---
    // These are relocated by FUN_0071f2c0 (int16) or FUN_0071fe10 (uint32).
    // Semantic names not recovered in the Ghidra trace; ordered by header position.
    // 0x7C: int16[] — likely textureLookup (maps batch.textureComboIndex → texture index)
    public const int TextureLookupCountOffset = 0x7C;
    public const int TextureLookupOffsetOffset = 0x80;

    // 0x84: uint32[]
    public const int LookupUint32_84_CountOffset = 0x84;
    public const int LookupUint32_84_OffsetOffset = 0x88;

    // 0x8C: int16[] — likely textureUnitLookup or renderFlags lookup
    public const int LookupInt16_8C_CountOffset = 0x8C;
    public const int LookupInt16_8C_OffsetOffset = 0x90;

    // 0x94: int16[]
    public const int LookupInt16_94_CountOffset = 0x94;
    public const int LookupInt16_94_OffsetOffset = 0x98;

    // 0x9C: int16[]
    public const int LookupInt16_9C_CountOffset = 0x9C;
    public const int LookupInt16_9C_OffsetOffset = 0xA0;

    // 0xA4: int16[]
    public const int LookupInt16_A4_CountOffset = 0xA4;
    public const int LookupInt16_A4_OffsetOffset = 0xA8;

    // 0xAC: int16[]
    public const int LookupInt16_AC_CountOffset = 0xAC;
    public const int LookupInt16_AC_OffsetOffset = 0xB0;

    // --- Bounding box (gap 0xB4–0xEB, matches 1.12.1 V100 layout) ---
    public const int BoundsOffset = 0xB4;
    public const int BoundsRadiusOffset = 0xCC;
    public const int CollisionBoundsOffset = 0xD0;
    public const int CollisionBoundsRadiusOffset = 0xE8;

    // 0xEC: int16[]
    public const int LookupInt16_EC_CountOffset = 0xEC;
    public const int LookupInt16_EC_OffsetOffset = 0xF0;

    // 0xF4 / 0xFC: 0xc-stride blocks
    public const int Block_F4_CountOffset = 0xF4;
    public const int Block_F4_OffsetOffset = 0xF8;
    public const int Block_FC_CountOffset = 0xFC;
    public const int Block_FC_OffsetOffset = 0x100;

    // --- Named blocks ---

    public const int AttachmentCountOffset = 0x104;
    public const int AttachmentOffsetOffset = 0x108;

    public const int AttachmentLookupCountOffset = 0x10C;
    public const int AttachmentLookupOffsetOffset = 0x110;

    public const int EventCountOffset = 0x114;
    public const int EventOffsetOffset = 0x118;

    public const int LightCountOffset = 0x11C;
    public const int LightOffsetOffset = 0x120;

    public const int CameraCountOffset = 0x124;
    public const int CameraOffsetOffset = 0x128;

    public const int CameraLookupCountOffset = 0x12C;
    public const int CameraLookupOffsetOffset = 0x130;

    public const int RibbonCountOffset = 0x134;
    public const int RibbonOffsetOffset = 0x138;

    public const int ParticleCountOffset = 0x13C;
    public const int ParticleOffsetOffset = 0x140;

    // --- Element strides (differ from WotLK and 1.12.1!) ---

    public const int SequenceStride = 0x44;   // 68 B — same as 1.12.1 V100
    public const int BoneStride = 0x6C;       // 108 B — WotLK is 0x58
    public const int VertexStride = 0x30;     // 48 B — interleaved pos/normal/uv/bones
    public const int DivisionStride = 0x2C;   // 44 B — embedded skin profile
    public const int ColorStride = 0x38;      // 56 B — WotLK is 0x28, 1.12.1 is 0x1C
    public const int TextureStride = 0x10;    // 16 B — {type, flags, nameLen, nameOfs}
    public const int TextureWeightStride = 0x1C; // 28 B — WotLK is 0x14
    public const int TextureTransformStride = 0x1C; // 28 B — WotLK is 0x3C
    public const int ReplaceableTexLookupStride = 0x54; // 84 B
    public const int LightStride = 0xD4;       // 212 B — WotLK is 0x9C
    public const int CameraStride = 0x7C;      // 124 B
    public const int RibbonStride = 0xDC;      // 220 B
    public const int ParticleStride = 0x1F8;   // 504 B
    public const int AttachmentStride = 0x30;  // 48 B
    public const int EventStride = 0x2C;       // 44 B

    /// <summary>
    /// M2Track stride for version 0x100 (1.0.0). The OLD track format (0x1c / 28 bytes)
    /// includes an interpolation-ranges array that Wrath+ (264+) drops (track → 0x14).
    /// See: wow-1.0.0-m2-camera-tracks-2026-07-15.md §2.1.
    /// </summary>
    public const int TrackStride = 0x1C;

    // --- M2Division internal layout (0x2C = 44 bytes) ---

    public const int DivisionVertexLookupCountOffset = 0x00;
    public const int DivisionVertexLookupOffsetOffset = 0x04;
    public const int DivisionIndicesCountOffset = 0x08;
    public const int DivisionIndicesOffsetOffset = 0x0C;
    public const int DivisionUint32ArrayCountOffset = 0x10;
    public const int DivisionUint32ArrayOffsetOffset = 0x14;
    public const int DivisionSectionsCountOffset = 0x18;
    public const int DivisionSectionsOffsetOffset = 0x1C;
    public const int DivisionBatchesCountOffset = 0x20;
    public const int DivisionBatchesOffsetOffset = 0x24;

    /// <summary>Render section / geometry group record (0x20 = 32 bytes).</summary>
    public const int SectionStride = 0x20;

    /// <summary>Material/texture binding record (0x18 = 24 bytes).</summary>
    public const int BatchStride = 0x18;

    // --- M2Vertex internal layout (0x30 = 48 bytes) ---

    public const int VertexPositionOffset = 0x00;  // C3Vector (12 B)
    public const int VertexNormalOffset = 0x0C;     // C3Vector (12 B)
    public const int VertexTexCoords0Offset = 0x18; // C2Vector (8 B)
    public const int VertexTexCoords1Offset = 0x20; // C2Vector (8 B)
    public const int VertexBoneWeightsOffset = 0x28; // 4 B (packed)
    public const int VertexBoneIndicesOffset = 0x2C; // 4 B (packed)

    // --- M2Texture internal layout (0x10 = 16 bytes) ---

    public const int TextureTypeOffset = 0x00;     // uint32
    public const int TextureFlagsOffset = 0x04;    // uint32
    public const int TextureNameLenOffset = 0x08;  // uint32
    public const int TextureNameOfsOffset = 0x0C;  // uint32

    // --- M2Batch internal layout (0x18 = 24 bytes) ---

    public const int BatchFlagsOffset = 0x00;           // uint8
    public const int BatchPriorityPlaneOffset = 0x01;   // uint8
    public const int BatchShaderIdOffset = 0x02;        // uint16
    public const int BatchSkinSectionIndexOffset = 0x04; // uint16
    public const int BatchGeosetIndexOffset = 0x06;     // uint16
    public const int BatchColorIndexOffset = 0x08;       // uint16
    public const int BatchMaterialIndexOffset = 0x0A;    // uint16
    public const int BatchMaterialLayerOffset = 0x0C;    // uint16
    public const int BatchTextureCountOffset = 0x0E;    // uint16
    public const int BatchTextureComboIndexOffset = 0x10; // uint16
    public const int BatchTextureCoordComboIndexOffset = 0x12; // uint16
    public const int BatchTextureWeightComboIndexOffset = 0x14; // uint16
    public const int BatchTextureTransformComboIndexOffset = 0x16; // uint16

    // --- M2Section internal layout (0x20 = 32 bytes) ---
    // Per-field layout not fully recovered from Ghidra; using standard M2SkinSection fields.

    public const int SectionSubmeshIdOffset = 0x00;  // uint16
    public const int SectionLevelOffset = 0x02;      // uint16
    public const int SectionVertexStartOffset = 0x04; // uint16
    public const int SectionVertexCountOffset = 0x06; // uint16
    public const int SectionIndexStartOffset = 0x08;  // uint32
    public const int SectionIndexCountOffset = 0x0C;  // uint32

    // --- Global loop / name ---

    public const int GlobalLoopStride = 0x04;
    public const int NameStride = 0x01;
}