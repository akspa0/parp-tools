namespace WowViewer.Core.IO.M2Era1121;

public static class M2Era1121Constants
{
    public const uint Md20Magic = 0x3032444Du;

    public const int SignatureSizeBytes = 4;
    public const int VersionFieldSizeBytes = 4;
    public const int DispatchHeaderSizeBytes = SignatureSizeBytes + VersionFieldSizeBytes;

    public const int MinimumHeaderSizeBytes = 0xD4;

    public const int VersionOffset = 0x04;
    public const int NameCountOffset = 0x08;
    public const int NameOffsetOffset = 0x0C;
    public const int FlagsOffset = 0x10;
    public const int GlobalLoopCountOffset = 0x14;
    public const int GlobalLoopOffsetOffset = 0x18;
    public const int SequenceCountOffset = 0x1C;
    public const int SequenceOffsetOffset = 0x20;
    public const int SequenceLookupCountOffset = 0x24;
    public const int SequenceLookupOffsetOffset = 0x28;
    public const int TexAnimCountOffset = 0x2C;
    public const int TexAnimOffsetOffset = 0x30;
    public const int BoneCountOffset = 0x34;
    public const int BoneOffsetOffset = 0x38;
    public const int ViewCountOffset = 0x3C;
    public const int ViewOffsetOffset = 0x40;
    public const int ColorCountOffset = 0x44;
    public const int ColorOffsetOffset = 0x48;
    public const int TextureCountOffset = 0x4C;
    public const int TextureOffsetOffset = 0x50;
    public const int TexWeightCountOffset = 0x54;
    public const int TexWeightOffsetOffset = 0x58;
    public const int TexLookupCountOffset = 0x5C;
    public const int TexLookupOffsetOffset = 0x60;
    public const int TexUnitLookupCountOffset = 0x64;
    public const int TexUnitLookupOffsetOffset = 0x68;
    public const int TexReplaceableLookupCountOffset = 0x6C;
    public const int TexReplaceableLookupOffsetOffset = 0x70;
    public const int TexFlagLookupCountOffset = 0x74;
    public const int TexFlagLookupOffsetOffset = 0x78;
    public const int BoundingTriCountOffset = 0x7C;
    public const int BoundingTriOffsetOffset = 0x80;
    public const int BoundingVertCountOffset = 0x84;
    public const int BoundingVertOffsetOffset = 0x88;
    public const int RenderFlagCountOffset = 0x8C;
    public const int RenderFlagOffsetOffset = 0x90;
    public const int LodTableCountOffset = 0x94;
    public const int LodTableOffsetOffset = 0x98;
    public const int CollisionCountOffset = 0x9C;
    public const int CollisionOffsetOffset = 0xA0;
    public const int AttachCountOffset = 0xA4;
    public const int AttachOffsetOffset = 0xA8;
    public const int LightCountOffset = 0xAC;
    public const int LightOffsetOffset = 0xB0;
    public const int CameraCountOffset = 0xB4;
    public const int CameraOffsetOffset = 0xB8;
    public const int CameraPerFrameCountOffset = 0xBC;
    public const int CameraPerFrameOffsetOffset = 0xC0;
    public const int RibbonCountOffset = 0xC4;
    public const int RibbonOffsetOffset = 0xC8;
    public const int ParticleCountOffset = 0xCC;
    public const int ParticleOffsetOffset = 0xD0;
    public const int UnkV101Extra0CountOffset = 0xD4;
    public const int UnkV101Extra0OffsetOffset = 0xD8;
    public const int UnkV101Extra1CountOffset = 0xDC;
    public const int UnkV101Extra1OffsetOffset = 0xE0;

    public const int HeaderSizeV100 = 0xD4;
    public const int HeaderSizeV101 = 0xE8;

    public const int SequenceStride = 0x6C;
    public const int SequenceLookupStride = 0x02;
    public const int TexAnimStride = 0x02;
    public const int BoneStridePlaceholder = 0x00;
    public const int ViewStride = 0x2C;
    public const int ColorStride = 0x1C;
    public const int TextureStride = 0x1C;
    public const int TexWeightStride = 0x08;
    public const int TexLookupStride = 0x04;
    public const int TexUnitLookupStride = 0x02;
    public const int TexReplaceableLookupStride = 0x02;
    public const int TexFlagLookupStride = 0x02;
    public const int BoundingTriStride = 0x0C;
    public const int BoundingVertStride = 0x0C;
    public const int RenderFlagStride = 0x10;
    public const int LodTableStride = 0x38;
    public const int CollisionStride = 0x02;
    public const int AttachStride = 0x0C;
    public const int LightStride = 0x0C;
    public const int CameraStride = 0x2C;
    public const int CameraPerFrameStride = 0xD4;
    public const int RibbonStride = 0x7C;
    public const int ParticleStride = 0xDC;
    public const int UnkV101Extra0Stride = 0x02;
    public const int UnkV101Extra1Stride = 0x1F8;
    public const int UnkV101Extra1SubTableCount = 29;
    public const int GlobalLoopStride = 0x04;
    public const int NameStride = 0x01;

    public const int BoundsOffset = 0xA0;
    public const int BoundsRadiusOffset = 0xB8;
}
