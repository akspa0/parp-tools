namespace WowViewer.Core.IO.M2Era1121;

public sealed class M2Era1121Layout
{
    public M2Era1121Version Version { get; }
    public int SequenceStride { get; }
    public int BoundsOffset { get; }
    public int BoundsRadiusOffset { get; }
    public int CollisionBoundsOffset { get; }
    public int CollisionBoundsRadiusOffset { get; }
    
    // Cameras, ribbons, particles
    public int CameraCountOffset { get; }
    public int CameraOffsetOffset { get; }
    public int CameraPerFrameCountOffset { get; }
    public int CameraPerFrameOffsetOffset { get; }
    public int RibbonCountOffset { get; }
    public int RibbonOffsetOffset { get; }
    public int ParticleCountOffset { get; }
    public int ParticleOffsetOffset { get; }
    public int UnkV101Extra0CountOffset { get; }
    public int UnkV101Extra0OffsetOffset { get; }
    public int UnkV101Extra1CountOffset { get; }
    public int UnkV101Extra1OffsetOffset { get; }

    // Geometry tables
    public int VertexCountOffset { get; }
    public int VertexOffsetOffset { get; }
    public int PositionCountOffset { get; }
    public int PositionOffsetOffset { get; }
    public int NormalCountOffset { get; }
    public int NormalOffsetOffset { get; }
    public int UvCountOffset { get; }
    public int UvOffsetOffset { get; }
    public int TriangleCountOffset { get; }
    public int TriangleOffsetOffset { get; }
    public int BatchCountOffset { get; }
    public int BatchOffsetOffset { get; }
    public int Extra0CountOffset { get; }
    public int Extra0OffsetOffset { get; }
    public int Extra1CountOffset { get; }
    public int Extra1OffsetOffset { get; }
    public int Extra2CountOffset { get; }
    public int Extra2OffsetOffset { get; }

    public M2Era1121Layout(M2Era1121Version version)
    {
        Version = version;
        if (version == M2Era1121Version.V100)
        {
            SequenceStride = 0x44;
            BoundsOffset = 0xB4;
            BoundsRadiusOffset = 0xCC;
            CollisionBoundsOffset = 0xD0;
            CollisionBoundsRadiusOffset = 0xE8;
            
            CameraCountOffset = -1;
            CameraOffsetOffset = -1;
            CameraPerFrameCountOffset = -1;
            CameraPerFrameOffsetOffset = -1;
            RibbonCountOffset = -1;
            RibbonOffsetOffset = -1;
            ParticleCountOffset = -1;
            ParticleOffsetOffset = -1;
            UnkV101Extra0CountOffset = -1;
            UnkV101Extra0OffsetOffset = -1;
            UnkV101Extra1CountOffset = -1;
            UnkV101Extra1OffsetOffset = -1;

            VertexCountOffset = 0xEC;
            VertexOffsetOffset = 0xF0;
            PositionCountOffset = 0xF4;
            PositionOffsetOffset = 0xF8;
            NormalCountOffset = 0xFC;
            NormalOffsetOffset = 0x100;
            UvCountOffset = 0x104;
            UvOffsetOffset = 0x108;
            TriangleCountOffset = 0x10C;
            TriangleOffsetOffset = 0x110;
            BatchCountOffset = 0x114;
            BatchOffsetOffset = 0x118;
            Extra0CountOffset = 0x11C;
            Extra0OffsetOffset = 0x120;
            Extra1CountOffset = 0x124;
            Extra1OffsetOffset = 0x128;
            Extra2CountOffset = 0x12C;
            Extra2OffsetOffset = 0x130;
        }
        else // V101
        {
            SequenceStride = 0x6C;
            
            CameraCountOffset = 0xB4;
            CameraOffsetOffset = 0xB8;
            CameraPerFrameCountOffset = 0xBC;
            CameraPerFrameOffsetOffset = 0xC0;
            RibbonCountOffset = 0xC4;
            RibbonOffsetOffset = 0xC8;
            ParticleCountOffset = 0xCC;
            ParticleOffsetOffset = 0xD0;
            UnkV101Extra0CountOffset = 0xD4;
            UnkV101Extra0OffsetOffset = 0xD8;
            UnkV101Extra1CountOffset = 0xDC;
            UnkV101Extra1OffsetOffset = 0xE0;

            BoundsOffset = 0xE4;
            BoundsRadiusOffset = 0xFC;
            CollisionBoundsOffset = 0x100;
            CollisionBoundsRadiusOffset = 0x118;

            VertexCountOffset = 0x11C;
            VertexOffsetOffset = 0x120;
            PositionCountOffset = 0x124;
            PositionOffsetOffset = 0x128;
            NormalCountOffset = 0x12C;
            NormalOffsetOffset = 0x130;
            UvCountOffset = 0x134;
            UvOffsetOffset = 0x138;
            TriangleCountOffset = 0x13C;
            TriangleOffsetOffset = 0x140;
            BatchCountOffset = 0x144;
            BatchOffsetOffset = 0x148;
            Extra0CountOffset = 0x14C;
            Extra0OffsetOffset = 0x150;
            Extra1CountOffset = 0x154;
            Extra1OffsetOffset = 0x158;
            Extra2CountOffset = 0x15C;
            Extra2OffsetOffset = 0x160;
        }
    }
}
