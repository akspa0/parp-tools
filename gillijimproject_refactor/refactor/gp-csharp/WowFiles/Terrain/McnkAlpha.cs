using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

/// <summary>
/// Alpha MCNK chunk with headerless sub-chunks (MCVT, MCNR, MCSH, MCAL, MCLQ, MCSE)
/// </summary>
public sealed class McnkAlpha : IChunkData
{
    public uint Tag => Tags.MCNK;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    // MCNK Header (128 bytes)
    public uint Flags { get; }
    public uint IndexX { get; }
    public uint IndexY { get; }
    public float Radius { get; }
    public uint NumLayers { get; }
    public uint NumDoodadRefs { get; }
    public uint OffsHeight { get; }  // MCVT offset
    public uint OffsNormal { get; }  // MCNR offset
    public uint OffsLayer { get; }   // MCLY offset
    public uint OffsRefs { get; }    // MCRF offset
    public uint OffsAlpha { get; }   // MCAL offset
    public uint SizeAlpha { get; }
    public uint OffsShadow { get; }  // MCSH offset
    public uint SizeShadow { get; }
    public uint AreaId { get; }
    public uint NumMapObjRefs { get; }
    public ushort Holes { get; }
    public ushort Pad0 { get; }
    public ushort[] PredTex { get; } = new ushort[8];
    public byte[] NoEffectDoodad { get; } = new byte[8];
    public uint OffsSndEmitters { get; } // MCSE offset
    public uint NumSndEmitters { get; }
    public uint OffsLiquid { get; }     // MCLQ offset
    public byte[] Pad1 { get; } = new byte[24];
    
    // Sub-chunk data (headerless)
    public ReadOnlyMemory<byte> McvtData { get; } // 145 floats
    public ReadOnlyMemory<byte> McnrData { get; } // 145 normals + 13 pad
    public ReadOnlyMemory<byte> McalData { get; } // Alpha data
    public ReadOnlyMemory<byte> McshData { get; } // Shadow data
    public ReadOnlyMemory<byte> MclqData { get; } // Liquid data
    public ReadOnlyMemory<byte> McseData { get; } // Sound emitters
    
    private McnkAlpha(ReadOnlyMemory<byte> rawData, long sourceOffset,
        uint flags, uint indexX, uint indexY, float radius, uint numLayers, uint numDoodadRefs,
        uint offsHeight, uint offsNormal, uint offsLayer, uint offsRefs,
        uint offsAlpha, uint sizeAlpha, uint offsShadow, uint sizeShadow,
        uint areaId, uint numMapObjRefs, ushort holes, ushort pad0,
        ushort[] predTex, byte[] noEffectDoodad, uint offsSndEmitters, uint numSndEmitters,
        uint offsLiquid, byte[] pad1,
        ReadOnlyMemory<byte> mcvtData, ReadOnlyMemory<byte> mcnrData,
        ReadOnlyMemory<byte> mcalData, ReadOnlyMemory<byte> mcshData,
        ReadOnlyMemory<byte> mclqData, ReadOnlyMemory<byte> mcseData)
    {
        RawData = rawData;
        SourceOffset = sourceOffset;
        Flags = flags;
        IndexX = indexX;
        IndexY = indexY;
        Radius = radius;
        NumLayers = numLayers;
        NumDoodadRefs = numDoodadRefs;
        OffsHeight = offsHeight;
        OffsNormal = offsNormal;
        OffsLayer = offsLayer;
        OffsRefs = offsRefs;
        OffsAlpha = offsAlpha;
        SizeAlpha = sizeAlpha;
        OffsShadow = offsShadow;
        SizeShadow = sizeShadow;
        AreaId = areaId;
        NumMapObjRefs = numMapObjRefs;
        Holes = holes;
        Pad0 = pad0;
        PredTex = predTex;
        NoEffectDoodad = noEffectDoodad;
        OffsSndEmitters = offsSndEmitters;
        NumSndEmitters = numSndEmitters;
        OffsLiquid = offsLiquid;
        Pad1 = pad1;
        McvtData = mcvtData;
        McnrData = mcnrData;
        McalData = mcalData;
        McshData = mcshData;
        MclqData = mclqData;
        McseData = mcseData;
    }
    
    public static McnkAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCNK, "Expected MCNK tag");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MCNK data");
        
        return Parse(buffer, absoluteOffset);
    }
    
    public static McnkAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCNK, "Expected MCNK tag");
        
        var chunkData = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        
        // Parse MCNK header (128 bytes)
        var span = chunkData;
        uint flags = BitConverter.ToUInt32(span[0..4]);
        uint indexX = BitConverter.ToUInt32(span[4..8]);
        uint indexY = BitConverter.ToUInt32(span[8..12]);
        float radius = BitConverter.ToSingle(span[12..16]);
        uint numLayers = BitConverter.ToUInt32(span[16..20]);
        uint numDoodadRefs = BitConverter.ToUInt32(span[20..24]);
        uint offsHeight = BitConverter.ToUInt32(span[24..28]);
        uint offsNormal = BitConverter.ToUInt32(span[28..32]);
        uint offsLayer = BitConverter.ToUInt32(span[32..36]);
        uint offsRefs = BitConverter.ToUInt32(span[36..40]);
        uint offsAlpha = BitConverter.ToUInt32(span[40..44]);
        uint sizeAlpha = BitConverter.ToUInt32(span[44..48]);
        uint offsShadow = BitConverter.ToUInt32(span[48..52]);
        uint sizeShadow = BitConverter.ToUInt32(span[52..56]);
        uint areaId = BitConverter.ToUInt32(span[56..60]);
        uint numMapObjRefs = BitConverter.ToUInt32(span[60..64]);
        ushort holes = BitConverter.ToUInt16(span[64..66]);
        ushort pad0 = BitConverter.ToUInt16(span[66..68]);
        
        var predTex = new ushort[8];
        for (int i = 0; i < 8; i++)
            predTex[i] = BitConverter.ToUInt16(span[(68 + i * 2)..(70 + i * 2)]);
        
        var noEffectDoodad = span[84..92].ToArray();
        
        uint offsSndEmitters = BitConverter.ToUInt32(span[92..96]);
        uint numSndEmitters = BitConverter.ToUInt32(span[96..100]);
        uint offsLiquid = BitConverter.ToUInt32(span[100..104]);
        var pad1 = span[104..128].ToArray();
        
        // Parse headerless sub-chunks using offsets (relative to end of MCNK header)
        ReadOnlyMemory<byte> mcvtData = ReadOnlyMemory<byte>.Empty;
        ReadOnlyMemory<byte> mcnrData = ReadOnlyMemory<byte>.Empty;
        ReadOnlyMemory<byte> mcalData = ReadOnlyMemory<byte>.Empty;
        ReadOnlyMemory<byte> mcshData = ReadOnlyMemory<byte>.Empty;
        ReadOnlyMemory<byte> mclqData = ReadOnlyMemory<byte>.Empty;
        ReadOnlyMemory<byte> mcseData = ReadOnlyMemory<byte>.Empty;
        
        if (offsHeight > 0 && offsHeight + 580 <= chunkData.Length) // 145 floats = 580 bytes
            mcvtData = chunkData.Slice((int)offsHeight, 580).ToArray();
        
        if (offsNormal > 0 && offsNormal + 448 <= chunkData.Length) // 145 * 3 + 13 = 448 bytes
            mcnrData = chunkData.Slice((int)offsNormal, 448).ToArray();
        
        if (offsAlpha > 0 && sizeAlpha > 0 && offsAlpha + sizeAlpha <= chunkData.Length)
            mcalData = chunkData.Slice((int)offsAlpha, (int)sizeAlpha).ToArray();
        
        if (offsShadow > 0 && sizeShadow > 0 && offsShadow + sizeShadow <= chunkData.Length)
            mcshData = chunkData.Slice((int)offsShadow, (int)sizeShadow).ToArray();
        
        if (offsLiquid > 0)
        {
            // MCLQ size calculation - estimate based on remaining data
            int mclqSize = Math.Min(84, (int)(chunkData.Length - offsLiquid)); // Standard MCLQ size
            if (mclqSize > 0)
                mclqData = chunkData.Slice((int)offsLiquid, mclqSize).ToArray();
        }
        
        if (offsSndEmitters > 0 && numSndEmitters > 0)
        {
            int mcseSize = (int)(numSndEmitters * 28); // Each sound emitter is 28 bytes
            if (offsSndEmitters + mcseSize <= chunkData.Length)
                mcseData = chunkData.Slice((int)offsSndEmitters, mcseSize).ToArray();
        }
        
        return new McnkAlpha(chunkData.ToArray(), absoluteOffset,
            flags, indexX, indexY, radius, numLayers, numDoodadRefs,
            offsHeight, offsNormal, offsLayer, offsRefs,
            offsAlpha, sizeAlpha, offsShadow, sizeShadow,
            areaId, numMapObjRefs, holes, pad0,
            predTex, noEffectDoodad, offsSndEmitters, numSndEmitters,
            offsLiquid, pad1,
            mcvtData, mcnrData, mcalData, mcshData, mclqData, mcseData);
    }
    
    public byte[] ToBytes()
    {
        var result = new byte[8 + RawData.Length];
        BitConverter.GetBytes(Tag).CopyTo(result, 0);
        BitConverter.GetBytes((uint)RawData.Length).CopyTo(result, 4);
        RawData.Span.CopyTo(result.AsSpan(8));
        return result;
    }
}
