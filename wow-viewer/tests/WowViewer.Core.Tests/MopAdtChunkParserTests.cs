using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public class MopAdtChunkParserTests
{
    [Fact]
    public void ParseMdidChunk_ExtractsFileDataIds()
    {
        byte[] buffer = new byte[12];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0, 4), 123456);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(4, 4), 234567);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(8, 4), 345678);

        List<uint> ids = MopAdtChunkParser.ParseMdidChunk(buffer);
        Assert.Equal(3, ids.Count);
        Assert.Equal(123456u, ids[0]);
        Assert.Equal(234567u, ids[1]);
        Assert.Equal(345678u, ids[2]);
    }

    [Fact]
    public void ParseMhidChunk_ExtractsHeightFileDataIds()
    {
        byte[] buffer = new byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0, 4), 987654);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(4, 4), 876543);

        List<uint> ids = MopAdtChunkParser.ParseMhidChunk(buffer);
        Assert.Equal(2, ids.Count);
        Assert.Equal(987654u, ids[0]);
        Assert.Equal(876543u, ids[1]);
    }

    [Fact]
    public void ParseMcxhChunk_ExtractsHeightScaleAndOffsets()
    {
        byte[] buffer = new byte[16];
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(0, 4), 2.5f);
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(4, 4), -0.15f);

        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(8, 4), 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(12, 4), 0.0f);

        List<MopHeightBlendLayer> layers = MopAdtChunkParser.ParseMcxhChunk(buffer);
        Assert.Equal(2, layers.Count);
        Assert.Equal(2.5f, layers[0].HeightScale, 2);
        Assert.Equal(-0.15f, layers[0].HeightOffset, 2);
        Assert.Equal(1.0f, layers[1].HeightScale, 2);
        Assert.Equal(0.0f, layers[1].HeightOffset, 2);
    }

    [Fact]
    public void ParseMtexChunk_ParsesNullTerminatedStringList()
    {
        string[] textures = ["tileset/terrain/grass.blp", "tileset/terrain/dirt.blp", "tileset/terrain/rock.blp"];
        using var ms = new MemoryStream();
        foreach (var tex in textures)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(tex);
            ms.Write(bytes);
            ms.WriteByte(0);
        }

        List<string> parsed = MopAdtChunkParser.ParseMtexChunk(ms.ToArray());
        Assert.Equal(3, parsed.Count);
        Assert.Equal("tileset/terrain/grass.blp", parsed[0]);
        Assert.Equal("tileset/terrain/dirt.blp", parsed[1]);
        Assert.Equal("tileset/terrain/rock.blp", parsed[2]);
    }

    [Fact]
    public void ParseMddfChunk_ParsesM2PlacementsWithScaling()
    {
        byte[] buffer = new byte[36];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0, 4), 101); // nameId
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(4, 4), 5001); // uniqueId
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(8, 4), 1000f); // X
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(12, 4), 2000f); // Y
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(16, 4), 300f); // Z
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(20, 4), 0f); // rotX
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(24, 4), 90f); // rotY
        BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(28, 4), 0f); // rotZ
        BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan(32, 2), 2048); // scale = 2048 / 1024 = 2.0f
        BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan(34, 2), 0x01); // flags

        List<MopObjectPlacement> placements = MopAdtChunkParser.ParseMddfChunk(buffer);
        Assert.Single(placements);
        Assert.False(placements[0].IsWmo);
        Assert.Equal(101u, placements[0].NameId);
        Assert.Equal(5001u, placements[0].UniqueId);
        Assert.Equal(new Vector3(1000f, 2000f, 300f), placements[0].Position);
        Assert.Equal(2.0f, placements[0].Scale, 2);
    }

    [Fact]
    public void Mcnk_HeaderlessSubchunkStream_ParsesMclyAndMcalCorrectly()
    {
        // Build synthetic headerless MCNK payload: MCLY chunk + MCAL chunk
        using var ms = new MemoryStream();

        // 1. MCLY chunk: 2 layers (32 bytes)
        ms.Write(Encoding.ASCII.GetBytes("MCLY"));
        ms.Write(BitConverter.GetBytes(32)); // size
        // Layer 0: texture 0, no alpha
        ms.Write(BitConverter.GetBytes(0u)); // textureId
        ms.Write(BitConverter.GetBytes(0u)); // flags
        ms.Write(BitConverter.GetBytes(0u)); // alpha offset
        ms.Write(BitConverter.GetBytes(0u)); // effectId
        // Layer 1: texture 1, alpha offset 0
        ms.Write(BitConverter.GetBytes(1u)); // textureId
        ms.Write(BitConverter.GetBytes(0x200u)); // flags (compressed alpha)
        ms.Write(BitConverter.GetBytes(0u)); // alpha offset
        ms.Write(BitConverter.GetBytes(0u)); // effectId

        // 2. MCAL chunk: 64 bytes
        ms.Write(Encoding.ASCII.GetBytes("MCAL"));
        ms.Write(BitConverter.GetBytes(64)); // size
        ms.Write(new byte[64]); // dummy alpha bytes

        byte[] headerlessMcnkBytes = ms.ToArray();

        var mcnk = new WowViewer.Core.IO.Lk.Mcnk(headerlessMcnkBytes);

        Assert.NotNull(mcnk.TextureLayers);
        Assert.Equal(2, mcnk.TextureLayers.Count);
        Assert.Equal(0u, mcnk.TextureLayers[0].TextureId);
        Assert.Equal(1u, mcnk.TextureLayers[1].TextureId);
        Assert.NotNull(mcnk.McalRawData);
        Assert.Equal(64, mcnk.McalRawData.Length);
    }
}
