using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtTextureReaderTests
{
    [Fact]
    public void Read_SyntheticRootAdt_ProducesChunkLayerDetails()
    {
        byte[] packedAlpha = Enumerable.Repeat((byte)0x10, 2048).ToArray();
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MTEX", CreateStringBlock("base.blp", "snow.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateRootTextureMcnkPayload(indexX: 7, indexY: 3, flags: 0x8000u, mclyPayload: CreateMclyPayload([0u, 0u], [0u, 0u]), mcalPayload: packedAlpha)),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_7_3.adt");
        AdtTextureFile textureFile = AdtTextureReader.Read(stream, summary);

        Assert.Equal(MapFileKind.Adt, textureFile.Kind);
        Assert.Equal(AdtMcalDecodeProfile.LichKingStrict, textureFile.DecodeProfile);
        Assert.Single(textureFile.Chunks);
        AdtTextureChunk chunk = textureFile.Chunks[0];
        Assert.Equal(7, chunk.ChunkX);
        Assert.Equal(3, chunk.ChunkY);
        Assert.True(chunk.DoNotFixAlphaMap);
        Assert.Equal(2, chunk.Layers.Count);
        Assert.Equal("base.blp", chunk.Layers[0].TexturePath);
        Assert.Equal("snow.blp", chunk.Layers[1].TexturePath);
        Assert.Equal(AdtMcalAlphaEncoding.Packed4Bit, chunk.Layers[1].DecodedAlpha!.Encoding);
        Assert.False(chunk.Layers[1].DecodedAlpha!.AppliedFixup);
    }

    [Fact]
    public void Read_SyntheticTexAdt_ProducesChunkLayerDetails()
    {
        byte[] packedAlpha = Enumerable.Repeat((byte)0x10, 2048).ToArray();
        byte[] compressedAlpha =
        [
            0x81, 0x44,
            0x01, 0x55,
        ];

        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MTEX", CreateStringBlock("base.blp", "snow.blp", "rock.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateTexChunkPayload(
                CreateMclyPayload([0u, 0x100u, 0x300u], [0u, 0u, (uint)packedAlpha.Length]),
                [.. packedAlpha, .. compressedAlpha])),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_0_0_tex0.adt");
        AdtTextureFile textureFile = AdtTextureReader.Read(stream, summary);

        Assert.Equal(MapFileKind.AdtTex, textureFile.Kind);
        Assert.Equal(AdtMcalDecodeProfile.Cataclysm400, textureFile.DecodeProfile);
        Assert.Single(textureFile.Chunks);
        Assert.Equal(3, textureFile.TextureNames.Count);
        AdtTextureChunk chunk = textureFile.Chunks[0];
        Assert.Equal(0, chunk.ChunkIndex);
        Assert.Equal(3, chunk.Layers.Count);
        Assert.Equal("base.blp", chunk.Layers[0].TexturePath);
        Assert.Null(chunk.Layers[0].DecodedAlpha);
        Assert.Equal(AdtMcalAlphaEncoding.Packed4Bit, chunk.Layers[1].DecodedAlpha!.Encoding);
        Assert.Equal(AdtMcalAlphaEncoding.Compressed, chunk.Layers[2].DecodedAlpha!.Encoding);
    }

    [Fact]
    public void Read_DevelopmentRootAdt_AcceptsRootAdtAndReportsStableSplitFamilySignals()
    {
        AdtTextureFile textureFile = AdtTextureReader.Read(MapTestPaths.DevelopmentRootAdtPath);

        Assert.Equal(MapFileKind.Adt, textureFile.Kind);
        Assert.Equal(AdtMcalDecodeProfile.LichKingStrict, textureFile.DecodeProfile);
        Assert.Empty(textureFile.TextureNames);
        Assert.Equal(256, textureFile.Chunks.Count);
        Assert.Equal(0, textureFile.Chunks.Sum(chunk => chunk.Layers.Count));
        Assert.Equal(0, textureFile.Chunks.Sum(chunk => chunk.DecodedLayerCount));
    }

    [Fact]
    public void Read_DevelopmentTexAdt_ProducesStableRealDataChunkSignals()
    {
        AdtTextureFile textureFile = AdtTextureReader.Read(MapTestPaths.DevelopmentTexAdtPath);

        Assert.Equal(MapFileKind.AdtTex, textureFile.Kind);
        Assert.Equal(AdtMcalDecodeProfile.Cataclysm400, textureFile.DecodeProfile);
        Assert.Equal(5, textureFile.TextureNames.Count);
        Assert.Equal(256, textureFile.Chunks.Count);
        Assert.Equal(775, textureFile.Chunks.Sum(chunk => chunk.Layers.Count));
        Assert.Equal(519, textureFile.Chunks.Sum(chunk => chunk.Layers.Count(layer => layer.DecodedAlpha is not null)));
        Assert.Equal(515, textureFile.Chunks.Sum(chunk => chunk.Layers.Count(layer => layer.DecodedAlpha?.Encoding == AdtMcalAlphaEncoding.Compressed)));
        Assert.Equal(4, textureFile.Chunks.Sum(chunk => chunk.Layers.Count(layer => layer.DecodedAlpha?.Encoding == AdtMcalAlphaEncoding.BigAlpha)));
        Assert.Equal(0, textureFile.Chunks.Sum(chunk => chunk.Layers.Count(layer => layer.DecodedAlpha?.Encoding == AdtMcalAlphaEncoding.BigAlphaFixed)));
        Assert.Equal(0, textureFile.Chunks.Sum(chunk => chunk.Layers.Count(layer => layer.DecodedAlpha?.Encoding == AdtMcalAlphaEncoding.Packed4Bit)));
    }

    private static byte[] CreateTexChunkPayload(byte[] mclyPayload, byte[] mcalPayload)
    {
        using MemoryStream stream = new();
        stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk("MCLY", mclyPayload));
        stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk("MCAL", mcalPayload));
        return stream.ToArray();
    }

    private static byte[] CreateRootTextureMcnkPayload(uint indexX, uint indexY, uint flags, byte[] mclyPayload, byte[] mcalPayload)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x0C, 4), (uint)(mclyPayload.Length / 16));
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x28, 4), (uint)(8 + mcalPayload.Length));

        using MemoryStream stream = new();
        stream.Write(header);
        stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk("MCLY", mclyPayload));
        stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk("MCAL", mcalPayload));
        return stream.ToArray();
    }

    private static byte[] CreateMclyPayload(uint[] layerFlags, uint[] layerOffsets)
    {
        byte[] payload = new byte[layerFlags.Length * 16];
        for (int index = 0; index < layerFlags.Length; index++)
        {
            int offset = index * 16;
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset, 4), (uint)index);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 4, 4), layerFlags[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 8, 4), layerOffsets[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 12, 4), 0u);
        }

        return payload;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }
}