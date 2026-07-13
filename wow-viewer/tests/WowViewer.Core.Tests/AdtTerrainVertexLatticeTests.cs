using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtTerrainVertexLatticeTests
{
    [Fact]
    public void Build_PreservesRawMcvtTopologyZeroHeightsAndMcnrAxisOrder()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"wowviewer_mcvt_{Guid.NewGuid():N}.adt");
        try
        {
            using (FileStream stream = File.Create(tempPath))
            {
                byte[] mver = CreateChunk("MVER", CreateUInt32Payload(18));
                stream.Write(mver);
                for (int chunkY = 0; chunkY < TerrainVertexLattice.ChunksPerAxis; chunkY++)
                {
                    for (int chunkX = 0; chunkX < TerrainVertexLattice.ChunksPerAxis; chunkX++)
                    {
                        byte[] mcnk = CreateChunk("MCNK", CreateRootMcnkPayload(chunkX, chunkY));
                        stream.Write(mcnk);
                    }
                }
            }

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(tempPath, buildVersion: "3.3.5.12340");
            TerrainVertexLattice lattice = Assert.IsType<TerrainVertexLattice>(pack.TerrainVertices);
            TerrainWdlLattice wdl = Assert.IsType<TerrainWdlLattice>(pack.WdlLattice);

            Assert.Equal(16 * 16 * 145, lattice.Present.Cast<bool>().Count(static value => value));
            Assert.Equal(33_025, lattice.DenseValidMask.Cast<bool>().Count(static value => value));
            Assert.True(lattice.DenseValidMask[0, 0]);
            Assert.True(lattice.DenseValidMask[1, 1]);
            Assert.False(lattice.DenseValidMask[0, 1]);
            Assert.False(lattice.DenseValidMask[1, 0]);
            Assert.Equal(256, TerrainVertexLattice.ChunkTriangleIndices.GetLength(0));
            Assert.Equal(3, TerrainVertexLattice.ChunkTriangleIndices.GetLength(1));

            // ADT tile X maps to world Y and tile Y maps to world X.
            Assert.Equal(32f * 533.33333f, lattice.WorldX[0, 0, 0], 3);
            Assert.Equal(32f * 533.33333f, lattice.WorldY[0, 0, 0], 3);
            Assert.True(lattice.WorldX[0, 0, 144] < lattice.WorldX[0, 0, 0]);
            Assert.True(lattice.WorldY[0, 0, 144] < lattice.WorldY[0, 0, 0]);

            // Zero is a legitimate absolute height, not a missing-data sentinel.
            Assert.Equal(0f, lattice.VertexZ[0, 0, 0]);
            Assert.Equal(0f, pack.Height257![0, 0]);

            Assert.Equal(TerrainWdlLattice.SampleCount,
                wdl.OuterPresent.Cast<bool>().Count(static value => value)
                + wdl.InnerPresent.Cast<bool>().Count(static value => value));
            Assert.Equal(pack.Height257[0, 0], wdl.Outer17[0, 0]);
            Assert.Equal(pack.Height257[256, 256], wdl.Outer17[16, 16]);
            Assert.Equal(pack.Height257[8, 8], wdl.Inner16[0, 0]);
            Assert.Equal(pack.Height257[248, 248], wdl.Inner16[15, 15]);

            WdlHeightTile writerTile = WdlWriter.ExtractTileHeightsFromAlpha(pack.Height257, 0, 0);
            Assert.Equal(writerTile.OuterHeights,
                wdl.Outer17.Cast<float>().Select(static value => (short)Math.Clamp(Math.Round(value), short.MinValue, short.MaxValue)));
            Assert.Equal(writerTile.InnerHeights,
                wdl.Inner16.Cast<float>().Select(static value => (short)Math.Clamp(Math.Round(value), short.MinValue, short.MaxValue)));

            // Disk MCNR is X,Z,Y. The public tensor is normalized XYZ.
            Assert.Equal(1f, pack.McnrNormalXyz![0, 0, 0], 5);
            Assert.Equal(0f, pack.McnrNormalXyz[0, 0, 1], 5);
            Assert.Equal(0f, pack.McnrNormalXyz[0, 0, 2], 5);

            TerrainVertexLattice.ResolveDenseCoordinates(0, 0, 1, out int zX, out int zY);
            Assert.Equal(0f, pack.McnrNormalXyz[zY, zX, 0], 5);
            Assert.Equal(0f, pack.McnrNormalXyz[zY, zX, 1], 5);
            Assert.Equal(1f, pack.McnrNormalXyz[zY, zX, 2], 5);

            TerrainVertexLattice.ResolveDenseCoordinates(0, 0, 2, out int yX, out int yY);
            Assert.Equal(0f, pack.McnrNormalXyz[yY, yX, 0], 5);
            Assert.Equal(1f, pack.McnrNormalXyz[yY, yX, 1], 5);
            Assert.Equal(0f, pack.McnrNormalXyz[yY, yX, 2], 5);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    [Fact]
    public void ResolveDenseCoordinates_CoversEveryNativeMcvtSampleExactly()
    {
        HashSet<(int x, int y)> positions = [];
        for (int sample = 0; sample < TerrainVertexLattice.SamplesPerChunk; sample++)
        {
            TerrainVertexLattice.ResolveDenseCoordinates(0, 0, sample, out int x, out int y);
            Assert.True(positions.Add((x, y)), $"duplicate native sample coordinate ({x},{y})");
            Assert.Equal(x & 1, y & 1);
        }

        Assert.Equal(TerrainVertexLattice.SamplesPerChunk, positions.Count);
    }

    [Fact]
    public void ResolveSampleIndex_RoundTripsEveryNativeMcvtSample()
    {
        for (int sample = 0; sample < TerrainVertexLattice.SamplesPerChunk; sample++)
        {
            TerrainVertexLattice.ResolveLocalHalfStepCoordinates(sample, out int x, out int y);
            Assert.Equal(sample, TerrainVertexLattice.ResolveSampleIndex(x, y));
        }

        Assert.Throws<ArgumentException>(() => TerrainVertexLattice.ResolveSampleIndex(1, 0));
    }

    private static byte[] CreateRootMcnkPayload(int chunkX, int chunkY)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x04, 4), chunkX);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x08, 4), chunkY);

        float[] heights = new float[TerrainVertexLattice.SamplesPerChunk];
        for (int sample = 0; sample < heights.Length; sample++)
            heights[sample] = (chunkY * 10_000) + (chunkX * 200) + sample;
        heights[0] = chunkX == 0 && chunkY == 0 ? 0f : heights[0];

        byte[] mcvtPayload = new byte[heights.Length * sizeof(float)];
        for (int sample = 0; sample < heights.Length; sample++)
            BinaryPrimitives.WriteSingleLittleEndian(mcvtPayload.AsSpan(sample * sizeof(float), sizeof(float)), heights[sample]);

        byte[] mcnrPayload = new byte[0x1C0];
        for (int sample = 0; sample < TerrainVertexLattice.SamplesPerChunk; sample++)
            mcnrPayload[(sample * 3) + 1] = 127; // disk Z -> public Z
        if (chunkX == 0 && chunkY == 0)
        {
            mcnrPayload[0] = 127; mcnrPayload[1] = 0; mcnrPayload[2] = 0;
            mcnrPayload[3] = 0; mcnrPayload[4] = 127; mcnrPayload[5] = 0;
            mcnrPayload[6] = 0; mcnrPayload[7] = 0; mcnrPayload[8] = 127;
        }

        using MemoryStream stream = new();
        stream.Write(header);
        stream.Write(CreateChunk("MCVT", mcvtPayload));
        stream.Write(CreateChunk("MCNR", mcnrPayload));
        return stream.ToArray();
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(FourCC.FromString(id).ToFileBytes(), bytes, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }
}
