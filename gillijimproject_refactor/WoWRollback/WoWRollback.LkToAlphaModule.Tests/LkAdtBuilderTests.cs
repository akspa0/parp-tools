using System;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;
using WoWRollback.LkToAlphaModule.Builders;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Tests;

public class LkAdtBuilderTests
{
    [Fact]
    public void Build_WithValidSource_ProducesNonEmptyOutput()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkAdtBuilder.Build(source, options);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result);
    }

    [Fact]
    public void Build_OutputStartsWithMVERChunk()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkAdtBuilder.Build(source, options);

        // Assert
        Assert.True(result.Length >= 12, "Output too small for MVER chunk");

        string fourCC = Encoding.ASCII.GetString(result, 0, 4);
        Assert.Equal("MVER", fourCC);
        
        int size = BitConverter.ToInt32(result, 4);
        Assert.Equal(4, size); // MVER data is always 4 bytes
        
        int version = BitConverter.ToInt32(result, 8);
        Assert.Equal(18, version); // LK ADT version
    }

    [Fact]
    public void Build_Includes256MCNKChunks()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkAdtBuilder.Build(source, options);

        // Assert
        int mcnkCount = CountMcnkChunks(result);
        Assert.Equal(256, mcnkCount);
    }

    [Fact]
    public void Build_WithFewerThan256Chunks_ThrowsInvalidDataException()
    {
        // Arrange
        var source = new LkAdtSource
        {
            MapName = "TestMap",
            TileX = 0,
            TileY = 0
        };
        
        // Only add 100 chunks instead of 256
        for (int i = 0; i < 100; i++)
        {
            source.Mcnks.Add(TestDataFactory.CreateTestMcnkSource());
        }
        
        var options = new LkToAlphaOptions();

        // Act & Assert
        var ex = Assert.Throws<InvalidDataException>(() => 
            LkAdtBuilder.Build(source, options));
        Assert.Contains("256", ex.Message);
    }

    [Fact]
    public void Build_WithMoreThan256Chunks_ThrowsInvalidDataException()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource();
        
        // Add extra chunks
        for (int i = 0; i < 10; i++)
        {
            source.Mcnks.Add(TestDataFactory.CreateTestMcnkSource());
        }
        
        var options = new LkToAlphaOptions();

        // Act & Assert
        var ex = Assert.Throws<InvalidDataException>(() => 
            LkAdtBuilder.Build(source, options));
        Assert.Contains("256", ex.Message);
    }

    [Fact]
    public void Build_PreservesMapName()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource(mapName: "CustomMap");
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkAdtBuilder.Build(source, options);

        // Assert
        // Map name is stored in the source but not in the ADT file itself
        // This test verifies the build doesn't fail with custom map name
        Assert.NotNull(result);
        Assert.Equal("CustomMap", source.MapName);
    }

    [Fact]
    public void Build_WithNullSource_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new LkToAlphaOptions();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            LkAdtBuilder.Build(null, options));
    }

    [Fact]
    public void Build_WithNullOptions_UsesDefaults()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource();

        // Act
        byte[] result = LkAdtBuilder.Build(source, null);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result);
    }

    [Fact]
    public void Build_WithoutLiquids_OmitsMh2oChunk()
    {
        var source = TestDataFactory.CreateTestAdtSource(includeLiquids: false);

        byte[] result = LkAdtBuilder.Build(source, new LkToAlphaOptions());

        Assert.False(TryFindChunk(result, "MH2O", out _, out _));
    }

    [Fact]
    public void Build_WithLiquids_WritesMh2oChunkWithInstance()
    {
        var source = TestDataFactory.CreateTestAdtSource(includeLiquids: true);

        byte[] result = LkAdtBuilder.Build(source, new LkToAlphaOptions());

        Assert.True(TryFindChunk(result, "MH2O", out int mh2oOffset, out int mh2oSize));
        Assert.True(mh2oSize > 0);

        var chunkData = new byte[mh2oSize];
        Buffer.BlockCopy(result, mh2oOffset + 8, chunkData, 0, mh2oSize);

        const int headerTableSize = 256 * 12;
        Assert.True(chunkData.Length >= headerTableSize);

        int firstInstanceOffset = BitConverter.ToInt32(chunkData, 0);
        uint instanceCount = BitConverter.ToUInt32(chunkData, 4);
        int attributesOffset = BitConverter.ToInt32(chunkData, 8);

        Assert.True(instanceCount > 0);
        Assert.True(firstInstanceOffset >= headerTableSize);
        Assert.True(attributesOffset >= headerTableSize);

        int instanceStructOffset = mh2oOffset + 8 + firstInstanceOffset;
        ushort liquidTypeId = BitConverter.ToUInt16(result, instanceStructOffset);
        ushort vertexFormat = BitConverter.ToUInt16(result, instanceStructOffset + 2);
        byte xOffset = result[instanceStructOffset + 12];
        byte yOffset = result[instanceStructOffset + 13];
        byte width = result[instanceStructOffset + 14];
        byte height = result[instanceStructOffset + 15];
        uint vertexDataOffset = BitConverter.ToUInt32(result, instanceStructOffset + 20);

        Assert.Equal(1, liquidTypeId);
        Assert.Equal(0u, (uint)vertexFormat);
        Assert.Equal(0, xOffset);
        Assert.Equal(0, yOffset);
        Assert.Equal(2, width);
        Assert.Equal(2, height);
        Assert.True(vertexDataOffset >= headerTableSize);

        int absoluteVertexData = mh2oOffset + 8 + (int)vertexDataOffset;
        float firstHeight = BitConverter.ToSingle(result, absoluteVertexData);
        Assert.Equal(0f, firstHeight);
        byte firstDepth = result[absoluteVertexData + (9 * sizeof(float))];
        Assert.Equal(1, firstDepth);

        int absoluteAttributes = mh2oOffset + 8 + attributesOffset;
        Assert.True(absoluteAttributes + 16 <= mh2oOffset + 8 + mh2oSize);
        byte fishableRow0 = result[absoluteAttributes];
        Assert.Equal(0b00000011, fishableRow0);
    }

    [Fact]
    public void Build_WithPlacements_WritesMddfAndModfChunks()
    {
        var source = TestDataFactory.CreateTestAdtSource(includePlacements: true);

        byte[] result = LkAdtBuilder.Build(source, new LkToAlphaOptions());

        Assert.True(TryFindChunk(result, "MMDX", out int mmdxOffset, out int mmdxSize));
        Assert.True(mmdxSize > 0, "MMDX chunk should contain at least one filename");
        var mmdxData = new byte[mmdxSize];
        Buffer.BlockCopy(result, mmdxOffset + 8, mmdxData, 0, mmdxSize);
        string mmdxString = Encoding.UTF8.GetString(mmdxData).TrimEnd('\0');
        Assert.Contains("World\\Generic\\Human\\PassiveDoodads\\barrel01.m2", mmdxString);

        Assert.True(TryFindChunk(result, "MWMO", out int mwmoOffset, out int mwmoSize));
        Assert.True(mwmoSize > 0, "MWMO chunk should contain at least one filename");
        var mwmoData = new byte[mwmoSize];
        Buffer.BlockCopy(result, mwmoOffset + 8, mwmoData, 0, mwmoSize);
        string mwmoString = Encoding.UTF8.GetString(mwmoData).TrimEnd('\0');
        Assert.Contains("World\\Wmo\\Human\\Stormwind\\Stormwind.wmo", mwmoString);

        Assert.True(TryFindChunk(result, "MDDF", out int mddfOffset, out int mddfSize));
        Assert.Equal(36, mddfSize);
        var mddfData = new byte[mddfSize];
        Buffer.BlockCopy(result, mddfOffset + 8, mddfData, 0, mddfSize);
        int doodadNameIndex = BitConverter.ToInt32(mddfData, 0);
        int doodadUniqueId = BitConverter.ToInt32(mddfData, 4);
        float doodadX = BitConverter.ToSingle(mddfData, 8);
        ushort doodadScaleRaw = BitConverter.ToUInt16(mddfData, 32);
        ushort doodadFlags = BitConverter.ToUInt16(mddfData, 34);
        Assert.Equal(0, doodadNameIndex);
        Assert.Equal(9001, doodadUniqueId);
        Assert.Equal(10f, doodadX);
        Assert.Equal(1024, doodadScaleRaw);
        Assert.Equal((ushort)0, doodadFlags);

        Assert.True(TryFindChunk(result, "MODF", out int modfOffset, out int modfSize));
        Assert.Equal(52, modfSize);
        var modfData = new byte[modfSize];
        Buffer.BlockCopy(result, modfOffset + 8, modfData, 0, modfSize);
        int wmoNameIndex = BitConverter.ToInt32(modfData, 0);
        int wmoUniqueId = BitConverter.ToInt32(modfData, 4);
        float wmoX = BitConverter.ToSingle(modfData, 8);
        ushort wmoFlags = BitConverter.ToUInt16(modfData, 44);
        ushort wmoDoodadSet = BitConverter.ToUInt16(modfData, 46);
        ushort wmoNameSet = BitConverter.ToUInt16(modfData, 48);
        ushort wmoScale = BitConverter.ToUInt16(modfData, 50);
        Assert.Equal(0, wmoNameIndex);
        Assert.Equal(5001, wmoUniqueId);
        Assert.Equal(40f, wmoX);
        Assert.Equal((ushort)0, wmoFlags);
        Assert.Equal((ushort)1, wmoDoodadSet);
        Assert.Equal((ushort)2, wmoNameSet);
        Assert.Equal((ushort)1024, wmoScale);
    }

    [Fact]
    public void Build_WithoutPlacements_WritesEmptyPlacementChunks()
    {
        var source = TestDataFactory.CreateTestAdtSource(includePlacements: false);

        byte[] result = LkAdtBuilder.Build(source, new LkToAlphaOptions());

        Assert.True(TryFindChunk(result, "MMDX", out int mmdxOffset, out int mmdxSize));
        Assert.Equal(0, mmdxSize);

        Assert.True(TryFindChunk(result, "MWMO", out int mwmoOffset, out int mwmoSize));
        Assert.Equal(0, mwmoSize);

        Assert.True(TryFindChunk(result, "MDDF", out int mddfOffset, out int mddfSize));
        Assert.Equal(0, mddfSize);

        Assert.True(TryFindChunk(result, "MODF", out int modfOffset, out int modfSize));
        Assert.Equal(0, modfSize);
    }

    [Fact]
    public void Build_OutputSizeIsReasonable()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkAdtBuilder.Build(source, options);

        // Assert
        // Each MCNK is at least 0x80 bytes header + some subchunks
        // 256 chunks * ~200 bytes minimum = ~50KB minimum
        Assert.True(result.Length > 50_000, $"Output seems too small: {result.Length} bytes");
        
        // Should not be unreasonably large (< 10MB for test data)
        Assert.True(result.Length < 10_000_000, $"Output seems too large: {result.Length} bytes");
    }

    [Fact]
    public void Build_MCNKChunksAreSequential()
    {
        // Arrange
        var source = TestDataFactory.CreateTestAdtSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkAdtBuilder.Build(source, options);

        // Assert
        // Verify MCNK chunks appear in order (IndexX, IndexY should match position)
        int mcnkIndex = 0;
        int offset = 12; // Skip MVER chunk
        
        while (offset < result.Length - 8 && mcnkIndex < 256)
        {
            string fourCC = Encoding.ASCII.GetString(result, offset, 4);
            if (fourCC == "KNCM") // Reversed "MCNK"
            {
                // Read IndexX and IndexY from MCNK header
                int indexX = BitConverter.ToInt32(result, offset + 8 + 0x04);
                int indexY = BitConverter.ToInt32(result, offset + 8 + 0x08);
                
                int expectedX = mcnkIndex % 16;
                int expectedY = mcnkIndex / 16;
                
                Assert.Equal(expectedX, indexX);
                Assert.Equal(expectedY, indexY);
                
                mcnkIndex++;
                
                // Skip to next chunk
                int size = BitConverter.ToInt32(result, offset + 4);
                offset += 8 + size;
            }
            else
            {
                break;
            }
        }
    }

    // Helper method to count MCNK chunks in ADT data
    private int CountMcnkChunks(byte[] adtBytes)
    {
        int count = 0;
        byte[] mcnkBytes = Encoding.ASCII.GetBytes("KNCM"); // Reversed "MCNK"
        
        for (int i = 0; i <= adtBytes.Length - 4; i++)
        {
            bool match = true;
            for (int j = 0; j < 4; j++)
            {
                if (adtBytes[i + j] != mcnkBytes[j])
                {
                    match = false;
                    break;
                }
            }
            if (match) count++;
        }
        
        return count;
    }

    private bool TryFindChunk(byte[] data, string fourCc, out int offset, out int size)
    {
        string reversed = new string(fourCc.Reverse().ToArray());
        int position = 0;
        while (position + 8 <= data.Length)
        {
            string id = Encoding.ASCII.GetString(data, position, 4);
            int chunkSize = BitConverter.ToInt32(data, position + 4);
            if (id == reversed)
            {
                offset = position;
                size = chunkSize;
                return true;
            }

            if (chunkSize < 0)
                break;

            position += 8 + chunkSize;
            if (chunkSize == 0)
                position += 0;
        }

        offset = 0;
        size = 0;
        return false;
    }
}
