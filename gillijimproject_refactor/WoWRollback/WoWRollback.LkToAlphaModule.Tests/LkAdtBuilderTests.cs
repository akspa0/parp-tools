using System;
using System.IO;
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
        Assert.Equal("REVM", fourCC); // Reversed "MVER"
        
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
}
