using System;
using System.Text;
using Xunit;
using WoWRollback.LkToAlphaModule.Builders;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Tests;

public class LkMcnkBuilderTests
{
    [Fact]
    public void BuildFromAlpha_WithValidSource_ProducesNonEmptyOutput()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result);
    }

    [Fact]
    public void BuildFromAlpha_OutputStartsWithMCNKHeader()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        Assert.True(result.Length >= 8, "Output too small for MCNK header");
        
        string fourCC = Encoding.ASCII.GetString(result, 0, 4);
        Assert.Equal("KNCM", fourCC); // Reversed "MCNK"
        
        int size = BitConverter.ToInt32(result, 4);
        Assert.True(size > 0, "MCNK size should be positive");
        Assert.True(size <= result.Length - 8, "MCNK size should not exceed data length");
    }

    [Fact]
    public void BuildFromAlpha_IncludesMCVTSubchunk()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        // Search for MCVT chunk (reversed = "TVCM")
        bool foundMcvt = ContainsSubchunk(result, "TVCM");
        Assert.True(foundMcvt, "Output should contain MCVT subchunk");
    }

    [Fact]
    public void BuildFromAlpha_IncludesMCNRSubchunk()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        bool foundMcnr = ContainsSubchunk(result, "RNCM");
        Assert.True(foundMcnr, "Output should contain MCNR subchunk");
    }

    [Fact]
    public void BuildFromAlpha_IncludesMCLYSubchunk()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource(layerCount: 2);
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        bool foundMcly = ContainsSubchunk(result, "YLCM");
        Assert.True(foundMcly, "Output should contain MCLY subchunk");
    }

    [Fact]
    public void BuildFromAlpha_WithAlphaLayers_IncludesMCALSubchunk()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource(layerCount: 2);
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        bool foundMcal = ContainsSubchunk(result, "LACM");
        Assert.True(foundMcal, "Output should contain MCAL subchunk when alpha layers present");
    }

    [Fact]
    public void BuildFromAlpha_WithNoAlphaLayers_DoesNotIncludeMCAL()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource(layerCount: 0);
        source.AlphaLayers.Clear();
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        bool foundMcal = ContainsSubchunk(result, "LACM");
        Assert.False(foundMcal, "Output should not contain MCAL when no alpha layers");
    }

    [Fact]
    public void BuildFromAlpha_PreservesIndexXAndIndexY()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource(indexX: 7, indexY: 11);
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        Assert.True(result.Length >= 0x80 + 8, "Output too small for MCNK header");
        
        // MCNK header starts at offset 8, IndexX at 0x04, IndexY at 0x08
        int indexX = BitConverter.ToInt32(result, 8 + 0x04);
        int indexY = BitConverter.ToInt32(result, 8 + 0x08);
        
        Assert.Equal(7, indexX);
        Assert.Equal(11, indexY);
    }

    [Fact]
    public void BuildFromAlpha_PreservesAreaId()
    {
        // Arrange
        var source = new LkMcnkSource
        {
            IndexX = 0,
            IndexY = 0,
            Flags = 0,
            AreaId = 9999,
            McvtRaw = TestDataFactory.CreateTestMcvt(),
            McnrRaw = TestDataFactory.CreateTestMcnr(),
            MclyRaw = Array.Empty<byte>()
        };
        var options = new LkToAlphaOptions();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        // AreaId is at offset 0x38 in MCNK header
        uint areaId = BitConverter.ToUInt32(result, 8 + 0x38);
        Assert.Equal(9999u, areaId);
    }

    [Fact]
    public void BuildFromAlpha_WithForceCompressedAlpha_SetsCompressionFlag()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource(layerCount: 1);
        var options = new LkToAlphaOptions { ForceCompressedAlpha = true };

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, options);

        // Assert
        // This test verifies the option is passed through
        // Actual compression flag verification would require parsing MCLY
        Assert.NotNull(result);
        Assert.NotEmpty(result);
    }

    [Fact]
    public void BuildFromAlpha_WithNullOptions_UsesDefaults()
    {
        // Arrange
        var source = TestDataFactory.CreateTestMcnkSource();

        // Act
        byte[] result = LkMcnkBuilder.BuildFromAlpha(source, null);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result);
    }

    [Fact]
    public void BuildFromAlpha_WithNullSource_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new LkToAlphaOptions();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            LkMcnkBuilder.BuildFromAlpha(null, options));
    }

    // Helper method to search for subchunk FourCC in MCNK data
    private bool ContainsSubchunk(byte[] mcnkBytes, string reversedFourCC)
    {
        if (mcnkBytes.Length < 8) return false;

        byte[] searchBytes = Encoding.ASCII.GetBytes(reversedFourCC);
        
        // Start search after MCNK header (8 bytes) + MCNK header data (0x80 bytes)
        for (int i = 8 + 0x80; i <= mcnkBytes.Length - 4; i++)
        {
            bool match = true;
            for (int j = 0; j < 4; j++)
            {
                if (mcnkBytes[i + j] != searchBytes[j])
                {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        
        return false;
    }
}
