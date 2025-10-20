using System;
using System.IO;
using Xunit;
using WoWRollback.LkToAlphaModule.Services;
using WoWRollback.LkToAlphaModule.Tests.Helpers;

namespace WoWRollback.LkToAlphaModule.Tests.Services;

public class AlphaDataExtractorTests
{
    [Fact]
    public void ExtractSingleMcnk_Minimal_Works()
    {
        // Arrange
        var adtBytes = new SyntheticAlphaAdtBuilder()
            .AddMcnk(0, 0)
                .WithLayers(1)
                .WithAreaId(42)
            .And()
            .Build();

        string tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(tempFile, adtBytes);

            // Act
            var result = AlphaDataExtractor.ExtractFromAlphaAdt(tempFile);

            // Assert
            Xunit.Assert.NotNull(result);
            Xunit.Assert.Equal(256, result.Mcnks.Count); // Should have all 256 chunks
            
            var firstChunk = result.Mcnks[0];
            Xunit.Assert.Equal(0, firstChunk.IndexX);
            Xunit.Assert.Equal(0, firstChunk.IndexY);
            Xunit.Assert.Equal(42u, firstChunk.AreaId);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}
