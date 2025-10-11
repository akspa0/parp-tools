using WoWToolbox.Core.v2.Foundation.Data;

namespace WoWToolbox.Core.v2.Tests;

/// <summary>
/// Initial validation tests to confirm Core.v2 setup is working correctly.
/// These tests validate the foundation before algorithms are backported.
/// </summary>
public class PM4LoadingValidationTests
{
    private const string TestDataPath = "../../../../../test_data/development335/development_54_22.pm4";

    [Fact]
    public void PM4File_FromFile_ShouldLoadSuccessfully()
    {
        // Arrange & Act
        var pm4File = PM4File.FromFile(TestDataPath);

        // Assert
        Assert.NotNull(pm4File);
        Assert.NotNull(pm4File.MSLK);
        Assert.NotNull(pm4File.MSVT);
        Assert.NotNull(pm4File.MSUR);
        Assert.True(pm4File.MSLK.Entries.Count > 0, "MSLK should have entries");
        Assert.True(pm4File.MSVT.Vertices.Count > 0, "MSVT should have vertices");
        Assert.True(pm4File.MSUR.Entries.Count > 0, "MSUR should have entries");
    }

    [Fact]
    public void PM4File_GetChunkAvailability_ShouldReturnValidInfo()
    {
        // Arrange
        var pm4File = PM4File.FromFile(TestDataPath);

        // Act
        var availability = pm4File.GetChunkAvailability();

        // Assert
        Assert.True(availability.HasMSLK);
        Assert.True(availability.HasMSVT);
        Assert.True(availability.HasMSUR);
    }

    [Fact]
    public void PM4File_GetAllTriangles_ShouldReturnValidTriangles()
    {
        // Arrange
        var pm4File = PM4File.FromFile(TestDataPath);

        // Act
        var triangles = pm4File.GetAllTriangles();

        // Assert
        Assert.NotEmpty(triangles);
        
        // Validate triangle indices are within vertex bounds
        var vertexCount = pm4File.MSVT!.Vertices.Count;
        foreach (var (a, b, c) in triangles)
        {
            Assert.True(a >= 0 && a < vertexCount, $"Triangle vertex {a} out of bounds");
            Assert.True(b >= 0 && b < vertexCount, $"Triangle vertex {b} out of bounds");
            Assert.True(c >= 0 && c < vertexCount, $"Triangle vertex {c} out of bounds");
        }
    }

    [Fact]
    public void PM4File_ExtractBuildings_ShouldReturnBuildingModels()
    {
        // Arrange
        var pm4File = PM4File.FromFile(TestDataPath);

        // Act
        var buildings = pm4File.ExtractBuildings();

        // Assert
        Assert.NotEmpty(buildings);
        
        // Validate each building has valid data
        foreach (var building in buildings)
        {
            Assert.NotNull(building.FileName);
            Assert.True(building.Vertices.Count > 0, $"Building {building.FileName} should have vertices");
        }
    }
}
