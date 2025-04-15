using System;
using System.IO;
using System.Linq;
using System.Numerics;
using Warcraft.NET.Files.Structures;
using WoWToolbox.Core.ADT;
using Xunit;

namespace WoWToolbox.Tests.ADT
{
    public class AdtServiceTests
    {
        private const string TestAdtPath = "test_data/original_development/development_0_0.adt";

        [Fact]
        public void ExtractPlacements_ShouldExtractDataCorrectly()
        {
            Console.WriteLine("--- AdtServiceTests.ExtractPlacements_ShouldExtractDataCorrectly START ---");

            // Arrange
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var inputFilePath = Path.Combine(baseDir, TestAdtPath);

            Console.WriteLine($"Loading ADT from: {inputFilePath}");
            Assert.True(File.Exists(inputFilePath), $"Test ADT file not found: {inputFilePath}");

            var adtFile = new ADTFile(inputFilePath);
            var adtService = new AdtService();

            // Act
            var placements = adtService.ExtractPlacements(adtFile).ToList();

            // Assert
            Assert.NotNull(placements);
            Console.WriteLine($"Extracted {placements.Count} total placements.");
            Assert.True(placements.Count > 0, "Expected at least one placement to be extracted.");

            // --- Assertions for specific known placements --- 
            // We need specific data from development_0_0.adt to make these assertions meaningful.
            // For now, let's find the first ModelPlacement and WmoPlacement and check basic properties.

            var firstModel = placements.OfType<ModelPlacement>().FirstOrDefault();
            Assert.NotNull(firstModel); // Assuming there's at least one M2
            if (firstModel != null)
            {
                Console.WriteLine($"First Model: ID={firstModel.UniqueId}, NameID={firstModel.NameId}, Pos={firstModel.Position}, Rot={firstModel.Rotation}, Scale={firstModel.Scale}");
                // Example Assertions (Replace with actual expected values if known):
                // Assert.Equal(12345u, firstModel.UniqueId);
                // Assert.Equal(6789u, firstModel.NameId);
                // Assert.Equal(new Vector3(10.0f, 20.0f, 30.0f), firstModel.Position);
                Assert.NotEqual(0.0f, firstModel.Scale); // Scale should generally not be zero
            }

            var firstWmo = placements.OfType<WmoPlacement>().FirstOrDefault();
            Assert.NotNull(firstWmo); // Assuming there's at least one WMO
            if (firstWmo != null)
            {
                Console.WriteLine($"First WMO: ID={firstWmo.UniqueId}, NameID={firstWmo.NameId}, Pos={firstWmo.Position}, Rot={firstWmo.Rotation}, Scale={firstWmo.Scale}, BBox={firstWmo.BoundingBox}");
                // Example Assertions (Replace with actual expected values if known):
                // Assert.Equal(54321u, firstWmo.UniqueId);
                // Assert.Equal(9876u, firstWmo.NameId);
                // Assert.Equal(new Vector3(-10.0f, -20.0f, -30.0f), firstWmo.Position);
                Assert.NotEqual(0.0f, firstWmo.Scale); // Scale should generally not be zero
                Assert.NotEqual(firstWmo.BoundingBox.Minimum, firstWmo.BoundingBox.Maximum); // BBox min and max shouldn't be identical
            }

            Console.WriteLine("--- AdtServiceTests.ExtractPlacements_ShouldExtractDataCorrectly END ---");
        }
    }
} 