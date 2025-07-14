using System;
using System.IO;
using System.Linq;
using System.Numerics;
using Warcraft.NET.Files.Structures;
using WoWToolbox.Core.ADT;
using WoWToolbox.Core.Navigation.PM4;
using Xunit;

namespace WoWToolbox.Tests.ADT
{
    public class AdtServiceTests
    {
        private const string TestAdtPath = "test_data/original_development/development_0_0.adt";

        private static System.Collections.Generic.Dictionary<uint, string> LoadListfile(string filePath)
        {
            var data = new System.Collections.Generic.Dictionary<uint, string>();
            foreach (var line in File.ReadLines(filePath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(';');
                if (parts.Length >= 2 && uint.TryParse(parts[0], out uint fileDataId) && !string.IsNullOrWhiteSpace(parts[1]))
                {
                    data.TryAdd(fileDataId, parts[1].Trim());
                }
            }
            return data;
        }

        [Fact/*(Skip = "Skipping due to unsuitable/empty test data (development_15_28_obj0.adt has no placements).")*/]
        public void ExtractPlacements_ShouldExtractDataCorrectly()
        {
            Console.WriteLine("--- AdtServiceTests.ExtractPlacements_ShouldExtractDataCorrectly START ---");

            // Arrange
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var inputFilePath = Path.Combine(baseDir, "test_data/original_development/development_0_0_obj0.adt");
            var listfilePath = Path.Combine(baseDir, "src/community-listfile-withcapitals.csv");
            var listfileData = File.Exists(listfilePath) ? LoadListfile(listfilePath) : new System.Collections.Generic.Dictionary<uint, string>();

            Console.WriteLine($"Loading ADT _obj0 from: {inputFilePath}");
            Assert.True(File.Exists(inputFilePath), $"Test ADT _obj0 file not found: {inputFilePath}");

            byte[] adtObj0Bytes = File.ReadAllBytes(inputFilePath);
            var adtObj0 = new Warcraft.NET.Files.ADT.TerrainObject.Zero.TerrainObjectZero(adtObj0Bytes);
            var adtService = new AdtService();

            // Act
            var placements = adtService.ExtractPlacements(adtObj0, listfileData).ToList();

            // Assert
            Assert.NotNull(placements);
            Console.WriteLine($"Extracted {placements.Count} total placements.");
            Assert.True(placements.Count > 0, "Expected at least one placement to be extracted.");

            // --- Assertions for specific known placements --- 
            // We need specific data from development_00_00_obj0.adt to make these assertions meaningful.
            // For now, let's find the first ModelPlacement and WmoPlacement and check basic properties.

            var firstModel = placements.OfType<ModelPlacement>().FirstOrDefault();
            Assert.NotNull(firstModel); // Assuming there's at least one M2
            if (firstModel != null)
            {
                Console.WriteLine($"First Model: ID={firstModel.UniqueId}, NameID={firstModel.NameId}, Pos={firstModel.Position}, Rot={firstModel.Rotation}, Scale={firstModel.Scale}");
                Assert.NotEqual(0.0f, firstModel.Scale); // Scale should generally not be zero
            }

            var firstWmo = placements.OfType<WmoPlacement>().FirstOrDefault();
            Assert.NotNull(firstWmo); // Assuming there's at least one WMO
            if (firstWmo != null)
            {
                Console.WriteLine($"First WMO: ID={firstWmo.UniqueId}, NameID={firstWmo.NameId}, Pos={firstWmo.Position}, Rot={firstWmo.Rotation}, Scale={firstWmo.Scale}, BBox={firstWmo.BoundingBox}");
                Assert.NotEqual(0.0f, firstWmo.Scale); // Scale should generally not be zero
                Assert.NotEqual(firstWmo.BoundingBox.Minimum, firstWmo.BoundingBox.Maximum); // BBox min and max shouldn't be identical
            }

            Console.WriteLine("--- AdtServiceTests.ExtractPlacements_ShouldExtractDataCorrectly END ---");
        }

        [Fact(Skip = "Skipping: PM4 mesh extraction logic (Pm4MeshExtractor) is missing after refactor. Pending new mesh extraction API.")]
        public void CorrelatePm4MeshesWithAdtPlacements_ByUniqueId()
        {
            // Test skipped due to missing mesh extraction logic after refactor.
            // See project TODOs for new mesh extraction API integration.
        }
    }
} 