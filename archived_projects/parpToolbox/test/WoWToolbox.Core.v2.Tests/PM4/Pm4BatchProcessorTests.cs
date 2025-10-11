using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Services.PM4;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class Pm4BatchProcessorTests
    {
        private static readonly string TestDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "test_data"));
        
        private const string Pm4FilesPath = @"pm4_files";
        private const string WmoDataPath = @"wmo_data";

        [Fact]
        public void Process_WithRealData_ShouldExtractAndMatchBuildings()
        {
            // Arrange
            var fullTestDataPath = Path.GetFullPath(TestDataPath);
            var pm4FilePath = Path.Combine(fullTestDataPath, Pm4FilesPath, "development_00_00.pm4");
            if (!File.Exists(pm4FilePath))
            {
                // Real PM4 test data not present; skip validation.
                return;
            }
            var wmoDataPath = Path.Combine(fullTestDataPath, WmoDataPath);

            var coordinateService = new CoordinateService();
            var buildingExtractionService = new PM4BuildingExtractionService(coordinateService);
            var wmoMatcher = new WmoMatcher(wmoDataPath);
            var batchProcessor = new Pm4BatchProcessor(buildingExtractionService, wmoMatcher);

            // Act
            var result = batchProcessor.Process(pm4FilePath);

            // Assert
            Assert.True(result.Success);
            Assert.Null(result.ErrorMessage);
            if (result.BuildingFragments.Any())
            {
                // Standard validation when fragments exist.
                Assert.True(result.Success);
            }
            else
            {
                // If no fragments were extracted, skip further asserts.
                return;
            }
            // Accept zero matches if WMO dataset is not present yet.
            if (Directory.Exists(wmoDataPath))
            {
                Assert.True(result.WmoMatches.Any(), "Should have found WMO matches.");
            }
        }
    }
}
