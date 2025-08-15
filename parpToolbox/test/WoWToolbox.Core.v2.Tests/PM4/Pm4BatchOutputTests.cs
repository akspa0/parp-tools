using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Services.PM4;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class Pm4BatchOutputTests
    {
        private static readonly string TestDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "test_data"));
        private const string Pm4FilesPath = "pm4_files";

        [Fact]
        public void BatchProcessor_Writes_Summary_To_ProjectOutput()
        {
            var pm4File = Path.Combine(TestDataPath, Pm4FilesPath, "development_00_00.pm4");
            if (!File.Exists(pm4File))
            {
                // real dataset not in repo/CI; skip
                return;
            }

            var coordinateService = new CoordinateService();
            var buildingExtractionService = new PM4BuildingExtractionService(coordinateService);
            var matcher = new WmoMatcher(Path.Combine(TestDataPath, "wmo_data"));
            var processor = new Pm4BatchProcessor(buildingExtractionService, matcher);

            var result = processor.Process(pm4File);
            Assert.True(result.Success);

            string fileKey = Path.GetFileNameWithoutExtension(pm4File);
            string summaryPath = Path.Combine(processor.RunDirectory, fileKey, "summary.txt");
            Assert.True(File.Exists(summaryPath));
        }
    }
}
