using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Services.PM4;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class Pm4ModelBuilderTests
    {
        private readonly IPm4ModelBuilder _modelBuilder;
        private readonly ICoordinateService _coordinateService;

        public Pm4ModelBuilderTests()
        {
            _coordinateService = new CoordinateService();
            _modelBuilder = new Pm4ModelBuilder(_coordinateService);
        }

        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));

        [Fact]
        public void Build_WithValidPm4File_ReturnsCompleteModel()
        {
            // Arrange
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            Assert.True(File.Exists(testFile), $"Test file not found: {testFile}");
            var pm4File = PM4File.FromFile(testFile);

            // Act
            var result = _modelBuilder.Build(pm4File);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Vertices.Any(), "The resulting model should have vertices.");
        }
    }
}
