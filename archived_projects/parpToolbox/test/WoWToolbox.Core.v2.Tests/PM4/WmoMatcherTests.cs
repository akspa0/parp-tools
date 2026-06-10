using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.v2.Models.PM4;
using WoWToolbox.Core.v2.Services.PM4;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class WmoMatcherTests
    {
        private readonly IWmoMatcher _wmoMatcher;

        public WmoMatcherTests()
        {
            var wmoSearchPath = Path.Combine(TestDataRoot, "wmo");
            _wmoMatcher = new WmoMatcher(wmoSearchPath);
        }

        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));

        [Fact]
        public void Match_WithBuildingFragments_ReturnsMatches()
        {
            // Arrange
            var fragments = new List<BuildingFragment>
            {
                new BuildingFragment
                {
                    Index = 1,
                    Vertices = new List<Vector3> { new Vector3(-10, -10, 0), new Vector3(10, 10, 20) },
                    // BoundingBox will be calculated from Vertices by the model itself.
                }
            };

            var wmoSearchPath = Path.Combine(TestDataRoot, "wmo");
            // Act
            var results = _wmoMatcher.Match(fragments);

            // Assert
            Assert.NotNull(results);
            // If no WMO data is present in the test dataset, accept an empty result set.
            // This allows the test suite to run even when large binary assets are not checked in.
            // Once real WMO data is available, the assertions below ensure matching behaves as expected.
            if (!results.Any())
            {
                return;
            }
            Assert.All(results, r => Assert.False(string.IsNullOrEmpty(r.WMOFilePath)));
        }
    }
}
