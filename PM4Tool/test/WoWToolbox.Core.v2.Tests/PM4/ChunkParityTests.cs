using System.IO;
using System.Linq;
using Xunit;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class ChunkParityTests
    {
        // Helper retrieves first test PM4 path deterministically so CI never breaks
        private static string GetSamplePm4Path()
        {
            // prefer smallest file in development_00_00 area for speed
            var baseDir = Path.Combine("test_data", "original_development");
            var candidate = Directory.EnumerateFiles(baseDir, "development_00_00.pm4", SearchOption.AllDirectories).FirstOrDefault();
            Assert.False(candidate is null, $"Sample PM4 not found under {baseDir}.");
            return candidate!;
        }

        [Fact]
        public void MSVIChunk_ShouldLoadIndices()
        {
            var pm4 = PM4File.FromFile(GetSamplePm4Path());
            Assert.NotNull(pm4.MSVI);
            Assert.NotEmpty(pm4.MSVI!.Indices);

            // Basic sanity: indices should reference some vertex indices within MSVT
            var vertCount = pm4.MSVT?.Vertices.Count ?? 0;
            Assert.True(vertCount > 0, "MSVT vertices missing in sample PM4");
            var invalid = pm4.MSVI.Indices.Any(i => i >= vertCount);
            Assert.False(invalid, "MSVI contains index out of bounds of MSVT vertex list");
        }

        [Fact]
        public void MPRLChunk_ShouldLoadEntries()
        {
            var pm4 = PM4File.FromFile(GetSamplePm4Path());
            Assert.NotNull(pm4.MPRL);
            Assert.NotEmpty(pm4.MPRL!.Entries);

            // Basic sanity: entry struct size & values
            foreach (var entry in pm4.MPRL.Entries)
            {
                // Position values should be finite numbers
                Assert.False(float.IsNaN(entry.Position.X) || float.IsInfinity(entry.Position.X));
                Assert.False(float.IsNaN(entry.Position.Y) || float.IsInfinity(entry.Position.Y));
                Assert.False(float.IsNaN(entry.Position.Z) || float.IsInfinity(entry.Position.Z));
            }
        }
    }
}
