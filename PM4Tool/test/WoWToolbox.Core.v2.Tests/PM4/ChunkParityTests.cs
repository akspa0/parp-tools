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
            // First look relative to test output directory (fast path when files copied)
            var fastDir = Path.Combine(AppContext.BaseDirectory, "test_data", "original_development");
            if (Directory.Exists(fastDir))
            {
                var fastCand = Directory.EnumerateFiles(fastDir, "development_00_00.pm4", SearchOption.AllDirectories).FirstOrDefault();
                if (fastCand is not null) return fastCand;
            }

            // Fallback: walk up from output dir until repo root containing test_data/original_development is found
            string? dir = AppContext.BaseDirectory;
            string? candidate = null;
            while (dir is not null)
            {
                var baseDir = Path.Combine(dir, "test_data", "original_development");
                if (Directory.Exists(baseDir))
                {
                    candidate = Directory.EnumerateFiles(baseDir, "development_00_00.pm4", SearchOption.AllDirectories).FirstOrDefault();
                    break;
                }
                dir = Path.GetDirectoryName(dir);
            }
            Assert.False(candidate is null, "Sample PM4 not found in test_data/original_development tree.");
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
