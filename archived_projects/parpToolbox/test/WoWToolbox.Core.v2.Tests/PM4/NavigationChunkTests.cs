using System.IO;
using System.Linq;
using Xunit;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class NavigationChunkTests
    {
        private static string GetSamplePm4Path()
        {
            // Walk up from test output directory until we find test_data/original_development
            string? dir = AppContext.BaseDirectory;
            string? candidate = null;
            string? foundBaseDir = null;
            while (dir is not null)
            {
                var baseDir = Path.Combine(dir, "test_data", "original_development");
                if (Directory.Exists(baseDir))
                {
                    candidate = Directory.EnumerateFiles(baseDir, "development_00_00.pm4", SearchOption.AllDirectories).FirstOrDefault();
                    foundBaseDir = baseDir;
                    break;
                }
                dir = Path.GetDirectoryName(dir);
            }
            Assert.False(candidate is null, $"Sample PM4 not found under {foundBaseDir ?? "(none)"}.");
            return candidate!;
        }

        private static PM4File LoadSample() => PM4File.FromFile(GetSamplePm4Path());

        [Fact]
        public void Mshd_ShouldExistAndBe32Bytes()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSHD);
            Assert.Equal(32u, pm4.MSHD!.GetSize());
        }

        [Fact]
        public void Mspv_ShouldContainVertices()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSPV);
            Assert.NotEmpty(pm4.MSPV!.Vertices);
        }

        [Fact]
        public void Mspi_ShouldValidateAgainstVertices()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSPI);
            Assert.NotNull(pm4.MSPV);
            var ok = pm4.MSPI!.ValidateIndices(pm4.MSPV!.Vertices.Count);
            Assert.True(ok, "MSPI indices out of MSPV vertex range");
        }

        [Fact]
        public void Mscn_ShouldContainExteriorVerts()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSCN);
            Assert.NotEmpty(pm4.MSCN!.ExteriorVertices);
        }

        [Fact]
        public void Mslk_ShouldMapToMspi()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSLK);
            Assert.NotNull(pm4.MSPI);
            // Verify at least one pointer maps into MSPI range
            var mspiCount = pm4.MSPI!.Indices.Count;
            var hasValid = pm4.MSLK!.Entries.Any(e => e.MspiFirstIndex >= 0 && e.MspiFirstIndex < mspiCount);
            Assert.True(hasValid, "No MSLK entry points into MSPI range");
        }

        [Fact]
        public void Msvt_ShouldContainVertices()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSVT);
            Assert.NotEmpty(pm4.MSVT!.Vertices);
        }

        [Fact]
        public void Msvi_ShouldContainIndices()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSVI);
            Assert.NotEmpty(pm4.MSVI!.Indices);
        }

        [Fact]
        public void Msur_ShouldContainEntries()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MSUR);
            Assert.NotEmpty(pm4.MSUR!.Entries);
        }

        [Fact]
        public void Mprl_ShouldContainEntries()
        {
            var pm4 = LoadSample();
            Assert.NotNull(pm4.MPRL);
            Assert.NotEmpty(pm4.MPRL!.Entries);
        }
    }
}
