using System.IO;
using System.Linq;
using System.Text;
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

        // ---------------- BYTE-PARITY TESTS ----------------
        private static readonly string[] NavChunkSigs =
        {
            "MSHD","MSPV","MSPI","MSVI","MSVT","MSUR","MSCN","MSLK","MPRL"
        };

        public static IEnumerable<object[]> SignatureData => NavChunkSigs.Select(s => new object[] { s });

        [Theory]
        [MemberData(nameof(SignatureData))]
        public void Chunk_Roundtrip_ShouldMatchOriginal(string sig)
        {
            string path = GetSamplePm4Path();
            byte[] fileBytes = File.ReadAllBytes(path);
            var pm4 = PM4File.FromFile(path);

            byte[] original = SliceChunk(fileBytes, sig);
            byte[] reserialized = SerializeChunk(pm4, sig);

            Assert.Equal(original, reserialized);
        }

        private static byte[] SliceChunk(byte[] fileBytes, string sig)
        {
            byte[] sigBytes = Encoding.ASCII.GetBytes(sig);
            byte[] revBytes = sigBytes.Reverse().ToArray();
            for (int i = 0; i + 8 <= fileBytes.Length; i++)
            {
                bool matchForward = fileBytes[i]==sigBytes[0] && fileBytes[i+1]==sigBytes[1] &&
                                     fileBytes[i+2]==sigBytes[2] && fileBytes[i+3]==sigBytes[3];
                bool matchReverse = fileBytes[i]==revBytes[0] && fileBytes[i+1]==revBytes[1] &&
                                     fileBytes[i+2]==revBytes[2] && fileBytes[i+3]==revBytes[3];

                if (matchForward || matchReverse)
                {
                    uint size = BitConverter.ToUInt32(fileBytes, i+4);
                    byte[] chunkData = new byte[size];
                    Buffer.BlockCopy(fileBytes, i+8, chunkData, 0, (int)size);
                    return chunkData;
                }
            }
            throw new InvalidOperationException($"Signature {sig} not found in sample PM4");
        }

        private static byte[] SerializeChunk(PM4File pm4, string sig)
        {
            return sig switch
            {
                "MSHD" => pm4.MSHD!.Serialize(),
                "MSPV" => pm4.MSPV!.Serialize(),
                "MSPI" => pm4.MSPI!.Serialize(),
                "MSVI" => pm4.MSVI!.Serialize(),
                "MSVT" => pm4.MSVT!.Serialize(),
                "MSUR" => pm4.MSUR!.Serialize(),
                "MSCN" => pm4.MSCN!.Serialize(),
                "MSLK" => pm4.MSLK!.Serialize(),
                "MPRL" => pm4.MPRL!.Serialize(),
                _ => throw new ArgumentOutOfRangeException(nameof(sig), sig, null)
            };
        }
    }
}
