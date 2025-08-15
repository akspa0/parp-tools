using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4.Chunks;
using WoWToolbox.Core.v2.Utilities;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    /// <summary>
    /// Verifies that the MSLKEntry struct layout and critical decoded fields match the authoritative
    /// documentation in <c>PM4Documentation/pm4_format_reference.md</c> using real PM4 samples.
    /// </summary>
    public class MslkEntryLayoutTests
    {
        private static readonly string SampleRoot = ResolveSampleRoot();

        [Fact(DisplayName = "MSLKEntry layout matches reference and LinkId decodes correctly")]
        public void MslkEntry_Layout_And_LinkId_Are_Valid()
        {
            Assert.True(Directory.Exists(SampleRoot), $"Test data folder not found: {SampleRoot}");

            // Grab a small deterministic subset of tiles to keep test time reasonable
            var pm4Files = Directory.EnumerateFiles(SampleRoot, "*.pm4", SearchOption.TopDirectoryOnly)
                                    .OrderBy(p => p)
                                    .Take(10)
                                    .ToList();
            Assert.NotEmpty(pm4Files);

            foreach (var file in pm4Files)
            {
                var pm4 = PM4File.FromFile(file);
                var entries = pm4.MSLK?.Entries;
                if (entries == null || entries.Count == 0) continue;

                foreach (var e in entries)
                {
                    // 1. Struct size must be 20 bytes
                    Assert.Equal(20, MSLKEntry.StructSize);

                    // 2. Flags field constant 0x8000 as per reference doc
                    Assert.Equal(0x8000, e.Unknown_0x12);

                    // 3. LinkId decoding test when sentinel present
                    if ((e.LinkIdRaw >> 16) == 0xFFFF)
                    {
                        Assert.True(LinkIdDecoder.TryDecode(e.LinkIdRaw, out var x, out var y),
                            $"Failed to decode LinkId 0x{e.LinkIdRaw:X8} in {Path.GetFileName(file)}");
                        // X and Y should be within expected 0..63 terrain tile bounds for dev builds
                        Assert.InRange(x, 0, 63);
                        Assert.InRange(y, 0, 63);
                    }
                }
            }
                }

        private static string ResolveSampleRoot()
        {
            string? dir = AppContext.BaseDirectory;
            // Step 1: find the PM4Tool repo root
            while (dir != null && !dir.EndsWith("PM4Tool", StringComparison.OrdinalIgnoreCase))
            {
                dir = Path.GetDirectoryName(dir);
            }

            if (dir == null)
                throw new DirectoryNotFoundException("Unable to locate PM4Tool repo root while resolving test_data path.");

            string candidate = Path.Combine(dir, "test_data", "original_development");
            if (!Directory.Exists(candidate))
                throw new DirectoryNotFoundException($"Expected folder not found: {candidate}");
            return candidate;
        }
    }
}
