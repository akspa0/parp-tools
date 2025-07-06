using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.WMO.V14;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.WMO
{
    public class V14StructureSmokeTests
    {
        private static readonly string DataDir = LocateDataDir();

        private static string LocateDataDir()
        {
            var dir = AppContext.BaseDirectory;
            for (int i = 0; i < 8; i++)
            {
                var candidate = Path.Combine(dir, "test_data", "053_wmo");
                if (Directory.Exists(candidate)) return candidate;
                dir = Path.GetDirectoryName(dir) ?? dir;
            }
            return string.Empty;
        }

        [Fact]
        public void Can_Load_All_Sample_WMOs()
        {
            if (string.IsNullOrEmpty(DataDir)) return; // skip
            var files = Directory.GetFiles(DataDir, "*.wmo", SearchOption.AllDirectories);
            if (files.Length == 0) return; // skip if none

            foreach (var file in files.Take(10)) // limit to 10 for CI time
            {
                var wmo = V14WmoFile.Load(file);
                Assert.Equal(wmo.Header.GroupCount, (uint)wmo.Groups.Count);
                if (wmo.TextureNames.Count > 0)
                    Assert.Equal(wmo.Header.TextureCount, (uint)wmo.TextureNames.Count);
            }
        }
    }
}
