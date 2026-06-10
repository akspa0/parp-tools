using System;
using System.IO;
using System.Linq;
using Xunit;
using WoWToolbox.Core.v2.Services.WMO;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class WmoV14ConverterTests
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));

        [Fact]
        public void ConvertToV17_Writes_Output_File()
        {
            var sampleFolder = Path.Combine(TestDataRoot, "053_wmo");
            var sampleFile = Directory.EnumerateFiles(sampleFolder, "*.wmo").OrderBy(f => new FileInfo(f).Length).First();
            var outWmo = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".wmo");
            try
            {
                var converter = new WmoV14Converter();
                converter.ConvertToV17(sampleFile, outWmo);
                Assert.True(File.Exists(outWmo), "Converted WMO was not written");
            }
            finally
            {
                if (File.Exists(outWmo)) File.Delete(outWmo);
            }
        }
    }
}
