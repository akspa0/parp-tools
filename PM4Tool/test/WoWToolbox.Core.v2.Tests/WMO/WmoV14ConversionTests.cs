using System;
using System.IO;
using System.Text;
using WoWToolbox.Core.v2.Services.WMO;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.WMO
{
    public class WmoV14ConversionTests
    {
        private static readonly string TestDataDir = LocateTestData();
        private static string LocateTestData()
        {
            var dir = AppContext.BaseDirectory;
            for (int i = 0; i < 10; i++)
            {
                var candidate = Path.Combine(dir, "test_data", "053_wmo");
                if (Directory.Exists(candidate)) return candidate;
                dir = Path.GetDirectoryName(dir)!;
            }
            return ""; // will cause skip below
        }
        private static string GetSamplePath(string name) => string.IsNullOrEmpty(TestDataDir)? "": Path.Combine(TestDataDir, name);

        [Theory]
        [InlineData("Ironforge_053.wmo")]
        [InlineData("Orgrimmar_053.wmo")]
        public void Converter_Upgrades_Header_To_V17(string filename)
        {
            var path = GetSamplePath(filename);
            if (string.IsNullOrEmpty(TestDataDir) || !File.Exists(path))
                return; // skip if dataset not present

            byte[] v14Data;
            try { v14Data = File.ReadAllBytes(path); }
            catch (IOException) { return; }
            // Verify input is v14
            uint originalVersion = BitConverter.ToUInt32(v14Data.AsSpan(8, 4));
            Assert.Equal(14u, originalVersion);

            var converter = new WmoV14Converter();
            var v17Data = converter.ConvertToV17(v14Data);

            // Verify output is v17
            uint newVersion = BitConverter.ToUInt32(v17Data.AsSpan(8, 4));
            Assert.Equal(17u, newVersion);
            // basic length sanity (should be >= original due to identical copy)
            Assert.True(v17Data.Length >= v14Data.Length - 8);
        }

        [Fact]
        public void Converter_Exports_First_Group_Obj()
        {
            var path = GetSamplePath("Ironforge_053.wmo");
            if (string.IsNullOrEmpty(TestDataDir) || !File.Exists(path))
                return; // skip if dataset not present

            var converter = new WmoV14Converter();
            string objPath = converter.ExportFirstGroupAsObj(path);
            Assert.True(File.Exists(objPath), $"OBJ not created at {objPath}");
            long len = new FileInfo(objPath).Length;
            Assert.True(len > 0, "OBJ file is empty");
            Console.WriteLine($"[OBJ] v14 first group exported to {objPath}, size {len} bytes");
        }
    }
}
