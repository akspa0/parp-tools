using System;
using System.IO;
using Xunit;

namespace WoWToolbox.Tests.WMO
{
    public class WmoV14ToV17ConverterTests
    {
        private static readonly string LogPath = WoWToolbox.Tests.OutputLocator.Central("wmo_v17_test", "WmoV14ToV17ConverterTests.log");

        private void Log(string message)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            string line = $"[{timestamp}] {message}";
            Console.WriteLine(line);
            Directory.CreateDirectory(Path.GetDirectoryName(LogPath)!);
            File.AppendAllText(LogPath, line + Environment.NewLine);
        }

        [Fact]
        public void CanConvertIronforge053V14ToV17()
        {
            Log("=== TEST START: CanConvertIronforge053V14ToV17 ===");
            // Arrange: Try both possible test data locations
            string[] possiblePaths = {
                Path.Combine("test_data", "053_wmo", "Ironforge_053.wmo"),
                Path.Combine("test_data", "wmo_v14", "Ironforge_053.wmo")
            };
            string inputPath = Array.Find(possiblePaths, File.Exists);
            if (inputPath == null)
            {
                Log("[SKIP] Test input not found in any known location.");
                Log("=== TEST END: CanConvertIronforge053V14ToV17 ===");
                return; // Skip test if input is missing
            }
            string outputDir = WoWToolbox.Tests.OutputLocator.Central("wmo_v17_test");
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);
            Log($"Input: {inputPath}");
            Log($"Output: {outputDir}");

            // Act
            try
            {
                WoWToolbox.WmoV14Converter.WmoV14ToV17Converter.Convert(inputPath, outputDir);
                Log("[INFO] Converter ran without exception.");
            }
            catch (Exception ex)
            {
                Log($"[ERROR] Exception during conversion: {ex.Message}\n{ex.StackTrace}");
                Log("=== TEST END: CanConvertIronforge053V14ToV17 ===");
                throw;
            }

            // Assert
            string rootOutPath = Path.Combine(outputDir, "Ironforge_053.wmo");
            bool rootExists = File.Exists(rootOutPath);
            long rootLen = rootExists ? new FileInfo(rootOutPath).Length : 0;
            Log($"Root file exists: {rootExists}, size: {rootLen}");
            Assert.True(rootExists, $"Root file not created: {rootOutPath}");
            Assert.True(rootLen > 0, $"Root file is empty: {rootOutPath}");

            // Check for at least one group file
            bool foundGroup = false;
            for (int i = 0; i < 10; i++) // Check first 10 groups
            {
                string groupOutPath = Path.Combine(outputDir, $"Ironforge_053_{i:D3}.wmo");
                if (File.Exists(groupOutPath) && new FileInfo(groupOutPath).Length > 0)
                {
                    foundGroup = true;
                    Log($"Found non-empty group file: {groupOutPath}");
                    break;
                }
            }
            Assert.True(foundGroup, "No non-empty group files were created.");
            Log("[PASS] WMO v14 to v17 conversion test completed successfully.");
            Log("=== TEST END: CanConvertIronforge053V14ToV17 ===");
        }

        [Fact]
        public void CanExportFirstGroupAsObj()
        {
            Log("=== TEST START: CanExportFirstGroupAsObj ===");
            // Arrange: Try both possible test data locations
            string[] possiblePaths = {
                Path.Combine("test_data", "053_wmo", "Ironforge_053.wmo"),
                Path.Combine("test_data", "wmo_v14", "Ironforge_053.wmo")
            };
            string inputPath = Array.Find(possiblePaths, File.Exists);
            if (inputPath == null)
            {
                Log("[SKIP] Test input not found in any known location.");
                Log("=== TEST END: CanExportFirstGroupAsObj ===");
                return; // Skip test if input is missing
            }
            string outputDir = WoWToolbox.Tests.OutputLocator.Central("wmo_v17_test");
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);
            string objOutPath = Path.Combine(outputDir, "Ironforge_053_firstgroup.obj");
            Log($"Input: {inputPath}");
            Log($"OBJ Output: {objOutPath}");

            // Act
            try
            {
                WoWToolbox.WmoV14Converter.WmoV14ToV17Converter.ExportFirstGroupAsObj(inputPath, objOutPath);
                Log("[INFO] OBJ export ran without exception.");
            }
            catch (Exception ex)
            {
                Log($"[ERROR] Exception during OBJ export: {ex.Message}\n{ex.StackTrace}");
                Log("=== TEST END: CanExportFirstGroupAsObj ===");
                throw;
            }

            // Assert
            bool objExists = File.Exists(objOutPath);
            long objLen = objExists ? new FileInfo(objOutPath).Length : 0;
            Log($"OBJ file exists: {objExists}, size: {objLen}");
            Assert.True(objExists, $"OBJ file not created: {objOutPath}");
            Assert.True(objLen > 0, $"OBJ file is empty: {objOutPath}");
            Log("[PASS] OBJ export test completed successfully.");
            Log("=== TEST END: CanExportFirstGroupAsObj ===");
        }
    }
} 