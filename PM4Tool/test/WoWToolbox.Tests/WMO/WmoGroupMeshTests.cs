using Xunit;
using WoWToolbox.Core.WMO;
using WoWToolbox.Core; // For WmoRootLoader
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace WoWToolbox.Tests.WMO
{
    public class WmoGroupMeshTests
    {
        // Define paths relative to test execution
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(Assembly.GetExecutingAssembly().Location, "../../../../../../", "test_data")); // Navigating up 6 levels from bin/Debug/netX.X
        private static string TestOutputDir => Path.GetFullPath(Path.Combine(Assembly.GetExecutingAssembly().Location, "../../../../../../", "output", "WmoGroupMeshTests")); // Output to project_root/output/WmoGroupMeshTests

        // Example WMO and group index
        // private const string TestWmoRootFile = "wmo/Dungeon/Ulduar/Ulduar_Entrance.wmo"; // Example, replace with a known good test file
        // private const string TestWmoRootFile = "335_wmo/World/wmo/Dungeon/Ulduar/Ulduar_Raid.wmo"; // Using provided existing file
        private const string TestWmoRootFile = "335_wmo\\World\\wmo\\Northrend\\Buildings\\IronDwarf\\ND_IronDwarf_LargeBuilding\\ND_IronDwarf_LargeBuilding.wmo"; // Corrected path with extra dir level
        // private const int TestGroupIndex = 0;

        public WmoGroupMeshTests()
        {
            // Ensure output directory exists
            Console.WriteLine($"[DEBUG] Assembly Location: {Assembly.GetExecutingAssembly().Location}");
            Console.WriteLine($"[DEBUG] Calculated TestOutputDir: {TestOutputDir}");
            Directory.CreateDirectory(TestOutputDir);
        }

        [Fact]
        public void LoadAndExportAllWmoGroups_ShouldCreateMergedObjFile()
        {
            // Arrange
            string rootWmoPath = Path.Combine(TestDataRoot, TestWmoRootFile);
            string outputMergedObjPath = Path.Combine(TestOutputDir, Path.GetFileNameWithoutExtension(rootWmoPath) + "_merged.obj");

            Console.WriteLine($"Attempting to load root WMO: {rootWmoPath}");
            Console.WriteLine($"Attempting to save merged OBJ to: {outputMergedObjPath}");

            Assert.True(File.Exists(rootWmoPath), $"Root WMO test file not found: {rootWmoPath}");

            int groupCount = -1;
            List<string> groupNames = new List<string>();
            try
            {
                (groupCount, groupNames) = WmoRootLoader.LoadGroupInfo(rootWmoPath);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Failed to load root WMO file: {ex.Message}");
            }
            Assert.True(groupCount >= 0, $"Failed to read group count from root WMO (LoadGroupInfo returned {groupCount}).");
            Assert.True(groupCount > 0, "Root WMO reported 0 groups.");

            List<WmoGroupMesh> loadedMeshes = new List<WmoGroupMesh>();
            string rootWmoDirectory = Path.GetDirectoryName(rootWmoPath) ?? ".";

            // Act - Load all groups
            Console.WriteLine($"Root WMO has {groupCount} groups (found {groupNames.Count} names). Attempting to load...");
            for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
            {
                string groupFileName = Path.GetFileNameWithoutExtension(rootWmoPath) + $"_{groupIndex:000}.wmo";
                string groupFilePath = Path.Combine(rootWmoDirectory, groupFileName);

                Console.WriteLine($"  Attempting to load group {groupIndex}: {groupFilePath}");
                if (!File.Exists(groupFilePath))
                {
                    Console.WriteLine($"  [WARN] Group file not found, skipping: {groupFilePath}");
                    continue;
                }

                try
                {
                    using var stream = File.OpenRead(groupFilePath);
                    WmoGroupMesh? loadedMesh = WmoGroupMesh.LoadFromStream(stream, groupFilePath);
                    if (loadedMesh != null)
                    {
                        Console.WriteLine($"    -> Loaded group {groupIndex} successfully.");
                        loadedMeshes.Add(loadedMesh);
                    }
                    else
                    {
                        Console.WriteLine($"    -> Loading group {groupIndex} returned null.");
                    }
                }
                catch (Exception ex)
                {
                    // Log warning but continue trying other groups
                    Console.WriteLine($"  [WARN] Failed to load WMO group file {groupIndex}: {ex.Message}");
                }
            } // End group loop

            Assert.True(loadedMeshes.Count > 0, "Failed to load any WMO groups.");
            Console.WriteLine($"Successfully loaded {loadedMeshes.Count} WMO groups.");

            // Save the loaded mesh
            Console.WriteLine("Merging loaded meshes...");
            WmoGroupMesh mergedMesh = WmoGroupMesh.MergeMeshes(loadedMeshes);
            Assert.NotNull(mergedMesh); // Ensure merge succeeded

            Console.WriteLine("Saving merged mesh...");
            WmoGroupMesh.SaveToObj(mergedMesh, outputMergedObjPath);

            // Assert
            Assert.True(File.Exists(outputMergedObjPath), $"Expected merged WMO OBJ file was not created at: {outputMergedObjPath}");
            Assert.True(new FileInfo(outputMergedObjPath).Length > 100, $"Expected merged WMO OBJ file appears empty: {outputMergedObjPath}"); // Basic size check

            Console.WriteLine($"Successfully merged {loadedMeshes.Count} WMO groups and saved to {outputMergedObjPath}");
            Console.WriteLine($"  Total Vertices: {mergedMesh.Vertices.Count}");
            Console.WriteLine($"  Total Triangles: {mergedMesh.Triangles.Count}");
        }
    }
} 