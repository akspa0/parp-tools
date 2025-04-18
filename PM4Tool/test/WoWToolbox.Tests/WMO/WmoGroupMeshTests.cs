using Xunit;
using WoWToolbox.Core.WMO;
using WoWToolbox.Core; // For WmoRootLoader
using WoWToolbox.Core.Models; // Added for MeshData
using System.Collections.Generic;
using System.Globalization;
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

            List<MeshData> loadedMeshes = new List<MeshData>();
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
                    MeshData? loadedMesh = WmoGroupMesh.LoadFromStream(stream, groupFilePath);
                    if (loadedMesh != null && loadedMesh.Vertices.Count > 0 && loadedMesh.Indices.Count > 0)
                    {
                        Console.WriteLine($"    -> Loaded group {groupIndex} successfully ({loadedMesh.Vertices.Count} verts, {loadedMesh.Indices.Count / 3} tris).");
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
            MeshData mergedMesh = WmoGroupMesh.MergeMeshes(loadedMeshes);
            Assert.NotNull(mergedMesh); // Ensure merge succeeded
            Assert.True(mergedMesh.Vertices.Count > 0, "Merged mesh should have vertices.");
            Assert.True(mergedMesh.Indices.Count > 0, "Merged mesh should have indices.");

            Console.WriteLine("Saving merged mesh...");
            SaveMeshDataToObj(mergedMesh, outputMergedObjPath);

            // Assert
            Assert.True(File.Exists(outputMergedObjPath), $"Expected merged WMO OBJ file was not created at: {outputMergedObjPath}");
            Assert.True(new FileInfo(outputMergedObjPath).Length > 100, $"Expected merged WMO OBJ file appears empty: {outputMergedObjPath}"); // Basic size check

            Console.WriteLine($"Successfully merged {loadedMeshes.Count} WMO groups and saved to {outputMergedObjPath}");
            Console.WriteLine($"  Total Vertices: {mergedMesh.Vertices.Count}");
            Console.WriteLine($"  Total Triangles: {mergedMesh.Indices.Count / 3}");
        }

        // --- Added Helper Method (copied from Pm4MeshExtractorTests) ---
        private static void SaveMeshDataToObj(MeshData meshData, string outputPath)
        {
            if (meshData == null)
            {
                Console.WriteLine("[WARN] MeshData is null, cannot save OBJ.");
                return;
            }

            try
            {
                 string? directoryPath = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
                {
                    Console.WriteLine($"[DEBUG][SaveMeshDataToObj] Creating directory: {directoryPath}");
                    Directory.CreateDirectory(directoryPath);
                }

                using (var writer = new StreamWriter(outputPath, false)) // Overwrite if exists
                {
                    // Set culture for consistent decimal formatting
                    CultureInfo culture = CultureInfo.InvariantCulture;

                    // Write header
                    writer.WriteLine($"# Mesh saved by WoWToolbox.Tests.WMO.WmoGroupMeshTests");
                    writer.WriteLine($"# Vertices: {meshData.Vertices.Count}");
                    writer.WriteLine($"# Triangles: {meshData.Indices.Count / 3}");
                    writer.WriteLine($"# Generated: {DateTime.Now}");
                    writer.WriteLine();

                    // Write Vertices (v x y z)
                    if (meshData.Vertices.Count > 0)
                    {
                        writer.WriteLine("# Vertex Definitions");
                        foreach (var vertex in meshData.Vertices)
                        {
                            // Format using invariant culture to ensure '.' as decimal separator
                            writer.WriteLine(string.Format(culture, "v {0} {1} {2}", vertex.X, vertex.Y, vertex.Z));
                        }
                        writer.WriteLine(); // Blank line after vertices
                    }

                    // Write Faces (f v1 v2 v3) - OBJ uses 1-based indexing!
                    if (meshData.Indices.Count > 0)
                    {
                        writer.WriteLine("# Face Definitions");
                        for (int i = 0; i < meshData.Indices.Count; i += 3)
                        {
                            // Add 1 to each index for 1-based OBJ format
                            int idx0 = meshData.Indices[i + 0] + 1;
                            int idx1 = meshData.Indices[i + 1] + 1;
                            int idx2 = meshData.Indices[i + 2] + 1;
                            writer.WriteLine($"f {idx0} {idx1} {idx2}");
                        }
                    }
                } // StreamWriter automatically flushes and closes here
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR] Failed to save MeshData to OBJ file '{outputPath}': {ex.Message}");
                // Optionally rethrow or handle differently
                throw;
            }
        }
    }
} 