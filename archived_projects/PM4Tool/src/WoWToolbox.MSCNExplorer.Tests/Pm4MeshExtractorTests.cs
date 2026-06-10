using Xunit;
using WoWToolbox.Core;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Models; // For MeshData
using WoWToolbox.Common.Analysis; // Added for MeshAnalysisUtils
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;

namespace WoWToolbox.MSCNExplorer.Tests
{
    public class Pm4MeshExtractorTests
    {
        // Helper to get the root directory of the test data relative to the test execution path
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
        private static string TestOutputRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestOutput", "Pm4MeshExtractor"));

        // Use a known good PM4 file from previous tests
        // private const string TestPm4FileRelativePath = "original_development/development/development_22_18.pm4";
        private const string TestPm4FileRelativePath = "original_development/development_00_00.pm4"; // Test target potentially containing IronDwarf geometry

        // Define the target WMO for filtering
        private const string TargetWmoFilename = "335_wmo\\World\\wmo\\Northrend\\Buildings\\IronDwarf\\ND_IronDwarf_LargeBuilding\\ND_IronDwarf_LargeBuilding.wmo";

        public Pm4MeshExtractorTests()
        {
            // Ensure output directory exists for each test run
            Directory.CreateDirectory(TestOutputRoot);
        }

        [Fact]
        public void ExtractMesh_ValidPm4_ShouldReturnMeshDataAndSaveObj()
        {
            // Arrange
            string testPm4FullPath = Path.Combine(TestDataRoot, TestPm4FileRelativePath);
            // Use a distinct name for the unfiltered output
            string outputUnfilteredObjPath = Path.Combine(TestOutputRoot, "extracted_pm4_mesh_unfiltered_dev0000.obj");
            // New name for the largest component output
            string outputLargestComponentObjPath = Path.Combine(TestOutputRoot, "extracted_pm4_mesh_largest_component_dev0000.obj");

            Assert.True(File.Exists(testPm4FullPath), $"Test PM4 file not found: {testPm4FullPath}");

            var pm4File = PM4File.FromFile(testPm4FullPath);
            Assert.NotNull(pm4File); // Ensure file loaded correctly

            var extractor = new Pm4MeshExtractor();

            // Act - Extract *unfiltered* mesh first
            var meshDataUnfiltered = extractor.ExtractMesh(pm4File); // Call without filter argument

            // Assert - Basic checks on unfiltered MeshData
            Assert.NotNull(meshDataUnfiltered);
            Assert.True(meshDataUnfiltered.Vertices.Count > 0, "Unfiltered MeshData should contain vertices.");
            Assert.True(meshDataUnfiltered.Indices.Count > 0, "Unfiltered MeshData should contain indices.");
            Assert.True(meshDataUnfiltered.Indices.Count % 3 == 0, "Unfiltered Index count should be divisible by 3 for triangles.");

            // Act - Save unfiltered to OBJ (Optional - can be removed if only largest component is needed)
            SaveMeshDataToObj(meshDataUnfiltered, outputUnfilteredObjPath);
            Assert.True(File.Exists(outputUnfilteredObjPath), $"Unfiltered Output OBJ file was not created: {outputUnfilteredObjPath}");
            Console.WriteLine($"Successfully extracted unfiltered mesh and saved to: {outputUnfilteredObjPath}"); // Keep log for unfiltered

            // --- New: Analyze for Largest Component ---
            Console.WriteLine("Analyzing unfiltered mesh for largest connected component...");
            var largestComponent = MeshAnalysisUtils.GetLargestComponent(meshDataUnfiltered);

            // Assert - Check largest component
            Assert.NotNull(largestComponent); // Expecting dev_00_00 to have geometry
            Assert.True(largestComponent.Vertices.Count > 0, "Largest component should have vertices.");
            Assert.True(largestComponent.Indices.Count > 0, "Largest component should have indices.");
            Assert.True(largestComponent.Indices.Count % 3 == 0, "Largest component index count should be divisible by 3.");
            // Optionally, assert that largest component is smaller than the unfiltered mesh if multiple components are expected
            // Assert.True(largestComponent.Indices.Count < meshDataUnfiltered.Indices.Count, "Largest component should be smaller than the full mesh if multiple components exist.");

            // Act - Save LARGEST COMPONENT to OBJ
            SaveMeshDataToObj(largestComponent, outputLargestComponentObjPath);

            // Assert - Check LARGEST COMPONENT OBJ file
            Assert.True(File.Exists(outputLargestComponentObjPath), $"Largest Component Output OBJ file was not created: {outputLargestComponentObjPath}");
            Assert.True(new FileInfo(outputLargestComponentObjPath).Length > 0, "Largest Component Output OBJ file should not be empty.");

            Console.WriteLine($"Successfully extracted largest component ({largestComponent.Indices.Count / 3} triangles) and saved to: {outputLargestComponentObjPath}");
        }

        [Fact]
        public void ExtractMesh_FilteredByWmo_ShouldReturnFilteredMeshDataAndSaveObj()
        {
             // Arrange
            string testPm4FullPath = Path.Combine(TestDataRoot, TestPm4FileRelativePath); // Still use dev_00_00
            string outputFilteredObjPath = Path.Combine(TestOutputRoot, "extracted_pm4_mesh_filtered_IronDwarf.obj");

            Assert.True(File.Exists(testPm4FullPath), $"Test PM4 file not found: {testPm4FullPath}");

            var pm4File = PM4File.FromFile(testPm4FullPath);
            Assert.NotNull(pm4File); // Ensure file loaded correctly
            // Explicitly check for MDBH needed for this test
            Assert.NotNull(pm4File.MDBH); 
            Assert.NotEmpty(pm4File.MDBH.Entries);
            // Also check MDOS as it's needed for the link
            Assert.NotNull(pm4File.MDOS);
            Assert.NotEmpty(pm4File.MDOS.Entries);

            var extractor = new Pm4MeshExtractor();

            // Act - Extract *filtered* mesh
            var meshDataFiltered = extractor.ExtractMesh(pm4File, TargetWmoFilename);

            // Assert - Basic checks on filtered MeshData
            Assert.NotNull(meshDataFiltered); // Should return an object even if empty
            // We expect this specific WMO to exist and have geometry in dev_00_00
            Assert.True(meshDataFiltered.Vertices.Count > 0, $"Filtered MeshData for {TargetWmoFilename} should contain vertices.");
            Assert.True(meshDataFiltered.Indices.Count > 0, $"Filtered MeshData for {TargetWmoFilename} should contain indices.");
            Assert.True(meshDataFiltered.Indices.Count % 3 == 0, "Filtered Index count should be divisible by 3 for triangles.");
            // Optionally compare counts against expected or against unfiltered? Difficult without knowing expected counts.

            // Act - Save filtered to OBJ
            SaveMeshDataToObj(meshDataFiltered, outputFilteredObjPath);

            // Assert - Check filtered OBJ file
            Assert.True(File.Exists(outputFilteredObjPath), $"Filtered Output OBJ file was not created: {outputFilteredObjPath}");
            Assert.True(new FileInfo(outputFilteredObjPath).Length > 0, "Filtered Output OBJ file should not be empty.");

            Console.WriteLine($"Successfully extracted filtered mesh for {TargetWmoFilename} and saved to: {outputFilteredObjPath}");
        }

        [Fact]
        public void ExtractMeshesByUniqueId_ShouldReturnMeshesForEachUniqueId_AndSaveObjs()
        {
            // Arrange
            string testPm4FullPath = Path.Combine(TestDataRoot, TestPm4FileRelativePath);
            Assert.True(File.Exists(testPm4FullPath), $"Test PM4 file not found: {testPm4FullPath}");

            var pm4File = PM4File.FromFile(testPm4FullPath);
            Assert.NotNull(pm4File);

            var extractor = new Pm4MeshExtractor();

            // Act
            var meshesByUniqueId = extractor.ExtractMeshesByUniqueId(pm4File);

            // Assert
            Assert.NotNull(meshesByUniqueId);
            Assert.True(meshesByUniqueId.Count > 0, "Should extract at least one mesh by uniqueID.");

            int objCount = 0;
            foreach (var kvp in meshesByUniqueId)
            {
                uint uniqueId = kvp.Key;
                var meshData = kvp.Value;
                Assert.NotNull(meshData);
                Assert.True(meshData.Vertices.Count > 0, $"Mesh for uniqueID {uniqueId} should have vertices.");
                Assert.True(meshData.Indices.Count > 0, $"Mesh for uniqueID {uniqueId} should have indices.");
                Assert.True(meshData.Indices.Count % 3 == 0, $"Mesh for uniqueID {uniqueId} should have triangle indices.");

                string objPath = Path.Combine(TestOutputRoot, $"extracted_pm4_mesh_uniqueid_{uniqueId:X8}.obj");
                SaveMeshDataToObj(meshData, objPath);
                Assert.True(File.Exists(objPath), $"OBJ file for uniqueID {uniqueId} was not created: {objPath}");
                objCount++;
            }
            Console.WriteLine($"Extracted {meshesByUniqueId.Count} meshes by uniqueID and saved {objCount} OBJ files.");
        }

        // --- Helper Method ---

        private static void SaveMeshDataToObj(MeshData meshData, string outputPath)
        {
            if (meshData == null)
            {
                Console.WriteLine("[WARN] MeshData is null, cannot save OBJ.");
                return;
            }

            try
            {
                using (var writer = new StreamWriter(outputPath, false)) // Overwrite if exists
                {
                    // Set culture for consistent decimal formatting
                    CultureInfo culture = CultureInfo.InvariantCulture;

                    // Write header
                    writer.WriteLine($"# Mesh extracted from PM4 by WoWToolbox.MSCNExplorer.Tests");
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