using Xunit;
using WoWToolbox.Core.WMO;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Models;
using WoWToolbox.Common.Analysis;
using WoWToolbox.MSCNExplorer; // For Pm4MeshExtractor
// Assuming Warcraft.NET ADT classes are accessible, may need specific using
using Warcraft.NET.Files.ADT.TerrainObject.Zero; 
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Globalization; // Added using

namespace WoWToolbox.Tests.Analysis
{
    public class ComparisonTests
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(Assembly.GetExecutingAssembly().Location, "../../../../../../", "test_data"));
        private static string TestOutputRoot => Path.GetFullPath(Path.Combine(Assembly.GetExecutingAssembly().Location, "../../../../../../", "output", "ComparisonTests"));

        private const string TestPm4FileRelativePath = "original_development/development_00_00.pm4";
        private const string TestAdtObj0FileRelativePath = "original_development/development_0_0_obj0.adt";
        private const string WmoAssetBasePath = "335_wmo"; // Base folder within test_data for WMO assets

        public ComparisonTests()
        {
            Directory.CreateDirectory(TestOutputRoot);
        }

        [Fact/*(Skip = "Skipping due to unsuitable/empty test data (development_15_28.pm4 lacks MDSF links).")*/]
        public void ComparePm4UniqueIdComponents_ToTransformedWmoMeshes_Dev0000()
        {
            // Arrange: Load PM4
            string pm4Path = Path.Combine(TestDataRoot, TestPm4FileRelativePath);
            Assert.True(File.Exists(pm4Path), $"PM4 file not found: {pm4Path}");
            var pm4File = PM4File.FromFile(pm4Path);
            Assert.NotNull(pm4File);

            // Arrange: Load ADT Obj0
            string adtPath = Path.Combine(TestDataRoot, TestAdtObj0FileRelativePath);
            Assert.True(File.Exists(adtPath), $"ADT Obj0 file not found: {adtPath}");
            // Use the specific class from Warcraft.NET - Load via constructor
            // var adtObj0File = TerrainObjectZero.FromFile(adtPath); // Old incorrect way
            TerrainObjectZero? adtObj0File = null;
            try 
            {
                // Try constructor with path first, fallback to byte array if needed
                // Assuming constructor exists, otherwise need Warcraft.NET source check
                // adtObj0File = new TerrainObjectZero(adtPath); // Incorrect - expects byte[]?
                adtObj0File = new TerrainObjectZero(File.ReadAllBytes(adtPath)); // Load with byte array
            }
            catch(Exception ex)
            {
                 Assert.Fail($"Failed to instantiate TerrainObjectZero from path '{adtPath}': {ex.Message}");
            }
            // var adtObj0File = new TerrainObjectZero(File.ReadAllBytes(adtPath)); // Fallback if path constructor fails
            Assert.NotNull(adtObj0File);
            // Correct property names based on Warcraft.NET inspection
            Assert.NotNull(adtObj0File.WorldModelObjectPlacementInfo); // MODF chunk property
            Assert.NotNull(adtObj0File.WorldModelObjects); // MWMO chunk property

            // Arrange: Extract PM4 components by UniqueID
            var extractor = new Pm4MeshExtractor();
            var pm4Components = extractor.ExtractMeshesByUniqueId(pm4File);
            Assert.NotEmpty(pm4Components);

            int comparedCount = 0;
            int matchCount = 0;
            int potentialMatchCount = 0;
            int mismatchCount = 0;
            int errorCount = 0;

            // Act: Iterate through PM4 components and compare with corresponding WMO
            foreach (var kvp in pm4Components)
            {
                uint uniqueId = kvp.Key;
                MeshData pm4ComponentMeshData = kvp.Value;

                if (!pm4ComponentMeshData.IsValid())
                {
                     Console.WriteLine($"[CompareTest] Skipping UniqueID {uniqueId:X8}: Invalid PM4 component MeshData.");
                     errorCount++;
                     continue;
                }

                // Find ADT Placement (MODF entry)
                // Access correct property and cast uniqueId for comparison
                var placement = adtObj0File.WorldModelObjectPlacementInfo.MODFEntries.FirstOrDefault(p => p.UniqueId == (int)uniqueId);
                if (placement == null)
                {
                    Console.WriteLine($"[CompareTest] Skipping UniqueID {uniqueId:X8}: No matching MODF placement found in ADT.");
                    errorCount++; // Or Mismatch?
                    continue;
                }

                // Get WMO Filename from MWMO using NameId
                // Access correct property
                if (adtObj0File.WorldModelObjects == null || placement.NameId >= adtObj0File.WorldModelObjects.Filenames.Count)
                {
                    Console.WriteLine($"[CompareTest] Skipping UniqueID {uniqueId:X8}: Invalid NameId {placement.NameId} for MWMO chunk.");
                     errorCount++;
                     continue;
                }
                // Access correct property
                string wmoRelativePath = adtObj0File.WorldModelObjects.Filenames[(int)placement.NameId].Replace("\\", Path.DirectorySeparatorChar.ToString());
                string wmoFullPath = Path.Combine(TestDataRoot, WmoAssetBasePath, wmoRelativePath);

                 if (!File.Exists(wmoFullPath))
                {
                    Console.WriteLine($"[CompareTest] Skipping UniqueID {uniqueId:X8}: WMO file not found at '{wmoFullPath}'.");
                    errorCount++; // This asset might be missing from test data
                    continue;
                }

                // Load and Merge WMO Mesh Data
                MeshData? wmoLocalMeshData = null;
                try
                {
                    (int groupCount, _) = WmoRootLoader.LoadGroupInfo(wmoFullPath);
                    if (groupCount > 0)
                    {
                        var loadedGroups = new List<MeshData>();
                        string wmoDir = Path.GetDirectoryName(wmoFullPath) ?? ".";
                        string wmoBaseName = Path.GetFileNameWithoutExtension(wmoFullPath);
                        for (int i = 0; i < groupCount; i++)
                        {
                            string groupPath = Path.Combine(wmoDir, $"{wmoBaseName}_{i:000}.wmo");
                            if (File.Exists(groupPath))
                            {
                                using var stream = File.OpenRead(groupPath);
                                var groupMesh = WmoGroupMesh.LoadFromStream(stream, groupPath);
                                if (groupMesh != null && groupMesh.IsValid()) loadedGroups.Add(groupMesh);
                            }
                        }
                         wmoLocalMeshData = WmoGroupMesh.MergeMeshes(loadedGroups);
                    }
                    else
                    {
                         // wmoLocalMeshData = MeshData.Empty; // WMO with no groups - Error: MeshData has no Empty
                         wmoLocalMeshData = new MeshData(); // Use new empty mesh
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[CompareTest] Skipping UniqueID {uniqueId:X8}: Error loading WMO '{wmoRelativePath}': {ex.Message}");
                     errorCount++;
                    continue;
                }
                
                if (wmoLocalMeshData == null || !wmoLocalMeshData.IsValid())
                {
                     Console.WriteLine($"[CompareTest] Skipping UniqueID {uniqueId:X8}: Invalid or empty WMO MeshData loaded for '{wmoRelativePath}'.");
                     errorCount++;
                     continue;
                }

                // Transform WMO Mesh Data
                // ADT MODF position/rotation/scale needs to be correctly interpreted.
                // Assuming MODF has Position, Rotation (Quaternion/Euler?), Scale fields.
                // Placeholder: Use placement.Position, placement.Rotation, placement.Scale
                // Rotation: ADT uses Pitch (X), Yaw (Y), Roll (Z) in degrees.
                // WoW rotation order is ZYX Euler. We need to convert this to a Quaternion.
                // Create Quaternions for each axis rotation (degrees to radians).
                float pitchRad = placement.Rotation.Pitch * (MathF.PI / 180.0f);
                float yawRad = placement.Rotation.Yaw * (MathF.PI / 180.0f);
                float rollRad = placement.Rotation.Roll * (MathF.PI / 180.0f);

                Quaternion qX = Quaternion.CreateFromAxisAngle(Vector3.UnitX, pitchRad);
                Quaternion qY = Quaternion.CreateFromAxisAngle(Vector3.UnitY, yawRad);
                Quaternion qZ = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, rollRad);

                // Combine in ZYX order (multiply in reverse: X * Y * Z)
                var rotation = qX * qY * qZ; // Correct multiplication order for applying Z, then Y, then X rotations

                // Convert ADT placement (Y-up) to PM4 coordinate system (Z-up)
                // Position: Swap Y and Z, negate new Z -- CORRECTING THIS NEGATION
                // Standard Y-up (X,Z,Y) to Z-up (X,Y,Z): WorldX=AdtX, WorldY=AdtZ, WorldZ=AdtY
                var position = new Vector3(placement.Position.X, placement.Position.Z, placement.Position.Y); // Removed negation from Y

                // Scale: MODF uses a single ushort (divide by 1024 for float)
                var scale = placement.Scale / 1024.0f; // MODF uses a single scale factor

                MeshData wmoWorldMeshData = MeshComparisonUtils.TransformMeshData(wmoLocalMeshData, position, rotation, scale);
                
                if (!wmoWorldMeshData.IsValid())
                {
                     Console.WriteLine($"[CompareTest] Skipping UniqueID {uniqueId:X8}: Invalid WMO MeshData after transformation for '{wmoRelativePath}'.");
                     errorCount++;
                     continue;
                }

                // Compare
                Console.WriteLine($"[CompareTest] Comparing UniqueID {uniqueId:X8} (PM4) vs WMO '{wmoRelativePath}'");
                ComparisonResult result = MeshComparisonUtils.CompareMeshesBasic(pm4ComponentMeshData, wmoWorldMeshData);
                Console.WriteLine($"    -> Result: {result}");

                // Save OBJs for inspection (optional)
                string pm4ObjPath = Path.Combine(TestOutputRoot, $"compare_{uniqueId:X8}_pm4.obj");
                string wmoObjPath = Path.Combine(TestOutputRoot, $"compare_{uniqueId:X8}_wmo_{Path.GetFileNameWithoutExtension(wmoRelativePath)}.obj");
                SaveMeshDataToObjHelper(pm4ComponentMeshData, pm4ObjPath); // Use local helper
                SaveMeshDataToObjHelper(wmoWorldMeshData, wmoObjPath); // Use local helper

                comparedCount++;
                switch (result)
                {
                    case ComparisonResult.Match: matchCount++; break;
                    case ComparisonResult.PotentialMatch: potentialMatchCount++; break;
                    case ComparisonResult.Mismatch: mismatchCount++; break;
                    default: errorCount++; break;
                }
            }

            // Assert Summary
            Console.WriteLine($"\nComparison Summary:");
            Console.WriteLine($"  Total PM4 Components by UniqueID: {pm4Components.Count}");
            Console.WriteLine($"  Compared: {comparedCount}");
            Console.WriteLine($"  Matches: {matchCount}");
            Console.WriteLine($"  Potential Matches: {potentialMatchCount}");
            Console.WriteLine($"  Mismatches: {mismatchCount}");
            Console.WriteLine($"  Errors/Skipped: {errorCount}");

            Assert.True(comparedCount > 0, "No comparisons were successfully performed.");
            // Add more specific assertions if expected outcomes are known
            // Assert.True(matchCount + potentialMatchCount > 0, "Expected at least some matches or potential matches.");
             Assert.True(mismatchCount < comparedCount, "Expected fewer mismatches than total comparisons."); // Basic sanity check
        }

         // --- Added Helper Method (copied) ---
        private static void SaveMeshDataToObjHelper(MeshData meshData, string outputPath)
        {
            // (Implementation is the same as the one added to WmoGroupMeshTests.cs)
            // ... copy implementation here ...
             if (meshData == null)
            {
                Console.WriteLine($"[WARN] MeshData is null, cannot save OBJ.");
                return;
            }
            try
            {
                 string? directoryPath = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
                {
                    Directory.CreateDirectory(directoryPath);
                }
                using (var writer = new StreamWriter(outputPath, false))
                {
                    CultureInfo culture = CultureInfo.InvariantCulture;
                    writer.WriteLine($"# Mesh saved by WoWToolbox.Tests.Analysis.ComparisonTests");
                    writer.WriteLine($"# Vertices: {meshData.Vertices.Count}");
                    writer.WriteLine($"# Triangles: {meshData.Indices.Count / 3}");
                    writer.WriteLine($"# Generated: {DateTime.Now}");
                    writer.WriteLine();
                    if (meshData.Vertices.Count > 0)
                    {
                        writer.WriteLine("# Vertex Definitions");
                        foreach (var vertex in meshData.Vertices)
                        {
                            writer.WriteLine(string.Format(culture, "v {0} {1} {2}", vertex.X, vertex.Y, vertex.Z));
                        }
                        writer.WriteLine();
                    }
                    if (meshData.Indices.Count > 0)
                    {
                        writer.WriteLine("# Face Definitions");
                        for (int i = 0; i < meshData.Indices.Count; i += 3)
                        {
                            int idx0 = meshData.Indices[i + 0] + 1;
                            int idx1 = meshData.Indices[i + 1] + 1;
                            int idx2 = meshData.Indices[i + 2] + 1;
                            writer.WriteLine($"f {idx0} {idx1} {idx2}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR] Failed to save MeshData to OBJ file '{outputPath}': {ex.Message}");
                throw;
            }
        }
    }
} 