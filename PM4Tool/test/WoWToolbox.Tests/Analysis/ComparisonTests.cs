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
using WoWToolbox.Tests;

namespace WoWToolbox.Tests.Analysis
{
    public class ComparisonTests
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(Assembly.GetExecutingAssembly().Location, "../../../../../../", "test_data"));
        private static string TestOutputRoot = OutputLocator.Central("ComparisonTests");

        private const string TestPm4FileRelativePath = "original_development/development_00_00.pm4";
        private const string TestAdtObj0FileRelativePath = "original_development/development_0_0_obj0.adt";
        private const string WmoAssetBasePath = "335_wmo"; // Base folder within test_data for WMO assets

        public ComparisonTests()
        {
            Directory.CreateDirectory(TestOutputRoot);
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

        private static MeshData WmoGroupMeshToMeshData(WmoGroupMesh mesh)
        {
            // Convert WmoGroupMesh to MeshData for compatibility with MeshComparisonUtils
            var md = new MeshData();
            if (mesh == null) return md;
            foreach (var v in mesh.Vertices)
                md.Vertices.Add(v.Position);
            foreach (var tri in mesh.Triangles)
            {
                md.Indices.Add(tri.Index0);
                md.Indices.Add(tri.Index1);
                md.Indices.Add(tri.Index2);
            }
            return md;
        }
    }
} 