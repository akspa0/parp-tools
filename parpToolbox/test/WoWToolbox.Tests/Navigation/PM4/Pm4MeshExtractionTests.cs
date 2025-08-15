using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Tests;
using Xunit;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class Pm4MeshExtractionTests
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
        private static string OutputRoot => OutputLocator.Central("mesh_extraction");

        [Fact]
        public void ExtractMeshAndMscnBoundary_FromPm4File()
        {
            // Example PM4 file (update as needed)
            var pm4Path = Path.Combine(TestDataRoot, "original_development", "development_00_00.pm4");
            Assert.True(File.Exists(pm4Path), $"Test PM4 file not found: {pm4Path}");
            Directory.CreateDirectory(OutputRoot);

            var pm4File = PM4File.FromFile(pm4Path);
            Assert.NotNull(pm4File);

            // Extract mesh data
            var meshVertices = pm4File.MSVT?.Vertices?.Select(v => v.ToWorldCoordinates()).ToList() ?? new List<Vector3>();
            var meshIndices = new List<int>();
            if (pm4File.MSUR != null && pm4File.MSVI != null)
            {
                foreach (var msur in pm4File.MSUR.Entries)
                {
                    for (int i = 0; i < msur.IndexCount - 2; i++)
                    {
                        int baseIdx = (int)msur.MsviFirstIndex;
                        uint idx0 = pm4File.MSVI.Indices[baseIdx];
                        uint idx1 = pm4File.MSVI.Indices[baseIdx + i + 1];
                        uint idx2 = pm4File.MSVI.Indices[baseIdx + i + 2];
                        meshIndices.Add((int)idx0);
                        meshIndices.Add((int)idx1);
                        meshIndices.Add((int)idx2);
                    }
                }
            }

            // Extract MSCN points
            var mscnPoints = pm4File.MSCN?.ExteriorVertices ?? new List<Vector3>();

            // Write mesh OBJ
            var meshObjPath = Path.Combine(OutputRoot, "render_mesh.obj");
            using (var writer = new StreamWriter(meshObjPath))
            {
                writer.WriteLine("o RenderMesh");
                foreach (var v in meshVertices)
                    writer.WriteLine($"v {v.X} {v.Y} {v.Z}");
                for (int i = 0; i < meshIndices.Count; i += 3)
                    writer.WriteLine($"f {meshIndices[i] + 1} {meshIndices[i + 1] + 1} {meshIndices[i + 2] + 1}");
            }

            // Write MSCN OBJ
            var mscnObjPath = Path.Combine(OutputRoot, "mscn_points.obj");
            using (var writer = new StreamWriter(mscnObjPath))
            {
                writer.WriteLine("o MSCN_Boundary");
                foreach (var v in mscnPoints)
                    writer.WriteLine($"v {v.X} {v.Y} {v.Z}");
            }

            // Simple asserts
            Assert.True(meshVertices.Count > 0, "No mesh vertices extracted");
            Assert.True(meshIndices.Count > 0, "No mesh indices extracted");
            Assert.True(mscnPoints.Count > 0, "No MSCN points extracted");
        }
    }
} 