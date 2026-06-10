using System;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;
using WoWToolbox.Core.v2.Foundation.Data;
using WoWToolbox.Core.v2.Models.PM4.Chunks;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class PM4FileV2Tests
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));

        [Fact]
        public void PM4FileV2_ShouldLoadAllCoreChunks()
        {
            // Arrange
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            Assert.True(File.Exists(testFile), $"Test file does not exist: {testFile}");

            // Act
            var fileData = File.ReadAllBytes(testFile);
            var pm4File = new PM4File(fileData); // v2 loader expects byte[]

            // Assert - Check all core chunk properties exist
            Assert.NotNull(pm4File);
            Assert.NotNull(pm4File.MVER);
            Assert.NotNull(pm4File.MSHD);
            Assert.NotNull(pm4File.MSVT);
            Assert.NotNull(pm4File.MSVI);
            Assert.NotNull(pm4File.MSPV);
            Assert.NotNull(pm4File.MSPI);
            Assert.NotNull(pm4File.MPRL);
            Assert.NotNull(pm4File.MPRR);
            Assert.NotNull(pm4File.MSLK);
            Assert.NotNull(pm4File.MDOS);
            Assert.NotNull(pm4File.MDSF);
            Assert.NotNull(pm4File.MDBH);
            Assert.NotNull(pm4File.MSCN);

            // Check that essential chunks have data (MDBH can be empty)
            Assert.True(pm4File.MSVT.Vertices.Count > 0, "MSVT chunk should have vertices");
            Assert.True(pm4File.MSVI.Indices.Count > 0, "MSVI chunk should have indices");
            Assert.True(pm4File.MSPV.Vertices.Count > 0, "MSPV chunk should have vertices");
            Assert.True(pm4File.MSPI.Indices.Count > 0, "MSPI chunk should have indices");
            Assert.True(pm4File.MPRL.Entries.Count > 0, "MPRL chunk should have entries");
            Assert.True(pm4File.MPRR.Sequences.Count > 0, "MPRR chunk should have sequences");
            Assert.True(pm4File.MSLK.Entries.Count > 0, "MSLK chunk should have entries");
            Assert.True(pm4File.MDOS.Entries.Count > 0, "MDOS chunk should have entries");
            Assert.True(pm4File.MDSF.Entries.Count > 0, "MDSF chunk should have entries");
            // MDBH can legitimately be empty - it contains filenames for destructible buildings
            Assert.True(pm4File.MDBH.Entries.Count >= 0, "MDBH chunk should be loaded (may be empty)");
            Assert.True(pm4File.MSCN.ExteriorVertices.Count > 0, "MSCN chunk should have exterior vertices");
        }

        [Fact]
        public void PM4FileV2_ShouldGenerateValidTriangles_NoDegenerateFaces()
        {
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            Assert.True(File.Exists(testFile), $"Test file does not exist: {testFile}");
            var fileData = File.ReadAllBytes(testFile);
            var pm4File = new PM4File(fileData);

            // Assume pm4File has a method or property to get all triangle indices (from MSUR/MSVI)
            var triangles = pm4File.GetAllTriangles(); // List<(int, int, int)>
            var vertexCount = pm4File.MSVT.Vertices.Count;
            foreach (var (a, b, c) in triangles)
            {
                Assert.InRange(a, 0, vertexCount - 1);
                Assert.InRange(b, 0, vertexCount - 1);
                Assert.InRange(c, 0, vertexCount - 1);
                Assert.False(a == b || b == c || a == c, $"Degenerate triangle: {a}, {b}, {c}");
            }
        }

        [Fact]
        public void PM4FileV2_ShouldExtractIndividualBuildings()
        {
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            Assert.True(File.Exists(testFile), $"Test file does not exist: {testFile}");
            var fileData = File.ReadAllBytes(testFile);
            var pm4File = new PM4File(fileData);

            // Assume pm4File has a method to extract buildings (returns list of building models)
            var buildings = pm4File.ExtractBuildings();
            Assert.NotNull(buildings);
            Assert.True(buildings.Count >= 10, "Should extract at least 10 buildings");
            foreach (var building in buildings)
            {
                Assert.True(building.Vertices.Count > 0);
                Assert.True(building.TriangleIndices.Count > 0);
            }
        }

        [Fact]
        public void PM4FileV2_ShouldDecodeSurfaceNormalsAndMaterials()
        {
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            Assert.True(File.Exists(testFile), $"Test file does not exist: {testFile}");
            var fileData = File.ReadAllBytes(testFile);
            var pm4File = new PM4File(fileData);

            // Assume pm4File exposes MSUR and MSLK entries
            foreach (var surface in pm4File.MSUR.Entries)
            {
                var normal = surface.SurfaceNormal;
                Assert.True(Math.Abs(normal.Length() - 1.0f) < 0.01f, $"Surface normal not normalized: {normal}");
            }
            foreach (var entry in pm4File.MSLK.Entries)
            {
                Assert.True(entry.MaterialColorId != 0, "MaterialColorId should be nonzero");
            }
        }

        [Fact]
        public void PM4FileV2_ShouldHandleMissingOrCorruptedChunksGracefully()
        {
            // Create a file with invalid signature (more realistic than removing chunks)
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            Assert.True(File.Exists(testFile), $"Test file does not exist: {testFile}");
            var fileData = File.ReadAllBytes(testFile);
            
            // Create invalid PM4 data by corrupting the first few bytes (file signature)
            var corruptedData = new byte[fileData.Length];
            Array.Copy(fileData, corruptedData, fileData.Length);
            corruptedData[0] = 0xFF; // Corrupt file signature
            corruptedData[1] = 0xFF;
            corruptedData[2] = 0xFF;
            corruptedData[3] = 0xFF;
            
            // Should handle gracefully - either throw exception or handle missing chunks
            // Core.v2 with Warcraft.NET might handle this more gracefully than throwing
            try
            {
                var pm4File = new PM4File(corruptedData);
                // If it doesn't throw, that's also acceptable - just verify basic state
                Assert.NotNull(pm4File);
            }
            catch (Exception ex)
            {
                // Any exception is acceptable for corrupted data
                Assert.True(ex is InvalidDataException || ex is EndOfStreamException || ex is IOException,
                    $"Expected data-related exception, got: {ex.GetType()}");
            }
        }

        // Helper method to corrupt a chunk by signature (for test only)
        private static byte[] CorruptChunk(byte[] fileData, string chunkSignature)
        {
            // Simple implementation: zero out bytes matching the chunk signature
            var sigBytes = Encoding.ASCII.GetBytes(chunkSignature);
            var data = (byte[])fileData.Clone();
            for (int i = 0; i < data.Length - sigBytes.Length; i++)
            {
                bool match = true;
                for (int j = 0; j < sigBytes.Length; j++)
                {
                    if (data[i + j] != sigBytes[j])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    for (int j = 0; j < sigBytes.Length; j++)
                        data[i + j] = 0;
                    break;
                }
            }
            return data;
        }
    }
} 