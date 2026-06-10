using System;
using System.IO;
using System.Linq;

using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Services.PM4;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class MslkObjectMeshExporterTests
    {
        private static readonly string TestDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "test_data", "development_335"));
        private static readonly string SamplePm4Name = "development_00_00.pm4";

        [Fact]
        public void ExportAllGroupsAsObj_WithRealData_ProducesObjFiles()
        {
            var pm4Path = Path.Combine(TestDataPath, SamplePm4Name);
            if (!File.Exists(pm4Path))
            {
                // Real data missing on CI – skip.
                return;
            }

            var outputRoot = Path.Combine(Path.GetTempPath(), $"mslk_export_test_{Guid.NewGuid():N}");
            Directory.CreateDirectory(outputRoot);

            try
            {
                // Load PM4 file (v2 model)
                var pm4 = PM4File.FromFile(pm4Path);

                // Export groups OBJ
                var exporter = new MslkObjectMeshExporter();
                exporter.ExportAllGroupsAsObj(pm4, outputRoot);

                // Basic assertions: At least one OBJ written with vertex data
                var objFiles = Directory.GetFiles(outputRoot, "*.obj", SearchOption.AllDirectories);
                Assert.NotEmpty(objFiles);

                foreach (var obj in objFiles)
                {
                    var vertexLines = File.ReadLines(obj).Count(l => l.StartsWith("v "));
                    var faceLines = File.ReadLines(obj).Count(l => l.StartsWith("f "));
                    Assert.True(vertexLines > 0, $"OBJ {obj} has no vertices.");
                    Assert.True(faceLines > 0, $"OBJ {obj} has no faces.");
                }
            }
            finally
            {
                // Clean-up temp directory
                if (Directory.Exists(outputRoot))
                {
                    Directory.Delete(outputRoot, true);
                }
            }
        }
    }
}
