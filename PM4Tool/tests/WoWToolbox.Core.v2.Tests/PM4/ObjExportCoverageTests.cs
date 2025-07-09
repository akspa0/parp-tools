using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using WoWToolbox.Core.v2.Foundation.PM4;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class ObjExportCoverageTests
    {
        private static readonly string Pm4DataRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "test_data", "development_335"));

        [Fact(DisplayName = "PM4 OBJ exporter should create OBJ for every PM4 that contains vertices")]        
        public async Task ObjExport_CoversAllTilesWithGeometry()
        {
            Assert.True(Directory.Exists(Pm4DataRoot), $"Test data folder not found: {Pm4DataRoot}");

            var pm4Files = Directory.EnumerateFiles(Pm4DataRoot, "*.pm4", SearchOption.AllDirectories).ToList();
            Assert.NotEmpty(pm4Files);

            var failures = new List<string>();
            string tempRoot = Path.Combine(Path.GetTempPath(), "pm4_obj_test", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempRoot);

            foreach (var file in pm4Files)
            {
                PM4File pm4;
                try
                {
                    pm4 = PM4File.FromFile(file);
                }
                catch (Exception ex)
                {
                    failures.Add($"LOAD_FAIL,{Path.GetFileName(file)},{ex.Message.Replace(',', ';')}");
                    continue;
                }

                int vertexCount = (pm4.MSPV?.Vertices.Count ?? 0) + (pm4.MSVT?.Vertices.Count ?? 0);
                if (vertexCount == 0)
                {
                    // No geometry expected – skip
                    continue;
                }

                string objOut = Path.Combine(tempRoot, Path.GetFileNameWithoutExtension(file) + ".obj");
                try
                {
                    await Pm4ObjExporter.ExportAsync(pm4, objOut, Path.GetFileName(file));
                    if (!File.Exists(objOut) || new FileInfo(objOut).Length == 0)
                    {
                        failures.Add($"NO_OBJ_WRITTEN,{Path.GetFileName(file)}");
                    }
                }
                catch (Exception ex)
                {
                    failures.Add($"EXPORT_FAIL,{Path.GetFileName(file)},{ex.Message.Replace(',', ';')}");
                }
            }

            if (failures.Count > 0)
            {
                string diagPath = Path.Combine(tempRoot, "obj_export_failures.csv");
                File.WriteAllLines(diagPath, failures);
                Assert.True(false, $"{failures.Count} PM4 files with geometry failed OBJ export. See {diagPath}");
            }
        }
    }
}
