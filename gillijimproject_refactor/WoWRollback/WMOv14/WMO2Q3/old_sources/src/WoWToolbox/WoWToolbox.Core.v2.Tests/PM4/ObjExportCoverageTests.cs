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

        [Fact(DisplayName = "OBJ exporter produces file for every PM4 with vertices")]
        public async Task Exporter_Produces_Obj_When_Geometry_Present()
        {
            Assert.True(Directory.Exists(Pm4DataRoot), $"Test data folder not found: {Pm4DataRoot}");

            var pm4Files = Directory.EnumerateFiles(Pm4DataRoot, "*.pm4", SearchOption.AllDirectories).ToList();
            Assert.NotEmpty(pm4Files);

            var failures = new List<string>();
            string tmpRoot = Path.Combine(Path.GetTempPath(), "pm4_obj_cov", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tmpRoot);

            foreach (var file in pm4Files)
            {
                PM4File pm4;
                try { pm4 = PM4File.FromFile(file); }
                catch (Exception ex)
                {
                    failures.Add($"LOAD_FAIL,{Path.GetFileName(file)},{ex.Message.Replace(',', ';')}");
                    continue;
                }

                int vertCount = (pm4.MSPV?.Vertices.Count ?? 0) + (pm4.MSVT?.Vertices.Count ?? 0);
                if (vertCount == 0) continue; // skip tiles without geometry

                string objPath = Path.Combine(tmpRoot, Path.GetFileNameWithoutExtension(file) + ".obj");
                try
                {
                    await Pm4ObjExporter.ExportAsync(pm4, objPath, Path.GetFileName(file));
                    if (!File.Exists(objPath) || new FileInfo(objPath).Length == 0)
                        failures.Add($"NO_OBJ,{Path.GetFileName(file)}");
                }
                catch (Exception ex)
                {
                    failures.Add($"EXPORT_FAIL,{Path.GetFileName(file)},{ex.Message.Replace(',', ';')}");
                }
            }

            if (failures.Count > 0)
            {
                string diagCsv = Path.Combine(tmpRoot, "obj_export_coverage_failures.csv");
                File.WriteAllLines(diagCsv, failures);
                Assert.True(false, $"{failures.Count} PM4 files failed OBJ export. See {diagCsv}");
            }
        }
    }
}
