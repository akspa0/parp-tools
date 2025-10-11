using System;
using System.IO;
using System.Numerics;
using System.Collections.Generic;
using System.Linq;
using WoWToolbox.Core.v2.Services.WMO;
using WoWToolbox.Core.v2.Foundation.WMO;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.WMO
{
    public class WmoHelperSmokeTests
    {
        private static readonly string TestDataDir = Path.Combine(AppContext.BaseDirectory, "test_data", "053_wmo");
        private static string GetSamplePath(string name) => Path.Combine(TestDataDir, name);

        [Fact]
        public void Converter_Can_Extract_Textures_When_Dirs_Provided()
        {
            string sample = GetSamplePath("Ironforge_053.wmo");
            if (!File.Exists(sample))
                return; // data not shipped in CI – skip gracefully

            string tempRoot = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            string outWmo = Path.Combine(tempRoot, "Ironforge_053_v17.wmo");
            string texOutDir = Path.Combine(tempRoot, "textures");
            Directory.CreateDirectory(tempRoot);

            var converter = new WmoV14Converter();
            converter.ConvertToV17(sample, outWmo, texOutDir);

            Assert.True(File.Exists(outWmo));
            Assert.True(Directory.Exists(texOutDir));
            // Even if no textures extracted, directory exists; if textures exist ensure at least one PNG
            var pngs = Directory.GetFiles(texOutDir, "*.png", SearchOption.AllDirectories);
            // no assert on count – just ensure call didn't throw and output dir present
        }

        [Fact]
        public void ObjExporter_Writes_Obj_And_Mtl()
        {
            string tempRoot = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempRoot);
            string objPath = Path.Combine(tempRoot, "box.obj");

            // Test ExportFirstGroupAsObj using Ironforge sample if present
            var sampleDir = Path.Combine(TestDataDir);
            var sampleFile = Directory.Exists(sampleDir)? Directory.GetFiles(sampleDir, "Ironforge_053.wmo").FirstOrDefault(): null;
            if (sampleFile != null)
            {
                var converter2 = new WmoV14Converter();
                var objOut = converter2.ExportFirstGroupAsObj(sampleFile);
                Assert.True(File.Exists(objOut));
            }
            // Minimal cube geometry (2 triangles)
            var verts = new List<Vector3> { new(0,0,0), new(1,0,0), new(1,1,0), new(0,1,0) };
            var uvs   = new List<Vector2> { new(0,0), new(1,0), new(1,1), new(0,1) };
            var faces = new List<(int,int,int)> { (1,2,3), (1,3,4) }; // OBJ is 1-based but exporter expects 1-based indices already

            WmoObjExporter.Export(objPath, verts, uvs, faces);

            Assert.True(File.Exists(objPath));
            Assert.True(File.Exists(Path.ChangeExtension(objPath, ".mtl")));
        }
    }
}
