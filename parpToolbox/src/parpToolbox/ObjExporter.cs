using System.IO;
using WoWFormatLib.Structs.WMO;

namespace parpToolbox
{
    public static class ObjExporter
    { 
        public static void Export(WMO wmo, string filename)
        {
            var outputDir = Path.GetDirectoryName(filename);
            var modelName = Path.GetFileNameWithoutExtension(filename);

            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            var objPath = Path.Combine(outputDir, modelName + ".obj");
            var mtlPath = Path.Combine(outputDir, modelName + ".mtl");

            WriteMtlFile(wmo, mtlPath);

            using (var writer = new StreamWriter(objPath))
            {
                writer.WriteLine($"# Exported from parpToolbox");
                writer.WriteLine($"mtllib {modelName}.mtl");

                int vertexOffset = 1; // OBJ indices are 1-based
                int normalOffset = 1;
                int texCoordOffset = 1;

                for (int i = 0; i < wmo.group.Length; i++)
                {
                    var group = wmo.group[i];
                    if (group == null || group.mogp.vertices == null) continue;

                    writer.WriteLine($"\ng group_{i}");

                    // Write vertices
                    foreach (var vert in group.mogp.vertices)
                    {
                        writer.WriteLine($"v {vert.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {vert.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {vert.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                    }

                    // Write normals
                    foreach (var normal in group.mogp.normals)
                    {
                        writer.WriteLine($"vn {normal.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {normal.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {normal.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                    }

                    // Write texture coordinates
                    foreach (var texCoord in group.mogp.textureCoords)
                    {
                        writer.WriteLine($"vt {texCoord.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {1 - texCoord.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)}"); // OBJ texcoords are flipped on Y
                    }

                    // Write faces
                    for (var j = 0; j < group.mogp.renderBatches.Length; j++)
                    {
                        var batch = group.mogp.renderBatches[j];
                        writer.WriteLine($"usemtl material_{batch.materialID}");

                        var end = batch.indexStart + batch.indexCount;
                        for (var k = batch.indexStart; k < end; k += 3)
                        {
                            var i1 = group.mogp.indices[k] + vertexOffset;
                            var i2 = group.mogp.indices[k + 1] + vertexOffset;
                            var i3 = group.mogp.indices[k + 2] + vertexOffset;

                            var n1 = group.mogp.indices[k] + normalOffset;
                            var n2 = group.mogp.indices[k + 1] + normalOffset;
                            var n3 = group.mogp.indices[k + 2] + normalOffset;

                            var t1 = group.mogp.indices[k] + texCoordOffset;
                            var t2 = group.mogp.indices[k + 1] + texCoordOffset;
                            var t3 = group.mogp.indices[k + 2] + texCoordOffset;

                            writer.WriteLine($"f {i1}/{t1}/{n1} {i2}/{t2}/{n2} {i3}/{t3}/{n3}");
                        }
                    }


                    vertexOffset += group.mogp.vertices.Length;
                    normalOffset += group.mogp.normals.Length;
                    texCoordOffset += group.mogp.textureCoords.Length;
                }
            }
        }

        private static void WriteMtlFile(WMO wmo, string mtlPath)
        {
            using (var writer = new StreamWriter(mtlPath))
            {
                for (int i = 0; i < wmo.materials.Length; i++)
                {
                    var material = wmo.materials[i];
                    var texture = wmo.textures[material.texture1];
                    var textureName = Path.GetFileName(texture);

                    writer.WriteLine($"newmtl material_{i}");
                    writer.WriteLine("Ka 1.0 1.0 1.0"); // Ambient color
                    writer.WriteLine("Kd 1.0 1.0 1.0"); // Diffuse color
                    writer.WriteLine("Ks 0.0 0.0 0.0"); // Specular color
                    writer.WriteLine("d 1.0"); // Alpha
                    writer.WriteLine("illum 2"); // Illumination model
                    writer.WriteLine($"map_Kd {textureName}"); // Diffuse texture map
                    writer.WriteLine();
                }
            }
        }
    }
}
