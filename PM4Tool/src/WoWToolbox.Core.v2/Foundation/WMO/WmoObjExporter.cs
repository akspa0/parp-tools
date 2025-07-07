using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using Warcraft.NET.Files.BLP; // Provides BLP texture handling
using BLPFile = Warcraft.NET.Files.BLP.BLP;

namespace WoWToolbox.Core.v2.Foundation.WMO
{
    /// <summary>
    /// Minimal OBJ/MTL exporter used by both v14 and v17 WMO pipelines. Not feature-complete but
    /// sufficient for validation tests and PM4 batch output.
    /// </summary>
    public static class WmoObjExporter
    {
        public static string Export(string objPath,
                                  IReadOnlyList<Vector3> vertices,
                                  IReadOnlyList<Vector2> uvs,
                                  IReadOnlyList<(int a,int b,int c)> faces,
                                  IReadOnlyList<string>? textureNames = null,
                                  IReadOnlyList<Vector3>? normals = null,
                                  IReadOnlyList<ushort>? faceMaterialIds = null)
        {
            // If caller passed a directory (or string ending with slash), generate file inside ProjectOutput
            bool isDirectoryHint = objPath.EndsWith(Path.DirectorySeparatorChar) || string.IsNullOrEmpty(Path.GetExtension(objPath));
            bool dirCollision = !isDirectoryHint && Path.GetExtension(objPath).Equals(".obj", StringComparison.OrdinalIgnoreCase) && Directory.Exists(objPath);
            if (isDirectoryHint || dirCollision)
            {
                objPath = Infrastructure.ProjectOutput.GetPath($"wmo_{DateTime.Now:HHmmss}.obj");
            }
            if (vertices.Count == 0) throw new ArgumentException("No vertices", nameof(vertices));
            string baseName = Path.GetFileNameWithoutExtension(objPath);
            string mtlName = baseName + ".mtl";

            // Prepare material dictionary
            var materialSet = new HashSet<ushort>();
            if (faceMaterialIds != null)
            {
                foreach (var id in faceMaterialIds)
                    materialSet.Add(id);
            }
            if (materialSet.Count == 0)
                materialSet.Add(0); // default material 0

            // Ensure destination directory exists
            Directory.CreateDirectory(Path.GetDirectoryName(objPath)!);
            using var objWriter = new StreamWriter(objPath, false, Encoding.UTF8);
            objWriter.WriteLine($"mtllib {mtlName}");
            objWriter.WriteLine("o mesh0");
            objWriter.WriteLine("usemtl material0"); // will switch as needed

            // vertices
            foreach (var v in vertices)
                objWriter.WriteLine(string.Create(CultureInfo.InvariantCulture, $"v {v.X} {v.Y} {v.Z}"));

            bool hasUvs = uvs.Count == vertices.Count;
            bool hasNormals = normals != null && normals.Count == vertices.Count;

            if (hasUvs)
            {
                foreach (var uv in uvs)
                    objWriter.WriteLine(string.Create(CultureInfo.InvariantCulture, $"vt {uv.X} {1-uv.Y}"));
            }
            if (hasNormals)
            {
                foreach (var n in normals!)
                    objWriter.WriteLine(string.Create(CultureInfo.InvariantCulture, $"vn {n.X} {n.Y} {n.Z}"));
            }
            // faces (OBJ is 1-based)
            int faceIndex = 0;
            ushort currentMat = 0;
            foreach (var (a,b,c) in faces)
            {
                if (faceMaterialIds != null && faceIndex < faceMaterialIds.Count)
                {
                    ushort matId = faceMaterialIds[faceIndex];
                    if (matId != currentMat)
                    {
                        currentMat = matId;
                        objWriter.WriteLine($"usemtl material{matId}");
                    }
                }
                if (hasUvs && hasNormals)
                    objWriter.WriteLine($"f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}");
                else if (hasUvs)
                    objWriter.WriteLine($"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}");
                else if (hasNormals)
                    objWriter.WriteLine($"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}");
                else
                    objWriter.WriteLine($"f {a+1} {b+1} {c+1}");
            }

            // Write MTL
            var mtlPath = Path.Combine(Path.GetDirectoryName(objPath)!, mtlName);
            using var mtlWriter = new StreamWriter(mtlPath, false, Encoding.UTF8);
            foreach (ushort matId in materialSet)
            {
                mtlWriter.WriteLine($"newmtl material{matId}");
                mtlWriter.WriteLine("Kd 1.0 1.0 1.0");
                string tex = (textureNames != null && matId < textureNames.Count) ? textureNames[matId] : null;
                if (!string.IsNullOrEmpty(tex))
                    mtlWriter.WriteLine($"map_Kd {tex}");
            }

            return objPath;
        }

        /// <summary>
        /// Extracts a BLP texture to PNG next to the OBJ and returns relative filename.
        /// </summary>
        public static string ExtractBlpToPng(byte[] blpData, string outputDir, string baseName)
        {
            var blp = new BLPFile(blpData);
            using var img = blp.GetMipMap(0);
            
            Directory.CreateDirectory(outputDir);
            string pngName = baseName + ".png";
            string pngPath = Path.Combine(outputDir, pngName);
            img.Save(pngPath, new PngEncoder());
            return pngName;
        }

        /// <summary>
        /// Convenience helper for cases where only vertices and triangle indices are available (no UVs).
        /// </summary>
        public static void WriteObj(string objPath, IReadOnlyList<Vector3> vertices, IReadOnlyList<int> indices)
        {
            if (indices.Count % 3 != 0)
                throw new ArgumentException("Index list length must be a multiple of 3 (triangles)", nameof(indices));
            var faces = new List<(int,int,int)>(indices.Count/3);
            for (int i = 0; i < indices.Count; i += 3)
                faces.Add((indices[i], indices[i+1], indices[i+2]));
            Export(objPath, vertices, Array.Empty<Vector2>(), faces);
        }
    }
}
