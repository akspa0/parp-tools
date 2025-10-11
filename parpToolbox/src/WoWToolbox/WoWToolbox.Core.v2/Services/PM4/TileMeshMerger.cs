using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.v2.Services.ADT;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Utility that merges a flat ADT-style white plate mesh with an existing OBJ produced from a PM4 tile.
    /// The merged OBJ keeps plate geometry first, followed by the original OBJ's vertices (with face indices shifted).
    /// </summary>
    public static class TileMeshMerger
    {
        /// <summary>
        /// Creates a merged OBJ combining a white plate (129×129 grid) with an existing tile OBJ.
        /// </summary>
        /// <param name="pm4ObjPath">Path to the existing OBJ produced for the PM4 tile.</param>
        /// <param name="tileX">Tile X coordinate (0-63).</param>
        /// <param name="tileY">Tile Y coordinate.</param>
        /// <param name="outputPath">Destination path for merged OBJ.</param>
        public static void MergeWithPlate(string pm4ObjPath, int tileX, int tileY, string outputPath)
        {
            if (!File.Exists(pm4ObjPath))
                throw new FileNotFoundException($"PM4 OBJ not found: {pm4ObjPath}");

            var plateVerts = new List<Vector3>();
            var plateTris = new List<(int a, int b, int c)>();
            AdtFlatPlateBuilder.BuildTile(tileX, tileY, plateVerts, plateTris);

            using var reader = new StreamReader(pm4ObjPath);
            var origLines = reader.ReadToEnd().Split('\n');
            int origVertCount = origLines.Count(l => l.StartsWith("v "));

            // Prepare writer
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
            using var writer = new StreamWriter(outputPath);
            writer.WriteLine($"# merged OBJ generated {DateTime.Now:O}");

            // 1. write plate
            writer.WriteLine("g white_plate");
            foreach (var v in plateVerts)
                writer.WriteLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
            foreach (var (a, b, c) in plateTris)
                writer.WriteLine($"f {a + 1} {b + 1} {c + 1}");

            int vertOffset = plateVerts.Count; // shift for original vertices

            // 2. copy original OBJ with vertex index shift
            foreach (var raw in origLines)
            {
                var line = raw.TrimEnd('\r');
                if (line.StartsWith("v ") || line.StartsWith("vt ") || line.StartsWith("vn "))
                {
                    writer.WriteLine(line); // vertices/normals/uv unchanged (vertex indices auto-sequential)
                    continue;
                }
                else if (line.StartsWith("f "))
                {
                    // shift vertex indices before any '/'
                    string shifted = ShiftFaceLine(line, vertOffset);
                    writer.WriteLine(shifted);
                }
                else if (!string.IsNullOrWhiteSpace(line) && !line.StartsWith("#"))
                {
                    writer.WriteLine(line);
                }
            }
        }

        private static string ShiftFaceLine(string faceLine, int offset)
        {
            // faceLine like "f 1/2/3 4/5/6 7/8/9" or "f 12 13 14"
            var parts = faceLine.Split(' ');
            if (parts.Length < 4) return faceLine; // not standard face? keep
            for (int i = 1; i < parts.Length; i++)
            {
                string elem = parts[i];
                var subParts = elem.Split('/');
                if (int.TryParse(subParts[0], out int idx))
                    subParts[0] = (idx + offset).ToString();
                if (subParts.Length > 1 && int.TryParse(subParts[1], out _))
                {
                    // texture coordinate indices remain unchanged (we didn't add vt)
                }
                if (subParts.Length > 2 && int.TryParse(subParts[2], out _))
                {
                    // normal indices remain unchanged (we didn't add vn)
                }
                parts[i] = string.Join('/', subParts);
            }
            return string.Join(' ', parts);
        }
    }
}
