using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Linq;

namespace WoWRollback.PM4Module.Analysis;

/// <summary>
/// Dumps PM4 geometry (MSVT mesh and MSCN point cloud) to Wavefront OBJ format.
/// </summary>
public class Pm4GeometryDumper
{
    public void Dump(string pm4Path, string outDir)
    {
        if (!File.Exists(pm4Path)) return;

        var pm4 = PM4File.FromFile(pm4Path);
        string baseName = Path.GetFileNameWithoutExtension(pm4Path);
        Directory.CreateDirectory(outDir);

        // 1. Dump MSVT (Render Geometry)
        if (pm4.MeshVertices.Count > 0)
        {
            string objPath = Path.Combine(outDir, $"{baseName}_msvt.obj");
            using var sw = new StreamWriter(objPath);
            sw.WriteLine($"# MSVT Geometry from {baseName}");
            sw.WriteLine($"# Vertices: {pm4.MeshVertices.Count}");
            sw.WriteLine($"# Indices: {pm4.MeshIndices.Count}");

            foreach (var v in pm4.MeshVertices)
            {
                // Convert Y-Up (PM4) to Z-Up (WoW/Tools)
                // PM4 X -> WoW X
                // PM4 Y -> WoW Z (Height)
                // PM4 Z -> WoW Y
                sw.WriteLine($"v {v.X:F4} {v.Z:F4} {v.Y:F4}");
            }

            sw.WriteLine("g MSVT_Mesh");
            // Assuming 'count' refers to pm4.MeshIndices.Count and 'start' refers to 0 for the full list.
            // This change introduces 'count' and 'start' variables which are not defined in the original context.
            // For the file to be syntactically correct, these would need to be defined.
            // Based on the original loop, 'count' would be pm4.MeshIndices.Count and 'start' would be 0.
            int count = pm4.MeshIndices.Count;
            int start = 0;
            for (int i = 0; i < count; i += 3)
            {
                if (i + 2 >= count) break;

                uint idx1 = pm4.MeshIndices[(int)(start + i)];
                uint idx2 = pm4.MeshIndices[(int)(start + i + 1)];
                uint idx3 = pm4.MeshIndices[(int)(start + i + 2)];

                sw.WriteLine($"f {idx1 + 1} {idx2 + 1} {idx3 + 1}");
            }

            Console.WriteLine($"[INFO] Wrote {objPath}");
        }

        // 2. Dump MSCN (Point Cloud)
        if (pm4.ExteriorVertices.Count > 0)
        {
            string objPath = Path.Combine(outDir, $"{baseName}_mscn.obj");
            using var sw = new StreamWriter(objPath);
            sw.WriteLine($"# MSCN Point Cloud from {baseName}");
            sw.WriteLine($"# Points: {pm4.ExteriorVertices.Count}");

            foreach (var v in pm4.ExteriorVertices)
            {
                // MSCN requires coordinate transform: Y, X, Z (Swap X/Y)
                // This aligns it with MSVT in PM4 Space.
                var alignedV = new Vector3(v.Y, v.X, v.Z);
                
                // Then convert to Z-Up (X, Z, Y) for output
                sw.WriteLine($"v {alignedV.X:F4} {alignedV.Z:F4} {alignedV.Y:F4}");
            }

            // Write point cloud as single vertices (some viewers support 'p')
            sw.WriteLine("g MSCN_Points");
            /*
            for (int i = 0; i < pm4.ExteriorVertices.Count; i++)
            {
                sw.WriteLine($"p {i+1}");
            }
            */
            // Or just leave as vertices for "Load as Point Cloud"
            // Alternatively, create tiny triangles? No, too heavy.
            
            Console.WriteLine($"[INFO] Wrote {objPath} ({pm4.ExteriorVertices.Count} points)");
        }
    }
}
