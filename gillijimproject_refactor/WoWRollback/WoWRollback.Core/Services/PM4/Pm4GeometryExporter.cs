using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Exports PM4 pathfinding geometry to OBJ format for visualization and comparison.
/// </summary>
public sealed class Pm4GeometryExporter
{
    /// <summary>
    /// Export all PM4 geometry from a file to OBJ format.
    /// </summary>
    public void ExportToObj(string pm4Path, string outputPath)
    {
        Console.WriteLine($"[INFO] Exporting PM4 geometry: {pm4Path}");

        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"[ERROR] PM4 file not found: {pm4Path}");
            return;
        }

        var vertices = new List<Vector3>();
        var indices = new List<int>();

        // Parse PM4 file for MSPV (vertices) and MSPI (indices)
        using var fs = File.OpenRead(pm4Path);
        using var br = new BinaryReader(fs);

        while (fs.Position < fs.Length - 8)
        {
            var chunkIdBytes = br.ReadBytes(4);
            var chunkId = new string(new[] { (char)chunkIdBytes[3], (char)chunkIdBytes[2], (char)chunkIdBytes[1], (char)chunkIdBytes[0] });
            var chunkSize = br.ReadUInt32();

            var chunkEnd = fs.Position + chunkSize;

            switch (chunkId)
            {
                case "MSPV": // Path vertices
                    vertices = ParseMspv(br, chunkSize);
                    Console.WriteLine($"[INFO] MSPV: {vertices.Count} vertices");
                    break;
                case "MSPI": // Path indices
                    indices = ParseMspi(br, chunkSize);
                    Console.WriteLine($"[INFO] MSPI: {indices.Count} indices ({indices.Count / 3} triangles)");
                    break;
            }

            fs.Position = chunkEnd;
        }

        if (vertices.Count == 0 || indices.Count == 0)
        {
            Console.WriteLine("[WARN] No geometry found in PM4 file");
            return;
        }

        // Write OBJ file
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("# PM4 Pathfinding Geometry");
        sw.WriteLine($"# Source: {Path.GetFileName(pm4Path)}");
        sw.WriteLine($"# Vertices: {vertices.Count}");
        sw.WriteLine($"# Triangles: {indices.Count / 3}");
        sw.WriteLine();

        // Write vertices
        foreach (var v in vertices)
        {
            // Apply coordinate transformation (PM4 uses different coordinate system)
            // X is negated for proper orientation
            sw.WriteLine($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}");
        }

        sw.WriteLine();
        sw.WriteLine("g PM4_Geometry");

        // Write faces (1-indexed)
        for (int i = 0; i < indices.Count; i += 3)
        {
            if (i + 2 < indices.Count)
            {
                int a = indices[i] + 1;
                int b = indices[i + 1] + 1;
                int c = indices[i + 2] + 1;

                // Validate indices
                if (a > 0 && a <= vertices.Count &&
                    b > 0 && b <= vertices.Count &&
                    c > 0 && c <= vertices.Count)
                {
                    sw.WriteLine($"f {a} {b} {c}");
                }
            }
        }

        Console.WriteLine($"[INFO] Exported PM4 geometry to: {outputPath}");

        // Compute and report bounds
        if (vertices.Count > 0)
        {
            var minX = float.MaxValue; var maxX = float.MinValue;
            var minY = float.MaxValue; var maxY = float.MinValue;
            var minZ = float.MaxValue; var maxZ = float.MinValue;

            foreach (var v in vertices)
            {
                if (v.X < minX) minX = v.X;
                if (v.X > maxX) maxX = v.X;
                if (v.Y < minY) minY = v.Y;
                if (v.Y > maxY) maxY = v.Y;
                if (v.Z < minZ) minZ = v.Z;
                if (v.Z > maxZ) maxZ = v.Z;
            }

            Console.WriteLine($"[INFO] Bounds: ({minX:F1}, {minY:F1}, {minZ:F1}) to ({maxX:F1}, {maxY:F1}, {maxZ:F1})");
        }
    }

    private List<Vector3> ParseMspv(BinaryReader br, uint chunkSize)
    {
        var vertices = new List<Vector3>();
        int count = (int)(chunkSize / 12); // 3 floats per vertex

        for (int i = 0; i < count; i++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            vertices.Add(new Vector3(x, y, z));
        }

        return vertices;
    }

    private List<int> ParseMspi(BinaryReader br, uint chunkSize)
    {
        var indices = new List<int>();
        int count = (int)(chunkSize / 4); // 4 bytes per index (uint32)

        for (int i = 0; i < count; i++)
        {
            indices.Add((int)br.ReadUInt32());
        }

        return indices;
    }
}
