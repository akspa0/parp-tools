using System;
using System.Collections.Generic;
using System.IO;

namespace GillijimProject.Next.Core.PM4;

/// <summary>
/// Minimal PM4 loader scaffold. Wires to ParpToolbox PM4 loaders and maps into a lightweight scene.
/// </summary>
public static class Pm4Loader
{
    /// <summary>
    /// Load a PM4 scene from a file or directory. When includeAdjacent=true, it aggregates neighboring tiles
    /// using ParpToolbox's Pm4Adapter.LoadRegion, which preserves placements (MPRL) for hierarchical assembly.
    /// </summary>
    public static Pm4Scene Load(string path, bool includeAdjacent, bool applyMscnRemap = true)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("path");

        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        if (includeAdjacent)
        {
            // Use adapter's region loader to ensure Placements are present
            string firstTilePath = File.Exists(path) ? path : FirstOrThrow(path, "*.pm4");
            var parpScene = adapter.LoadRegion(firstTilePath, new ParpToolbox.Services.PM4.Pm4LoadOptions());
            return MapFromParp(parpScene);
        }
        else
        {
            // Single tile: if a directory is passed, take the first *.pm4
            string file = path;
            if (Directory.Exists(path))
            {
                file = FirstOrThrow(path, "*.pm4");
            }
            var scene = adapter.Load(file);
            return MapFromParp(scene);
        }
    }

    /// <summary>
    /// Load and return the raw ParpToolbox PM4 scene for use with hierarchical assemblers.
    /// </summary>
    public static ParpToolbox.Formats.PM4.Pm4Scene LoadParp(string path, bool includeAdjacent, bool applyMscnRemap = true)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("path");

        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        if (includeAdjacent)
        {
            string firstTilePath = File.Exists(path) ? path : FirstOrThrow(path, "*.pm4");
            return adapter.LoadRegion(firstTilePath, new ParpToolbox.Services.PM4.Pm4LoadOptions());
        }
        else
        {
            string file = path;
            if (Directory.Exists(path))
            {
                file = FirstOrThrow(path, "*.pm4");
            }
            return adapter.Load(file);
        }
    }

    /// <summary>
    /// Back-compat helper for earlier scaffold usage.
    /// </summary>
    public static Pm4Scene LoadSingle(string path)
    {
        return Load(path, includeAdjacent: false, applyMscnRemap: true);
    }

    private static Pm4Scene MapFromParp(ParpToolbox.Formats.PM4.Pm4Scene src)
    {
        var dst = new Pm4Scene();
        // Vertices
        foreach (var v in src.Vertices)
        {
            dst.Vertices.Add(new P3(v.X, v.Y, v.Z));
        }
        // Triangles (already absolute indices)
        foreach (var t in src.Triangles)
        {
            dst.Triangles.Add((t.Item1, t.Item2, t.Item3));
        }
        // Flat index buffer fallback
        if (src.Indices is not null && src.Indices.Count > 0)
        {
            dst.Indices.AddRange(src.Indices);
        }
        // MSCN anchors (optional)
        if (src.MscnVertices is not null && src.MscnVertices.Count > 0)
        {
            for (int i = 0; i < src.MscnVertices.Count; i++)
            {
                var p = src.MscnVertices[i];
                dst.MscnAnchors.Add(new P3(p.X, p.Y, p.Z));
                if (src.MscnVertexTileIds is not null && i < src.MscnVertexTileIds.Count)
                {
                    dst.MscnTileIds.Add(src.MscnVertexTileIds[i]);
                }
            }
        }
        return dst;
    }

    private static string FirstOrThrow(string dir, string pattern)
    {
        var files = Directory.GetFiles(dir, pattern);
        if (files.Length == 0)
            throw new FileNotFoundException($"No files matching '{pattern}' in '{dir}'");
        return files[0];
    }
}
