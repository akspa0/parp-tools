using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common; // for IIffChunk


namespace PM4Rebuilder;

/// <summary>
/// Diagnostic utilities to dump MSCN vertex data and MSLK field values for offline analysis.
/// Generates CSV and OBJ files in the specified output directory.
/// </summary>
internal static class MscnAnalyzer
{
    // No scaling; MSCN vertices are already in correct units

    /// <summary>
    /// Dumps MSCN vertices (raw and scaled) and MSLK entry fields to disk.
    /// CSV files are UTF-8, OBJ uses scaled coordinates.
    /// </summary>
    /// <param name="scene">Loaded PM4 scene.</param>
    /// <param name="outDir">Output directory (already created).</param>
    public static void Dump(Pm4Scene scene, string outDir)
    {
        var mscn = scene.ExtraChunks.FirstOrDefault(ch => ch.GetType().Name.Contains("Mscn"));
        if (mscn == null)
        {
            Console.WriteLine("[MscnAnalyzer] No MSCN chunk present – nothing to dump.");
            return;
        }

        Directory.CreateDirectory(outDir);
        DumpMscnVertices(mscn, outDir);
        DumpMslkFields(scene, outDir);
    }

    private static void DumpMscnVertices(object mscn, string outDir)
    {
        var csvPath = Path.Combine(outDir, "mscn_vertices.csv");
        var objPath = Path.Combine(outDir, "mscn_points.obj");

        using var csv = new StreamWriter(csvPath); // UTF-8 by default
        csv.WriteLine("ID,X,Y,Z");

        using var obj = new StreamWriter(objPath);
        obj.WriteLine("# MSCN vertices (raw coordinates)");

        // Use reflection to access Vertices (assumed List<C3Vectori>)
        var vertsProp = mscn.GetType().GetProperty("Vertices");
        if (vertsProp == null)
        {
            Console.WriteLine("[MscnAnalyzer] MSCN chunk does not expose Vertices property – aborting dump.");
            return;
        }
        var vertices = vertsProp.GetValue(mscn) as System.Collections.IEnumerable;
        if (vertices == null)
        {
            Console.WriteLine("[MscnAnalyzer] MSCN vertices is null – aborting dump.");
            return;
        }

        int id = 1;
        foreach (var vObj in vertices)
        {
            dynamic v = vObj; // expects fields X,Y,Z (int)
            var raw = new Vector3((float)v.X, (float)v.Y, (float)v.Z);

            csv.WriteLine(string.Join(',',
                id,
                v.X.ToString(CultureInfo.InvariantCulture),
                v.Y.ToString(CultureInfo.InvariantCulture),
                v.Z.ToString(CultureInfo.InvariantCulture)));

            obj.WriteLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
            id++;
        }

        Console.WriteLine($"[MscnAnalyzer] Dumped {id - 1:N0} MSCN vertices to {csvPath} and {objPath}");
    }

    private static void DumpMslkFields(Pm4Scene scene, string outDir)
    {
        if (!scene.Links.Any())
        {
            Console.WriteLine("[MscnAnalyzer] No MSLK entries in scene – skipping MSLK dump.");
            return;
        }

        var csvPath = Path.Combine(outDir, "mslk_fields.csv");
        using var csv = new StreamWriter(csvPath);

        // Header – reflect public properties of first entry
        var props = scene.Links[0].GetType().GetProperties();
        csv.WriteLine(string.Join(',', new[] { "RowIndex" }.Concat(props.Select(p => p.Name))));

        for (int i = 0; i < scene.Links.Count; i++)
        {
            var entry = scene.Links[i];
            var values = new List<string> { i.ToString() };
            foreach (var p in props)
            {
                var val = p.GetValue(entry);
                values.Add(val?.ToString() ?? string.Empty);
            }
            csv.WriteLine(string.Join(',', values));
        }

        Console.WriteLine($"[MscnAnalyzer] Dumped {scene.Links.Count:N0} MSLK rows to {csvPath}");
    }
}
