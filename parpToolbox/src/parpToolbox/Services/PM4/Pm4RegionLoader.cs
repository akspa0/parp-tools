using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Utils;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Very first draft region loader: loads a single PM4 tile, then appends any sibling
/// tiles in the same folder whose filenames share the common prefix before the last
/// two numeric tokens (xx_yy). After concatenating, runs MSCN remapper to hook up
/// cross-tile vertices. This is a stepping-stone toward full 64×64 grid
/// reconstruction.
/// </summary>
internal sealed class Pm4RegionLoader
{
    private readonly Pm4Adapter _adapter = new();

    public Pm4Scene LoadRegion(string firstTilePath)
    {
        var dir = Path.GetDirectoryName(firstTilePath) ?? ".";
        var name = Path.GetFileNameWithoutExtension(firstTilePath);

        // crude pattern: strip trailing _XX_YY
        var parts = name.Split('_');
        if (parts.Length < 3)
        {
            ConsoleLogger.WriteLine("PM4 filename does not follow expected *_xx_yy pattern; loading single tile only.");
            var scene = _adapter.Load(firstTilePath);
            AttachMscn(scene);
            return scene;
        }
        var prefix = string.Join("_", parts.Take(parts.Length - 2));
        var candidateFiles = Directory.EnumerateFiles(dir, $"{prefix}_*.pm4");

        var scenes = new List<Pm4Scene>();
        foreach (var file in candidateFiles)
        {
            try
            {
                scenes.Add(_adapter.Load(file));
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Failed to load tile {file}: {ex.Message}");
            }
        }

        if (scenes.Count == 0)
            throw new InvalidOperationException("No PM4 tiles loaded for region.");

        // Begin merge – naive concatenation of vertices/indices/chunks.
        var baseScene = scenes[0];
        for (int i = 1; i < scenes.Count; i++)
        {
            var s = scenes[i];
            int vertexOffset = baseScene.Vertices.Count;
            baseScene.Vertices.AddRange(s.Vertices);
            baseScene.Indices.AddRange(s.Indices.Select(idx => idx + vertexOffset));
            baseScene.Surfaces.AddRange(s.Surfaces);
            baseScene.Links.AddRange(s.Links);
            baseScene.Placements.AddRange(s.Placements);
            // TODO: merge other chunks as needed
        }

        AttachMscn(baseScene);
        ConsoleLogger.WriteLine($"Region loader merged {scenes.Count} tiles. Vertices={baseScene.Vertices.Count}, Indices={baseScene.Indices.Count}");
        return baseScene;
    }

    private static void AttachMscn(Pm4Scene scene)
    {
        var mscn = scene.ExtraChunks.OfType<MscnChunk>().FirstOrDefault();
        if (mscn != null)
        {
            ConsoleLogger.WriteLine($"Applying MSCN remap: {mscn.Vertices.Count} exterior vertices");
            MscnRemapper.Apply(scene, mscn);
        //else
        //{
        //    ConsoleLogger.WriteLine("No MSCN chunk found – skipping remap.");
        //}
        }
        else
        {
            ConsoleLogger.WriteLine("No MSCN chunk found – skipping remap.");
        }
    }
}
