using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Region-loading extension for <see cref="Pm4Adapter"/> implemented as a partial class.
/// Provides unified cross-tile PM4 loading so that callers no longer need the separate
/// <c>Pm4RegionLoader</c> helper.
/// </summary>
public sealed partial class Pm4Adapter
{
    /// <summary>
    /// Loads a complete region of PM4 tiles using default <see cref="Pm4LoadOptions"/>.
    /// </summary>
    public Pm4Scene LoadRegion(string firstTilePath) => LoadRegion(firstTilePath, new Pm4LoadOptions());

    /// <summary>
    /// Loads a complete region of PM4 tiles based on the path to the first tile, merging
    /// vertices, indices, and related chunk data to resolve cross-tile references.
    /// </summary>
    /// <remarks>
    /// KEY DISCOVERY: Vertex indices routinely reference vertices stored in adjacent tiles.
    /// Processing a single tile in isolation therefore loses ~64 % of geometry.  Region
    /// loading resolves this by merging up to hundreds of neighbour tiles and applying the
    /// MSCN remapper for any remaining cross-tile vertex references.
    /// </remarks>
    /// <param name="firstTilePath">Absolute path to a single <c>*.pm4</c> tile (e.g. <c>development_00_00.pm4</c>)</param>
    /// <param name="options">Load options controlling validation / analysis settings.</param>
    public Pm4Scene LoadRegion(string firstTilePath, Pm4LoadOptions options)
    {
        if (string.IsNullOrWhiteSpace(firstTilePath))
            throw new ArgumentException("firstTilePath must be provided", nameof(firstTilePath));

        var dir   = Path.GetDirectoryName(firstTilePath) ?? Environment.CurrentDirectory;
        var name  = Path.GetFileNameWithoutExtension(firstTilePath);
        var parts = name.Split('_');

        // Filenames are expected to follow &lt;prefix&gt;_XX_YY.pm4 where XX/YY are 00-63 tile coords.
        if (parts.Length < 3)
        {
            ConsoleLogger.WriteLine("[Pm4Adapter] Filename does not follow *_XX_YY pattern; loading single tile only");
            return Load(firstTilePath, options);
        }

        var prefix         = string.Join("_", parts.Take(parts.Length - 2));
        var candidateFiles = Directory.EnumerateFiles(dir, $"{prefix}_*.pm4");

        var scenes = new List<Pm4Scene>();
        foreach (var file in candidateFiles)
        {
            try
            {
                scenes.Add(Load(file, options));
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"[Pm4Adapter] Failed to load tile {file}: {ex.Message}");
            }
        }

        if (scenes.Count == 0)
            throw new InvalidOperationException("No PM4 tiles loaded for region.");

        // Merge scenes – start with the first tile as the accumulator.
        var baseScene = scenes[0];
        for (int i = 1; i < scenes.Count; i++)
        {
            var s       = scenes[i];
            int vOffset = baseScene.Vertices.Count;

            baseScene.Vertices.AddRange(s.Vertices);
            baseScene.Indices.AddRange(s.Indices.Select(idx => idx + vOffset));
            baseScene.Triangles.AddRange(s.Triangles.Select(t => (t.Item1 + vOffset, t.Item2 + vOffset, t.Item3 + vOffset)));

            baseScene.Surfaces.AddRange(s.Surfaces);
            baseScene.Links.AddRange(s.Links);
            baseScene.Placements.AddRange(s.Placements);
            baseScene.Properties.AddRange(s.Properties);
            baseScene.ExtraChunks.AddRange(s.ExtraChunks);
        }

        AttachMscn(baseScene);
        ConsoleLogger.WriteLine($"[Pm4Adapter] Region loader merged {scenes.Count} tiles. Vertices={baseScene.Vertices.Count}, Indices={baseScene.Indices.Count}");
        return baseScene;
    }

    /// <summary>
    /// Applies the <see cref="MscnRemapper"/> to resolve any remaining cross-tile vertex
    /// references after merging tiles.
    /// </summary>
    private static void AttachMscn(Pm4Scene scene)
    {
        var mscn = scene.ExtraChunks.OfType<MscnChunk>().FirstOrDefault();
        if (mscn != null)
        {
            ConsoleLogger.WriteLine($"[Pm4Adapter] Applying MSCN remap ({mscn.Vertices.Count} exterior vertices)");
            MscnRemapper.Apply(scene, mscn);
        }
    }
}
