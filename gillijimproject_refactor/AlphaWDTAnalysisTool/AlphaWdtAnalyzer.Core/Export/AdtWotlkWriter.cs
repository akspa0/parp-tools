using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtWotlkWriter
{
    public sealed class WriteContext
    {
        public required string ExportDir { get; init; }
        public required string MapName { get; init; }
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required IEnumerable<PlacementRecord> Placements { get; init; }
        public required AssetFixupPolicy Fixup { get; init; }
        public bool ConvertToMh2o { get; init; }
    }

    public static void WritePlaceholder(WriteContext ctx)
    {
        Directory.CreateDirectory(ctx.ExportDir);
        var file = Path.Combine(ctx.ExportDir, $"{ctx.MapName}_{ctx.TileX}_{ctx.TileY}.adt.placeholder.txt");
        using var sw = new StreamWriter(file);
        sw.WriteLine($"Map={ctx.MapName} Tile=({ctx.TileX},{ctx.TileY})");
        sw.WriteLine($"ConvertToMh2o={ctx.ConvertToMh2o}");
        sw.WriteLine("Placements (with potential fixups):");
        foreach (var p in ctx.Placements.OrderBy(p => p.Type).ThenBy(p => p.AssetPath))
        {
            var type = p.Type;
            var path = p.AssetPath;
            var fixedPath = type switch
            {
                AssetType.Wmo => ctx.Fixup.Resolve(AssetType.Wmo, path),
                AssetType.MdxOrM2 => ctx.Fixup.Resolve(AssetType.MdxOrM2, path),
                _ => path
            };
            var flag = (fixedPath.Equals(path, StringComparison.OrdinalIgnoreCase)) ? "ok" : $"fixed -> {fixedPath}";
            sw.WriteLine($"  {type}: {path} [{flag}] UniqueId={p.UniqueId?.ToString() ?? ""}");
        }
        sw.WriteLine();
        sw.WriteLine("NOTE: This is a placeholder. Binary WotLK ADT writing will be implemented next.");
    }
}
