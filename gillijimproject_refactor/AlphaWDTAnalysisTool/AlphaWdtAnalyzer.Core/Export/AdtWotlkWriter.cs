using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtWotlkWriter
{
    private static readonly HashSet<string> InitializedCsv = new(StringComparer.OrdinalIgnoreCase);

    public sealed class WriteContext
    {
        public required string ExportDir { get; init; }
        public required string MapName { get; init; }
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required IEnumerable<PlacementRecord> Placements { get; init; }
        public required AssetFixupPolicy Fixup { get; init; }
        public bool ConvertToMh2o { get; init; }
        public AreaIdMapper? AreaMapper { get; init; }
        public IReadOnlyList<int>? AlphaAreaIds { get; init; }
    }

    public static void WritePlaceholder(WriteContext ctx)
    {
        Directory.CreateDirectory(ctx.ExportDir);
        var placeholderDir = Path.Combine(ctx.ExportDir, "placeholders", ctx.MapName);
        Directory.CreateDirectory(placeholderDir);
        var file = Path.Combine(placeholderDir, $"{ctx.MapName}_{ctx.TileX}_{ctx.TileY}.adt.placeholder.txt");
        using var sw = new StreamWriter(file);
        sw.WriteLine($"Map={ctx.MapName} Tile=({ctx.TileX},{ctx.TileY})");
        sw.WriteLine($"ConvertToMh2o={ctx.ConvertToMh2o}");

        // Area ID mapping summary and CSV (emit whenever we have AlphaAreaIds)
        if (ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
        {
            int mapped = 0, unmapped = 0, present = 0;
            for (int i = 0; i < 256; i++)
            {
                var aId = ctx.AlphaAreaIds[i];
                if (aId < 0) continue;
                present++;
                bool rowMapped = false;
                if (ctx.AreaMapper is not null)
                {
                    if (ctx.AreaMapper.TryMap(aId, out _, out _, out _))
                    {
                        rowMapped = true;
                    }
                }
                if (rowMapped) mapped++; else unmapped++;
            }
            sw.WriteLine($"AreaIds: present={present} mapped={mapped} unmapped={unmapped}");

            // Emit CSV rows per MCNK under <ExportDir>/csv/<MapName>/
            var csvDir = Path.Combine(ctx.ExportDir, "csv", ctx.MapName);
            Directory.CreateDirectory(csvDir);
            var csvPath = Path.Combine(csvDir, "areaid_mapping.csv");

            // Truncate once per process run to avoid mixing with previous runs
            bool firstForThisRun = !InitializedCsv.Contains(csvPath);
            if (firstForThisRun) InitializedCsv.Add(csvPath);
            using var cw = new StreamWriter(csvPath, append: !firstForThisRun);
            if (firstForThisRun)
            {
                cw.WriteLine("tile_x,tile_y,mcnk_index,alpha_area_id,alpha_name,lk_area_id,lk_name,mapped,mapping_reason");
            }
            for (int i = 0; i < 256; i++)
            {
                var aId = ctx.AlphaAreaIds[i];
                if (aId < 0) continue;

                bool mappedRow = false;
                int lkId = 0;
                string? aName = ctx.AreaMapper?.GetAlphaName(aId);
                string? lkName = null;
                string reason = string.Empty;

                if (ctx.AreaMapper is not null)
                {
                    if (ctx.AreaMapper.TryMap(aId, out lkId, out var aName2, out var lkName2))
                    {
                        mappedRow = true;
                        reason = "name";
                        if (string.IsNullOrEmpty(aName)) aName = aName2;
                        lkName = lkName2;
                    }
                }

                cw.WriteLine($"{ctx.TileX},{ctx.TileY},{i},{aId},{EscapeCsv(aName)},{(mappedRow ? lkId.ToString() : string.Empty)},{EscapeCsv(lkName)},{mappedRow},{reason}");
            }
        }

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

    private static string EscapeCsv(string? s)
    {
        if (string.IsNullOrEmpty(s)) return string.Empty;
        if (s.Contains(',') || s.Contains('"'))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }
}
