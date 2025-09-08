using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.WowFiles.Alpha;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtWotlkWriter
{
    private static readonly HashSet<string> InitializedCsv = new(StringComparer.OrdinalIgnoreCase);
    private static readonly HashSet<string> EmittedWdtForMap = new(StringComparer.OrdinalIgnoreCase);

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
        public required string WdtPath { get; init; }
        public required int AdtNumber { get; init; }
        public required int AdtOffset { get; init; }
        public required IReadOnlyList<string> MdnmFiles { get; init; }
        public required IReadOnlyList<string> MonmFiles { get; init; }
    }

    public static void WritePlaceholder(WriteContext ctx)
    {
        Directory.CreateDirectory(ctx.ExportDir);
        var file = Path.Combine(ctx.ExportDir, $"{ctx.MapName}_{ctx.TileX}_{ctx.TileY}.adt.placeholder.txt");
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

            // Emit CSV rows per MCNK
            var csvDir = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName);
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

    public static void WriteBinary(WriteContext ctx)
    {
        // Output to World/Maps/<map>/
        var mapsDir = Path.Combine(ctx.ExportDir, "World", "Maps", ctx.MapName);
        Directory.CreateDirectory(mapsDir);
        var outFile = Path.Combine(mapsDir, $"{ctx.MapName}_{ctx.TileX}_{ctx.TileY}.adt");

        // Build LK ADT from Alpha using WDT MDNM/MONM tables
        var alpha = new AdtAlpha(ctx.WdtPath, ctx.AdtOffset, ctx.AdtNumber);
        var fixedM2 = ctx.MdnmFiles.Select(n => ctx.Fixup.Resolve(AssetType.MdxOrM2, n)).ToList();
        var fixedWmo = ctx.MonmFiles.Select(n => ctx.Fixup.Resolve(AssetType.Wmo, n)).ToList();
        var adtLk = alpha.ToAdtLk(fixedM2, fixedWmo);
        adtLk.ToFile(outFile);

        // Emit/refresh WDT once per map in the same folder, rename *_new to <map>.wdt
        if (!EmittedWdtForMap.Contains(ctx.MapName))
        {
            try
            {
                var wdtAlpha = new WdtAlpha(ctx.WdtPath);
                var wdt = wdtAlpha.ToWdt();
                wdt.ToFile(mapsDir); // writes <basename>.wdt_new
                var newFile = Path.Combine(mapsDir, Path.GetFileName(ctx.WdtPath) + "_new");
                var finalFile = Path.Combine(mapsDir, ctx.MapName + ".wdt");
                if (File.Exists(finalFile)) File.Delete(finalFile);
                if (File.Exists(newFile)) File.Move(newFile, finalFile, overwrite: true);
                EmittedWdtForMap.Add(ctx.MapName);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[WDT] Failed to emit WDT for {ctx.MapName}: {ex.Message}");
            }
        }

        // Emit AreaID CSV (same as placeholder) to preserve analysis output
        if (ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
        {
            var csvDir = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName);
            Directory.CreateDirectory(csvDir);
            var csvPath = Path.Combine(csvDir, "areaid_mapping.csv");

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
                    else
                    {
                        // Keep reason from TryResolveById for diagnostics if we use fallback
                        if (ctx.AreaMapper.TryResolveById(aId, out var lkId2, out var r2))
                        {
                            lkId = lkId2;
                            reason = r2;
                        }
                    }
                }

                cw.WriteLine($"{ctx.TileX},{ctx.TileY},{i},{aId},{EscapeCsv(aName)},{(lkId != 0 ? lkId.ToString() : string.Empty)},{EscapeCsv(lkName)},{mappedRow},{reason}");
            }
        }

        // Patch per-MCNK AreaId in-place using mapper when available
        if (ctx.AreaMapper is not null && ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
        {
            try
            {
                PatchMcnkAreaIdsOnDisk(outFile, ctx.AlphaAreaIds, ctx.AreaMapper);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AreaPatch] Failed to patch AreaIDs for {outFile}: {ex.Message}");
            }
        }
    }

    private static void PatchMcnkAreaIdsOnDisk(string filePath, IReadOnlyList<int> alphaAreaIds, AreaIdMapper mapper)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        // Locate MCIN chunk by scanning top-level chunks
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mcinDataPos = -1;
        int mcinSize = 0;

        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(System.Text.Encoding.ASCII.GetString(fourccRevBytes));
            int size = br.ReadInt32();
            long dataPos = fs.Position;
            if (fourcc == "MCIN")
            {
                mcinDataPos = dataPos;
                mcinSize = size;
                break;
            }
            // skip data + pad
            fs.Position = dataPos + size + ((size & 1) == 1 ? 1 : 0);
        }

        if (mcinDataPos < 0 || mcinSize < (256 * 16)) return;

        for (int i = 0; i < 256; i++)
        {
            int aId = alphaAreaIds[i];
            if (aId < 0) continue; // no MCNK present

            fs.Position = mcinDataPos + (i * 16);
            int mcnkOffset = br.ReadInt32();
            // skip size and unused
            if (mcnkOffset <= 0) continue;

            // Decide LK AreaID
            int lkAreaId;
            string reason;
            string? alphaName;
            string? lkName;
            if (mapper.TryMap(aId, out lkAreaId, out alphaName, out lkName))
            {
                // ok
            }
            else if (!mapper.TryResolveById(aId, out lkAreaId, out reason))
            {
                // leave as-is if we can't resolve
                continue;
            }

            long areaFieldPos = (long)mcnkOffset + 8 + 0x34; // MCNK letters+size + AreaId offset in header
            if (areaFieldPos + 4 > fileLen) continue;

            long save = fs.Position;
            fs.Position = areaFieldPos;
            bw.Write(lkAreaId);
            fs.Position = save;
        }
    }

    private static string ReverseFourCC(string s)
    {
        if (s is null || s.Length != 4) return s;
        return new string(new[] { s[3], s[2], s[1], s[0] });
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
