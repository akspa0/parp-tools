using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
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

        // Build Alpha ADT handle once (also used to enumerate MTEX textures)
        var alpha = new AdtAlpha(ctx.WdtPath, ctx.AdtOffset, ctx.AdtNumber);

        // Before conversion, record any placements or textures we could not resolve at all
        var missingPath = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName, "missing_assets.csv");
        Directory.CreateDirectory(Path.GetDirectoryName(missingPath)!);
        using (var missing = new MissingAssetsLogger(missingPath))
        {
            // placements (WMO/M2)
            foreach (var p in ctx.Placements)
            {
                var _ = ctx.Fixup.ResolveWithMethod(p.Type, p.AssetPath, out var method);
                if (string.Equals(method, "preserve_missing", StringComparison.OrdinalIgnoreCase))
                {
                    missing.Write(new MissingAssetRecord
                    {
                        Type = p.Type.ToString(),
                        Original = p.AssetPath,
                        MapName = p.MapName,
                        TileX = p.TileX,
                        TileY = p.TileY,
                        UniqueId = p.UniqueId
                    });
                }
            }

            // textures (BLP via MTEX)
            foreach (var tex in alpha.GetMtexTextureNames())
            {
                var norm = ListfileLoader.NormalizePath(tex);
                if (string.IsNullOrWhiteSpace(norm)) continue;
                var _ = ctx.Fixup.ResolveTextureWithMethod(norm, out var method);
                if (string.Equals(method, "preserve_missing", StringComparison.OrdinalIgnoreCase))
                {
                    missing.Write(new MissingAssetRecord
                    {
                        Type = AssetType.Blp.ToString(),
                        Original = norm,
                        MapName = ctx.MapName,
                        TileX = ctx.TileX,
                        TileY = ctx.TileY,
                        UniqueId = null
                    });
                }
            }
        }

        // Build LK ADT from Alpha using WDT MDNM/MONM tables
        var fixedM2 = ctx.MdnmFiles.Select(n => ctx.Fixup.Resolve(AssetType.MdxOrM2, n)).ToList();
        var fixedWmo = ctx.MonmFiles.Select(n => ctx.Fixup.Resolve(AssetType.Wmo, n)).ToList();
        var adtLk = alpha.ToAdtLk(fixedM2, fixedWmo);
        adtLk.ToFile(outFile);

        // Patch MTEX in-place with capacity-aware replacements (do not change file size)
        try
        {
            PatchMtexOnDisk(outFile, ctx.Fixup);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[MTEX] Failed to patch MTEX for {outFile}: {ex.Message}");
        }

        // Patch MMDX (M2/MDX) and MWMO (WMO) name tables in-place
        try
        {
            PatchStringTableInPlace(outFile, "MMDX", AssetType.MdxOrM2, ctx.Fixup, (orig) => ctx.Fixup.ResolveWithMethod(AssetType.MdxOrM2, orig, out _));
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[MMDX] Failed to patch M2/MDX names for {outFile}: {ex.Message}");
        }
        try
        {
            PatchStringTableInPlace(outFile, "MWMO", AssetType.Wmo, ctx.Fixup, (orig) => ctx.Fixup.ResolveWithMethod(AssetType.Wmo, orig, out _));
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[MWMO] Failed to patch WMO names for {outFile}: {ex.Message}");
        }

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

    private static void PatchMtexOnDisk(string filePath, AssetFixupPolicy fixup)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        // Locate MTEX chunk by scanning top-level chunks
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mtexDataPos = -1;
        int mtexSize = 0;

        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int size = br.ReadInt32();
            long dataPos = fs.Position;
            if (fourcc == "MTEX")
            {
                mtexDataPos = dataPos;
                mtexSize = size;
                break;
            }
            // skip data + pad
            fs.Position = dataPos + size + ((size & 1) == 1 ? 1 : 0);
        }

        if (mtexDataPos < 0 || mtexSize <= 0) return; // no textures

        // Read MTEX data
        fs.Position = mtexDataPos;
        var data = br.ReadBytes(mtexSize);

        // Parse null-terminated strings and patch in-place if replacement fits
        int i = 0;
        while (i < data.Length)
        {
            int start = i;
            // find terminator
            while (i < data.Length && data[i] != 0) i++;
            int end = i; // points at 0 or data.Length
            int capacity = end - start; // bytes available before 0

            if (capacity > 0)
            {
                var original = Encoding.ASCII.GetString(data, start, capacity);
                var norm = ListfileLoader.NormalizePath(original);
                var resolved = fixup.ResolveTextureWithMethod(norm, out var method);

                // Enforce capacity: try resolved; if too long, try fallbacks; else skip
                ReadOnlySpan<byte> toWrite = Encoding.ASCII.GetBytes(resolved);
                string decision = "resolved";
                if (toWrite.Length > capacity)
                {
                    // tileset fallback
                    var tf = fixup.TilesetFallbackPath;
                    if (!string.IsNullOrWhiteSpace(tf) && fixup.ExistsPath(tf))
                    {
                        var tfBytes = Encoding.ASCII.GetBytes(tf);
                        if (tfBytes.Length <= capacity) { toWrite = tfBytes; decision = "capacity_fallback:tileset"; }
                    }
                }
                if (toWrite.Length > capacity)
                {
                    // non-tileset fallback
                    var nf = fixup.NonTilesetFallbackPath;
                    if (!string.IsNullOrWhiteSpace(nf) && fixup.ExistsPath(nf))
                    {
                        var nfBytes = Encoding.ASCII.GetBytes(nf);
                        if (nfBytes.Length <= capacity) { toWrite = nfBytes; decision = "capacity_fallback:non_tileset"; }
                    }
                }

                if (toWrite.Length <= capacity && !original.Equals(Encoding.ASCII.GetString(toWrite), StringComparison.OrdinalIgnoreCase))
                {
                    Array.Copy(toWrite.ToArray(), 0, data, start, toWrite.Length);
                    // pad remaining to zero
                    for (int k = start + toWrite.Length; k < end; k++) data[k] = 0;
                    if (decision.StartsWith("capacity_fallback", StringComparison.OrdinalIgnoreCase))
                    {
                        fixup.LogDiagnostic(AssetType.Blp, norm, Encoding.ASCII.GetString(toWrite), decision);
                    }
                }
                else
                {
                    // overflow or unchanged; leave as-is
                    if (toWrite.Length > capacity)
                    {
                        fixup.LogDiagnostic(AssetType.Blp, norm, resolved, "overflow_skip:mtex");
                    }
                }
            }

            // move past terminator
            i = end + 1;
        }

        // Write back patched MTEX payload
        fs.Position = mtexDataPos;
        bw.Write(data);
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
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int size2 = br.ReadInt32();
            long dpos2 = fs.Position;
            if (fourcc == "MCIN")
            {
                mcinDataPos = dpos2;
                mcinSize = size2;
                break;
            }
            fs.Position = dpos2 + size2 + ((size2 & 1) == 1 ? 1 : 0);
        }
        if (mcinDataPos >= 0 && mcinSize >= 16)
        {
            fs.Position = mcinDataPos;
            long end2 = mcinDataPos + mcinSize;
            while (fs.Position + 16 <= end2)
            {
                long entryPos = fs.Position;
                int textureIdOld = br.ReadInt32();
                // skip other fields (flags, ofsMcal, effectId)
                fs.Position = entryPos; // rewind to start
                // MTEX not moved; textureId offsets remain valid
                fs.Position = entryPos + 16; // next entry
            }
        }

        for (int i2 = 0; i2 < 256; i2++)
        {
            int aId = alphaAreaIds[i2];
            if (aId < 0) continue; // no MCNK present

            fs.Position = mcinDataPos + (i2 * 16);
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
        if (string.IsNullOrEmpty(s) || s.Length != 4) return s ?? string.Empty;
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

    private static void PatchStringTableInPlace(string filePath, string chunkFourCC, AssetType type, AssetFixupPolicy fixup, Func<string, string> resolve)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        // Locate the string table chunk by scanning top-level chunks
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long dataPos = -1;
        int size = 0;
        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == chunkFourCC)
            {
                dataPos = dpos;
                size = sz;
                break;
            }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }
        if (dataPos < 0 || size <= 0) return;

        // Read chunk payload
        fs.Position = dataPos;
        var data = br.ReadBytes(size);

        // Iterate null-terminated strings and patch when replacement fits
        int i = 0;
        while (i < data.Length)
        {
            int start = i;
            while (i < data.Length && data[i] != 0) i++;
            int end = i;
            int capacity = end - start;
            if (capacity > 0)
            {
                var original = Encoding.ASCII.GetString(data, start, capacity);
                var norm = ListfileLoader.NormalizePath(original);
                var resolved = resolve(norm);
                var bytes = Encoding.ASCII.GetBytes(resolved);
                if (bytes.Length <= capacity && !norm.Equals(resolved, StringComparison.OrdinalIgnoreCase))
                {
                    Array.Copy(bytes, 0, data, start, bytes.Length);
                    for (int k = start + bytes.Length; k < end; k++) data[k] = 0;
                }
                else if (bytes.Length > capacity)
                {
                    fixup.LogDiagnostic(type, norm, resolved, "overflow_skip:" + chunkFourCC.ToLowerInvariant());
                }
            }
            i = end + 1;
        }

        fs.Position = dataPos;
        bw.Write(data);
    }
}
