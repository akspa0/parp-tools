using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using DBCTool.V2.Core;
using GillijimProject.WowFiles.Alpha;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtWotlkWriter
{
    private static readonly HashSet<string> InitializedCsv = new(StringComparer.OrdinalIgnoreCase);
    private static readonly HashSet<string> EmittedWdtForMap = new(StringComparer.OrdinalIgnoreCase);
    // LK DBC caches (AreaID -> MapID/Name) loaded on demand when LkDbcDir is provided
    private static readonly object s_lkCacheLock = new();
    private static string? s_lkDbcDir;
    private static Dictionary<int, int>? s_lkMapByAreaId;
    private static Dictionary<int, string>? s_lkNameByAreaId;

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
        public AreaIdMapperV2? AreaMapperV2 { get; init; }
        public IReadOnlyList<int>? AlphaAreaIds { get; init; }
        public DbcPatchMapping? PatchMapping { get; init; }
        public required string WdtPath { get; init; }
        public required int AdtNumber { get; init; }
        public required int AdtOffset { get; init; }
        public required IReadOnlyList<string> MdnmFiles { get; init; }
        public required IReadOnlyList<string> MonmFiles { get; init; }
        public bool Verbose { get; init; } = false;
        public bool TrackAssets { get; init; } = false;
        public int? CurrentMapId { get; init; }
        // Visualization
        public bool VizSvg { get; init; } = false;
        public string? VizDir { get; init; }
        public string? LkDbcDir { get; init; }
        public bool VizHtml { get; init; } = false;
        public bool PatchOnly { get; init; } = false;
        public bool NoZoneFallback { get; init; } = false;
    }

    private static string BeautifyToken(string token)
    {
        var cleaned = (token ?? string.Empty).Replace('_', ' ').Replace('-', ' ').Trim();
        if (cleaned.Length == 0) return string.Empty;
        return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(cleaned);
    }

    private static string FormatChain(string? chain)
    {
        if (string.IsNullOrWhiteSpace(chain)) return string.Empty;
        var tokens = chain.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(BeautifyToken)
            .Where(t => !string.IsNullOrWhiteSpace(t));
        return string.Join(" / ", tokens);
    }

    private static string DescribeMap(int? mapId)
    {
        if (!mapId.HasValue || mapId.Value < 0) return "unknown";
        var name = ResolveTargetMapNameFromId(mapId.Value);
        return name is null ? mapId.Value.ToString(CultureInfo.InvariantCulture) : $"{mapId.Value} ({name})";
    }

    private static bool TryGetTargetMapId(int areaId, out int mapId)
    {
        mapId = -1;
        var cache = s_lkMapByAreaId;
        if (cache is null) return false;
        return cache.TryGetValue(areaId, out mapId);
    }

    private static string DescribeTarget(int areaId, DbcPatchMapping? patchMap)
    {
        if (areaId <= 0)
        {
            return areaId == 0 ? "0 (unassigned)" : areaId.ToString(CultureInfo.InvariantCulture);
        }

        EnsureLkCache();
        string name = string.Empty;
        if (patchMap is not null && patchMap.TryGetTargetName(areaId, out var nm) && !string.IsNullOrWhiteSpace(nm))
        {
            name = nm;
        }
        else if (s_lkNameByAreaId is not null && s_lkNameByAreaId.TryGetValue(areaId, out var lkName) && !string.IsNullOrWhiteSpace(lkName))
        {
            name = lkName;
        }

        string mapDisplay = string.Empty;
        if (TryGetTargetMapId(areaId, out var mapId) && mapId >= 0)
        {
            mapDisplay = DescribeMap(mapId);
        }

        var pieces = new List<string> { areaId.ToString(CultureInfo.InvariantCulture) };
        if (!string.IsNullOrWhiteSpace(name)) pieces.Add(name);
        if (!string.IsNullOrWhiteSpace(mapDisplay)) pieces.Add(mapDisplay);
        return string.Join(" | ", pieces);
    }

    private static string BuildSourceDisplay(string mapName, string midChain, int zoneBase, int subLo, int alphaId)
    {
        var pieces = new List<string>();
        if (!string.IsNullOrWhiteSpace(mapName)) pieces.Add(mapName.Trim());
        var chain = FormatChain(midChain);
        if (!string.IsNullOrWhiteSpace(chain)) pieces.Add(chain);
        pieces.Add($"alpha=0x{alphaId:X8}");
        pieces.Add($"zone=0x{zoneBase:X8} sub=0x{subLo:X4}");
        return string.Join(" | ", pieces);
    }

    // Resolve the human-friendly LK continent name used in tgt_mapName_xwalk columns
    // This differs from Map.dbc directory names (e.g., 0 -> "Eastern Kingdoms", not "Azeroth").
    private static string? ResolveTargetMapNameFromId(int? mapId)
    {
        if (!mapId.HasValue || mapId.Value < 0) return null;
        return mapId.Value switch
        {
            0 => "Eastern Kingdoms",
            1 => "Kalimdor",
            530 => "Outland",
            571 => "Northrend",
            _ => null,
        };
    }

    private static bool ValidateTargetMap(int lkAreaId, int? expectedMapId)
    {
        if (lkAreaId <= 0) return true; // allow 0
        if (!expectedMapId.HasValue || expectedMapId.Value < 0) return true; // no guard
        var cache = s_lkMapByAreaId;
        if (cache is null) return true; // no DBC
        return cache.TryGetValue(lkAreaId, out var map) ? map == expectedMapId.Value : true;
    }

    private static void EnsureLkCache()
    {
        if (s_lkMapByAreaId is not null && s_lkNameByAreaId is not null) return;
        var dir = s_lkDbcDir;
        if (string.IsNullOrWhiteSpace(dir)) return;
        lock (s_lkCacheLock)
        {
            if (s_lkMapByAreaId is not null && s_lkNameByAreaId is not null) return;
            try
            {
                var areaDbc = Path.Combine(dir, "AreaTable.dbc");
                if (!File.Exists(areaDbc)) return;
                // Minimal raw DBC reader for AreaTable: WDBC/WDB2 header, then records, then string block.
                using var fs = File.OpenRead(areaDbc);
                using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);
                var magicBytes = br.ReadBytes(4);
                if (magicBytes.Length != 4) return;
                int recordCount = br.ReadInt32();
                int fieldCount = br.ReadInt32();
                int recordSize  = br.ReadInt32();
                int stringBlockSize = br.ReadInt32();
                var recordsData = br.ReadBytes(recordCount * recordSize);
                var stringBlock = br.ReadBytes(stringBlockSize);
                var mapById = new Dictionary<int, int>();
                var nameById = new Dictionary<int, string>();
                for (int i = 0; i < recordCount; i++)
                {
                    int baseOff = i * recordSize;
                    // Read ints
                    var ints = new int[fieldCount];
                    for (int f = 0; f < fieldCount; f++)
                    {
                        int off = baseOff + (f * 4);
                        if (off + 4 <= recordsData.Length) ints[f] = BitConverter.ToInt32(recordsData, off);
                    }
                    int id = (fieldCount > 0) ? ints[0] : 0;
                    if (id <= 0) continue;
                    int contId = (fieldCount > 1) ? ints[1] : 0; // heuristic: ContinentID
                    // Find first string in row
                    string name = string.Empty;
                    for (int f = 0; f < fieldCount; f++)
                    {
                        int off = ints[f];
                        if (off > 0 && off < stringBlock.Length)
                        {
                            // read cstring
                            int end = off;
                            while (end < stringBlock.Length && stringBlock[end] != 0) end++;
                            if (end > off)
                            {
                                name = Encoding.UTF8.GetString(stringBlock, off, end - off);
                                break;
                            }
                        }
                    }
                    mapById[id] = contId;
                    if (!nameById.ContainsKey(id)) nameById[id] = name;
                }
                s_lkMapByAreaId = mapById;
                s_lkNameByAreaId = nameById;
            }
            catch { /* best-effort */ }
        }
    }

    private static (int present, int patched) PatchMcnkAreaIdsOnDiskV2(string filePath, string mapName, IReadOnlyList<int> alphaAreaIds, AreaIdMapperV2? mapper, bool verbose, int? currentMapId, AreaIdMapper? legacyMapper, DbcPatchMapping? patchMap, bool patchOnly, bool noZoneFallback)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read, bufferSize: 65536, options: FileOptions.RandomAccess);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        // Locate MCIN to get MCNK offsets
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mcinDataPos = -1;
        int mcinSize = 0;

        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == "MCIN") { mcinDataPos = dpos; mcinSize = sz; break; }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }

        int present = 0, patched = 0;
        int debugPrinted = 0;
        List<string>? verboseLog = verbose ? new List<string>() : null;

        static string NormalizeNameToken(string value)
        {
            return string.IsNullOrWhiteSpace(value) ? string.Empty : value.Trim().ToLowerInvariant();
        }
        // Pre-read MCIN offsets to avoid 256 seeks per ADT
        byte[] mcinBytes = Array.Empty<byte>();
        if (mcinDataPos >= 0 && mcinSize >= 16)
        {
            fs.Position = mcinDataPos;
            int need = Math.Min(mcinSize, 256 * 16);
            mcinBytes = br.ReadBytes(need);
        }

        if (mcinDataPos >= 0 && mcinSize >= 16)
        {
            for (int i2 = 0; i2 < 256; i2++)
            {
                int mcnkOffset;
                if (mcinBytes.Length >= (i2 + 1) * 16)
                    mcnkOffset = BitConverter.ToInt32(mcinBytes, i2 * 16);
                else
                {
                    fs.Position = mcinDataPos + (i2 * 16);
                    mcnkOffset = br.ReadInt32();
                }
                if (mcnkOffset <= 0) continue;
                present++;

                int contRaw = currentMapId ?? -1;
                int lkAreaId = 0;
                string method = "fallback0";
                // Use AlphaAreaIds captured from the Alpha ADT (zone<<16|sub). This is authoritative.
                int aIdNum = -1;
                if (alphaAreaIds is not null && alphaAreaIds.Count == 256)
                {
                    int alt = alphaAreaIds[i2];
                    if (alt > 0)
                    {
                        aIdNum = ((alt >> 16) == 0) ? (alt << 16) : alt;
                    }
                }

                // Try strategies in order (prefer precise target-map-locked mapping first)
                bool mapped = false;
                int zoneBase = aIdNum & unchecked((int)0xFFFF0000);
                int subLo = aIdNum & 0xFFFF;

                int midAreaHint = 0;
                int midMapHint = -1;
                int midParentHint = -1;
                string midChainHint = string.Empty;
                bool midViaHint = false;
                bool hasMidInfo = false;
                if (patchMap is not null && aIdNum > 0)
                {
                    hasMidInfo = patchMap.TryGetMidInfo(currentMapId, aIdNum, out midAreaHint, out midMapHint, out midParentHint, out midChainHint, out midViaHint);
                }

                bool hasZoneCandidate = false;
                int zoneCandidateId = 0;
                bool zoneCandidateVia = false;
                if (patchMap is not null && aIdNum > 0 && zoneBase != 0)
                {
                    hasZoneCandidate = patchMap.TryMapZone(zoneBase, currentMapId, out zoneCandidateId, out zoneCandidateVia);
                }

                if (verbose && patchMap is not null && aIdNum > 0 && subLo > 0 && debugPrinted < 24)
                {
                    var zoneDiagBase = zoneBase;
                    var subDiag = subLo;
                    bool mapScoped = currentMapId.HasValue && currentMapId.Value >= 0;
                    bool anyMatch = patchMap.TryMapSubZone(zoneDiagBase, subDiag, null, out var anyId, out var anyVia);
                    bool mapMatch = false;
                    int mapMatchId = 0;
                    bool mapMatchVia = false;
                    if (mapScoped)
                    {
                        mapMatch = patchMap.TryMapSubZone(zoneDiagBase, subDiag, currentMapId, out mapMatchId, out mapMatchVia);
                    }
                    Console.WriteLine($"  [Diag] zoneBase=0x{zoneDiagBase:X} subLo=0x{subDiag:X} mapId={currentMapId} anyMatch={anyMatch} anyId={anyId} anyVia={anyVia} mapMatch={mapMatch} mapMatchId={mapMatchId} mapMatchVia={mapMatchVia}");
                }

                // -1) CSV lookups keyed by hi/lo pairs and per-map area numbers
                var mapRejects = verbose ? new List<string>() : null;

                bool AcceptCandidate(int candidateId, string candidateMethod)
                {
                    if (candidateId <= 0) return false;
                    if (currentMapId.HasValue && currentMapId.Value >= 0 && !ValidateTargetMap(candidateId, currentMapId))
                    {
                        var expected = DescribeMap(currentMapId);
                        mapRejects?.Add($"    [map_mismatch] candidate={DescribeTarget(candidateId, patchMap)} method={candidateMethod} expectedMap={expected}");
                        return false;
                    }
                    lkAreaId = candidateId;
                    method = candidateMethod;
                    mapped = true;
                    return true;
                }

                if (!mapped && patchMap is not null && aIdNum > 0)
                {
                    if (subLo > 0 && patchMap.TryMapSubZone(zoneBase, subLo, currentMapId, out var csvSubId, out var viaSub))
                    {
                        AcceptCandidate(csvSubId, viaSub ? "patch_csv_sub_via060" : "patch_csv_sub");
                    }
                    else if (patchMap.TryMapBySrcAreaSimple(mapName, aIdNum, out var csvByName))
                    {
                        AcceptCandidate(csvByName, "patch_csv_num");
                    }
                    else if (patchMap.TryMapBySrcAreaNumber(aIdNum, out var csvExactId, out var viaExact))
                    {
                        AcceptCandidate(csvExactId, viaExact ? "patch_csv_exact_via060" : "patch_csv_exact");
                    }
                    else if (patchMap.TryMapViaMid(currentMapId, aIdNum, out var csvMidId, out var midAreaResolved, out var viaMid))
                    {
                        if (AcceptCandidate(csvMidId, viaMid ? "patch_csv_mid_via060" : "patch_csv_mid") && !hasMidInfo && midAreaResolved > 0)
                        {
                            midAreaHint = midAreaResolved;
                        }
                    }
                }

                if (!mapped && hasZoneCandidate && hasMidInfo && patchMap is not null)
                {
                    var childIds = patchMap.GetChildCandidateIds(zoneCandidateId);
                    var childNames = patchMap.GetChildCandidateNames(zoneCandidateId);
                    if (childIds.Count > 0 && childNames.Count > 0)
                    {
                        var tokens = new List<string>();
                        void AddToken(string value)
                        {
                            var norm = NormalizeNameToken(value);
                            if (!string.IsNullOrEmpty(norm) && !tokens.Contains(norm)) tokens.Add(norm);
                        }

                        AddToken(midChainHint);
                        if (!string.IsNullOrWhiteSpace(midChainHint))
                        {
                            var segments = midChainHint.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                            foreach (var seg in segments)
                            {
                                AddToken(seg);
                            }
                        }

                        for (int idx = 0; idx < Math.Min(childIds.Count, childNames.Count); idx++)
                        {
                            var childNorm = NormalizeNameToken(childNames[idx]);
                            if (tokens.Count == 0 || tokens.Contains(childNorm))
                            {
                                if (AcceptCandidate(childIds[idx], (midViaHint || zoneCandidateVia) ? "patch_csv_mid_child_via060" : "patch_csv_mid_child"))
                                {
                                    break;
                                }
                            }
                        }
                    }
                }

                if (!mapped && hasZoneCandidate)
                {
                    AcceptCandidate(zoneCandidateId, zoneCandidateVia ? "patch_csv_zone_via060" : "patch_csv_zone");
                }
                // 0a) CSV numeric direct by target mapId (guarded by CurrentMapId) with via060 preference
                if (!mapped && patchMap is not null && aIdNum > 0 && currentMapId.HasValue && currentMapId.Value >= 0
                    && patchMap.TryMapByTargetViaFirst(currentMapId.Value, aIdNum, out var csvIdNumMap, out var mapVia))
                {
                    AcceptCandidate(csvIdNumMap, mapVia ? "patch_csv_num_mapX_via060" : "patch_csv_num_mapX");
                }
                // 0b) CSV numeric direct by target mapName (strict map-locked by name) with via060 preference
                if (!mapped && patchMap is not null && aIdNum > 0)
                {
                    var tgtName = ResolveTargetMapNameFromId(currentMapId);
                    if (!string.IsNullOrWhiteSpace(tgtName)
                        && patchMap.TryMapByTargetNameViaFirst(tgtName!, aIdNum, out var csvIdNumMapName, out var nameVia))
                    {
                        AcceptCandidate(csvIdNumMapName, nameVia ? "patch_csv_num_mapNameX_via060" : "patch_csv_num_mapNameX");
                    }
                }
                // 0c) (reserved) -- handled in -1 block above
                // Strict mode: no other fallbacks
                if (!mapped) { lkAreaId = 0; method = "unmapped"; }

                long areaFieldPos = (long)mcnkOffset + 8 + 0x34; // LK MCNK header AreaId
                if (areaFieldPos + 4 > fileLen) continue;

                long save = fs.Position;
                fs.Position = areaFieldPos;
                uint existing = br.ReadUInt32();

                bool hasMappedTarget = mapped && lkAreaId > 0;
                int effectiveWrite = hasMappedTarget ? lkAreaId : (int)existing;
                string methodLogged = hasMappedTarget ? method : (method == "unmapped" ? "unmapped_preserve" : method);

                string mapDisplay = DescribeMap(currentMapId);
                string sourceDisplay = BuildSourceDisplay(mapName, midChainHint, zoneBase, subLo, aIdNum);
                string existingDisplay = DescribeTarget((int)existing, patchMap);
                string writeDisplay = hasMappedTarget ? DescribeTarget(lkAreaId, patchMap) : existingDisplay;
                string logEntry = $"  [AreaMap] map={mapDisplay} idx={i2:D3} src={sourceDisplay} existing={existingDisplay} -> write={writeDisplay} method={methodLogged}";
                verboseLog?.Add(logEntry);

                if (hasMappedTarget && existing != (uint)lkAreaId)
                {
                    fs.Position = areaFieldPos;
                    bw.Write((uint)lkAreaId);
                    patched++;
                }

                if (verbose && mapRejects is not null && mapRejects.Count > 0 && debugPrinted < 8)
                {
                    foreach (var reject in mapRejects)
                    {
                        Console.WriteLine(reject);
                    }
                }

                if (verbose && debugPrinted < 8)
                {
                    Console.WriteLine(logEntry);
                    debugPrinted++;
                }
                fs.Position = save;
            }
        }
        if (verboseLog is not null && verboseLog.Count > 0)
        {
            Console.WriteLine("[V2] AreaID mappings for tile:");
            foreach (var line in verboseLog)
            {
                Console.WriteLine(line);
            }
        }
        return (present, patched);
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
                    if (ctx.AreaMapper.TryMapDetailed(aId, ctx.CurrentMapId, out _, out _, out _, out var reason))
                    {
                        // Count as mapped only when write gate would allow it (explicit remap)
                        if (string.Equals(reason, "remap_explicit", StringComparison.OrdinalIgnoreCase))
                            rowMapped = true;
                    }
                }
                if (rowMapped) mapped++; else unmapped++;
            }
            sw.WriteLine($"AreaIds: present={present} mapped={mapped} unmapped={unmapped}");
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
        if (ctx.TrackAssets)
        {
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

        // Prepare LK DBC cache if provided (for cross-map guard and legend names)
        if (!string.IsNullOrWhiteSpace(ctx.LkDbcDir)) { s_lkDbcDir = ctx.LkDbcDir; EnsureLkCache(); }

        // Patch per-MCNK AreaId in-place when we have any modern mapping source (patch CSVs or V2 tables).
        if (ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256 && (ctx.PatchMapping is not null || ctx.AreaMapperV2 is not null))
        {
            try
            {
                var (present, patched) = PatchMcnkAreaIdsOnDiskV2(outFile, ctx.MapName, ctx.AlphaAreaIds, ctx.AreaMapperV2, ctx.Verbose, ctx.CurrentMapId, ctx.AreaMapper, ctx.PatchMapping, ctx.PatchOnly, ctx.NoZoneFallback);
                if (ctx.Verbose)
                {
                    Console.WriteLine($"[{ctx.MapName} {ctx.TileX},{ctx.TileY}] AreaIds(V2): present={present} patched={patched}");
                    try { WriteAreaVerifyCsv(ctx, outFile); } catch (Exception ex) { Console.Error.WriteLine($"[AreaVerify] Failed for {outFile}: {ex.Message}"); }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AreaPatchV2] Failed to patch AreaIDs for {outFile}: {ex.Message}");
            }
        }
        else if (ctx.AreaMapper is not null && ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
        {
            try
            {
                var (present, patched, ignored) = PatchMcnkAreaIdsOnDisk(outFile, ctx.AlphaAreaIds, ctx.AreaMapper, ctx.Verbose, ctx.CurrentMapId);
                if (ctx.Verbose)
                {
                    Console.WriteLine($"[{ctx.MapName} {ctx.TileX},{ctx.TileY}] AreaIds: present={present} patched={patched} ignored={ignored}");
                    try { WriteAreaVerifyCsv(ctx, outFile); } catch (Exception ex) { Console.Error.WriteLine($"[AreaVerify] Failed for {outFile}: {ex.Message}"); }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AreaPatch] Failed to patch AreaIDs for {outFile}: {ex.Message}");
            }
        }

        // Optional: visualization output (SVG grid + legend CSV)
        if (ctx.VizSvg)
        {
            try { WriteAreaVisualizationSvg(ctx, outFile); }
            catch (Exception ex) { Console.Error.WriteLine($"[Viz] Failed for {outFile}: {ex.Message}"); }
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

    private static (int present, int patched, int ignored) PatchMcnkAreaIdsOnDisk(string filePath, IReadOnlyList<int> alphaAreaIds, AreaIdMapper mapper, bool verbose, int? currentMapId)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read, bufferSize: 65536, options: FileOptions.RandomAccess);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        // Locate MCIN to get MCNK offsets
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mcinDataPos = -1;
        int mcinSize = 0;

        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == "MCIN") { mcinDataPos = dpos; mcinSize = sz; break; }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }

        int present = 0, patched = 0, ignored = 0;
        int debugPrinted = 0;
        // Pre-read MCIN offsets to avoid seeks
        byte[] mcinBytes = Array.Empty<byte>();
        if (mcinDataPos >= 0 && mcinSize >= 16)
        {
            fs.Position = mcinDataPos;
            int need = Math.Min(mcinSize, 256 * 16);
            mcinBytes = br.ReadBytes(need);
        }

        if (mcinDataPos >= 0 && mcinSize >= 16)
        {
            for (int i2 = 0; i2 < 256; i2++)
            {
                int aId = alphaAreaIds[i2];
                if (aId < 0) continue; // no MCNK present
                present++;

                int mcnkOffset;
                if (mcinBytes.Length >= (i2 + 1) * 16)
                    mcnkOffset = BitConverter.ToInt32(mcinBytes, i2 * 16);
                else
                {
                    fs.Position = mcinDataPos + (i2 * 16);
                    mcnkOffset = br.ReadInt32();
                }
                if (mcnkOffset <= 0) continue;

                // Map strictly by explicit remap, with map-awareness
                int lkAreaId;
                string reason;
                if (mapper.TryMapDetailed(aId, currentMapId, out lkAreaId, out _, out _, out reason))
                {
                    if (string.Equals(reason, "ignored", StringComparison.OrdinalIgnoreCase))
                    { ignored++; if (verbose && debugPrinted < 8) { Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) reason=ignored"); debugPrinted++; } continue; }
                    if (!string.Equals(reason, "remap_explicit", StringComparison.OrdinalIgnoreCase))
                    { if (verbose && debugPrinted < 8) { Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) reason={reason} (no write)"); debugPrinted++; } continue; }
                }
                else
                { if (verbose && debugPrinted < 8) { Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) reason=unmapped (no write)"); debugPrinted++; } continue; }

                long areaFieldPos = (long)mcnkOffset + 8 + 0x34; // LK MCNK header AreaId
                if (areaFieldPos + 4 > fileLen) continue;

                long save = fs.Position;
                fs.Position = areaFieldPos;
                uint onDisk = br.ReadUInt32();
                if (onDisk != (uint)lkAreaId)
                {
                    fs.Position = areaFieldPos;
                    bw.Write((uint)lkAreaId); // write full 32-bit LE
                    patched++;
                    if (verbose && debugPrinted < 8)
                    {
                        Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) existing={onDisk} (0x{onDisk:X}) -> write={lkAreaId} (0x{lkAreaId:X})");
                        debugPrinted++;
                    }
                }
                else if (verbose && debugPrinted < 8)
                {
                    Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) unchanged={onDisk} (0x{onDisk:X})");
                    debugPrinted++;
                }
                fs.Position = save;
            }
        }
        return (present, patched, ignored);
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

    // Emit a per-tile verification CSV of AreaIDs written on disk vs mapping (verbose mode only)
    private static void WriteAreaVerifyCsv(WriteContext ctx, string adtPath)
    {
        var verifyPath = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName, $"areaid_verify_{ctx.TileX}_{ctx.TileY}.csv");
        Directory.CreateDirectory(Path.GetDirectoryName(verifyPath)!);
        using var fs = new FileStream(adtPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs);

        // Locate MCIN
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mcinDataPos = -1;
        int mcinSize = 0;
        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == "MCIN") { mcinDataPos = dpos; mcinSize = sz; break; }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }

        using var sw = new StreamWriter(verifyPath, append: false, Encoding.UTF8);
        sw.WriteLine("tile_x,tile_y,chunk_index,alpha_raw,lk_areaid,tgt_parentid,on_disk,reason,lk_name");
        if (mcinDataPos < 0 || mcinSize < 16) return;

        for (int i = 0; i < 256; i++)
        {
            // Read Alpha Unknown3 per chunk for verification
            fs.Position = mcinDataPos + (i * 16);
            int mcnkOffset = br.ReadInt32();
            uint onDisk = 0;
            if (mcnkOffset > 0)
            {
                long areaFieldPos = (long)mcnkOffset + 8 + 0x34; // LK MCNK.AreaId
                if (areaFieldPos + 4 <= fileLen)
                {
                    fs.Position = areaFieldPos;
                    onDisk = br.ReadUInt32();
                }
            }
            // Use the original AlphaAreaIds captured from the source Alpha ADT
            int alphaRaw = -1;
            if (ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
            {
                int alt = ctx.AlphaAreaIds[i];
                if (alt > 0)
                {
                    alphaRaw = ((alt >> 16) == 0) ? (alt << 16) : alt;
                }
            }

            int lkAreaId = -1; string reason = "unmapped"; string lkName = string.Empty; int tgtParent = 0;
            if (alphaRaw >= 0)
            {
                // Strict: numeric CSV mapping in order: mapId-locked, mapName-locked, then per-map src-name
                if (ctx.PatchMapping is not null && ctx.CurrentMapId.HasValue && ctx.CurrentMapId.Value >= 0 && ctx.PatchMapping.TryMapByTarget(ctx.CurrentMapId.Value, alphaRaw, out var csvNumMap))
                {
                    lkAreaId = csvNumMap; reason = "patch_csv_num_mapX";
                }
                else if (ctx.PatchMapping is not null)
                {
                    var tName = ResolveTargetMapNameFromId(ctx.CurrentMapId);
                    if (!string.IsNullOrWhiteSpace(tName) && ctx.PatchMapping.TryMapByTargetName(tName!, alphaRaw, out var csvNumMapName))
                    {
                        lkAreaId = csvNumMapName; reason = "patch_csv_num_mapNameX";
                    }
                    else if (ctx.PatchMapping.TryMapBySrcAreaSimple(ctx.MapName, alphaRaw, out var csvNum))
                    {
                        lkAreaId = csvNum; reason = "patch_csv_num";
                    }
                    else { lkAreaId = 0; reason = "unmapped"; }
                }
            }

            if (lkAreaId > 0 && ctx.PatchMapping is not null)
            {
                // best-effort: capture parent id from CSV mapping for auditing
                if (!ctx.PatchMapping.TryGetTargetParentId(lkAreaId, out tgtParent)) tgtParent = 0;
            }

            sw.WriteLine(string.Join(',', new[]
            {
                ctx.TileX.ToString(),
                ctx.TileY.ToString(),
                i.ToString(),
                alphaRaw.ToString(),
                lkAreaId.ToString(),
                tgtParent.ToString(),
                onDisk.ToString(),
                reason,
                EscapeCsv(lkName)
            }));
        }
    }

    // Per-tile SVG visualization with legend CSV. Uses LK AreaID on disk and optional cached LK names.
    private static void WriteAreaVisualizationSvg(WriteContext ctx, string adtPath)
    {
        using var fs = new FileStream(adtPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs);
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mcinDataPos = -1; int mcinSize = 0;
        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32(); long dpos = fs.Position;
            if (fourcc == "MCIN") { mcinDataPos = dpos; mcinSize = sz; break; }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }
        if (mcinDataPos < 0 || mcinSize < 16) return;

        var grid = new int[256]; for (int i = 0; i < 256; i++) grid[i] = -1;
        for (int i = 0; i < 256; i++)
        {
            fs.Position = mcinDataPos + (i * 16);
            int mcnkOffset = br.ReadInt32();
            if (mcnkOffset <= 0) { grid[i] = -1; continue; }
            long pos = (long)mcnkOffset + 8 + 0x34;
            if (pos + 4 > fileLen) { grid[i] = -1; continue; }
            fs.Position = pos;
            grid[i] = (int)br.ReadUInt32();
        }

        // Aggregate counts for legend
        var counts = new Dictionary<int, int>();
        foreach (var v in grid) counts[v] = counts.TryGetValue(v, out var c) ? c + 1 : 1;

        // Colors
        static string HexFor(int id)
        {
            if (id < 0) return "#FF00FF"; // missing
            if (id == 0) return "#FFFFFF"; // unknown -> whiteplate
            var c = ColorForAreaId(id); return ToHex(c);
        }

        // SVG grid (16x16, 32px cells)
        int cell = 32, cols = 16, rows = 16, width = cols * cell, height = rows * cell;
        var sb = new StringBuilder();
        sb.AppendLine($"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>");
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int idx = (y * 16) + x; int id = grid[idx];
                int px = x * cell, py = y * cell; string fill = HexFor(id);
                sb.AppendLine($"<rect x='{px}' y='{py}' width='{cell}' height='{cell}' fill='{fill}' stroke='#999' stroke-width='1' />");
            }
        }
        sb.AppendLine("</svg>");

        string vizRoot = !string.IsNullOrWhiteSpace(ctx.VizDir) ? ctx.VizDir! : Path.Combine(ctx.ExportDir, "viz");
        string outDir = Path.Combine(vizRoot, "maps", ctx.MapName);
        Directory.CreateDirectory(outDir);
        File.WriteAllText(Path.Combine(outDir, $"tile_{ctx.TileX}_{ctx.TileY}.svg"), sb.ToString(), Encoding.UTF8);

        if (!ctx.VizHtml)
        {
            var nameById = s_lkNameByAreaId ?? new Dictionary<int, string>();
            using var sw = new StreamWriter(Path.Combine(outDir, $"legend_tile_{ctx.TileX}_{ctx.TileY}.csv"), append: false, Encoding.UTF8);
            sw.WriteLine("areaId,name,color_hex,count_chunks");
            foreach (var kv in counts.OrderBy(k => k.Key))
            {
                int id = kv.Key; int count = kv.Value; string color = HexFor(id);
                nameById.TryGetValue(id, out var nm); nm ??= string.Empty;
                sw.WriteLine(string.Join(',', id.ToString(), EscapeCsv(nm), color, count.ToString()));
            }
        }
    }

    private static (byte r, byte g, byte b) ColorForAreaId(int id)
    {
        // Stable HSL hash -> RGB. Avoid very light colors since 0 is white.
        double h = (Math.Abs(id) * 2654435761u % 360u);
        double s = 0.65; double l = 0.50;
        return HslToRgb(h, s, l);
    }

    private static (byte r, byte g, byte b) HslToRgb(double h, double s, double l)
    {
        h = h % 360; if (h < 0) h += 360;
        double c = (1 - Math.Abs(2 * l - 1)) * s;
        double x = c * (1 - Math.Abs((h / 60) % 2 - 1));
        double m = l - c / 2;
        double r1 = 0, g1 = 0, b1 = 0;
        if (h < 60) { r1 = c; g1 = x; }
        else if (h < 120) { r1 = x; g1 = c; }
        else if (h < 180) { g1 = c; b1 = x; }
        else if (h < 240) { g1 = x; b1 = c; }
        else if (h < 300) { r1 = x; b1 = c; }
        else { r1 = c; b1 = x; }
        byte r = (byte)Math.Round((r1 + m) * 255);
        byte g = (byte)Math.Round((g1 + m) * 255);
        byte b = (byte)Math.Round((b1 + m) * 255);
        return (r, g, b);
    }

    private static string ToHex((byte r, byte g, byte b) c) => $"#{c.r:X2}{c.g:X2}{c.b:X2}";
    
    // Stitched per-map HTML with inline SVG and a single legend. Reads on-disk ADTs only.
    public static void WriteMapVisualizationHtml(string exportDir, string mapName, DbcPatchMapping? patchMap, string? vizDir)
    {
        const int tiles = 64; const int chunksPerTile = 16; int grid = tiles * chunksPerTile; // 1024
        const int pxPerChunk = 4; int width = grid * pxPerChunk; int height = grid * pxPerChunk;

        // Aggregate AreaIDs per chunk across all tiles
        var worldDir = Path.Combine(exportDir, "World", "Maps", mapName);
        var matrix = new int[grid, grid];
        for (int y = 0; y < grid; y++) for (int x = 0; x < grid; x++) matrix[y, x] = -1;
        if (Directory.Exists(worldDir))
        {
            for (int ty = 0; ty < tiles; ty++)
            {
                for (int tx = 0; tx < tiles; tx++)
                {
                    var adt = Path.Combine(worldDir, $"{mapName}_{tx}_{ty}.adt");
                    if (!File.Exists(adt)) continue;
                    try
                    {
                        using var fs = new FileStream(adt, FileMode.Open, FileAccess.Read, FileShare.Read);
                        using var br = new BinaryReader(fs);
                        // Find MCIN
                        fs.Seek(0, SeekOrigin.Begin);
                        long len = fs.Length; long mcinPos = -1; int mcinSize = 0;
                        while (fs.Position + 8 <= len)
                        {
                            var rev = br.ReadBytes(4); if (rev.Length < 4) break;
                            var fcc = ReverseFourCC(Encoding.ASCII.GetString(rev));
                            int sz = br.ReadInt32(); long dpos = fs.Position;
                            if (fcc == "MCIN") { mcinPos = dpos; mcinSize = sz; break; }
                            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
                        }
                        if (mcinPos < 0 || mcinSize < 16) continue;
                        for (int i = 0; i < 256; i++)
                        {
                            fs.Position = mcinPos + (i * 16);
                            int mcnkOff = br.ReadInt32();
                            int cx = (tx * chunksPerTile) + (i % 16);
                            int cy = (ty * chunksPerTile) + (i / 16);
                            if (mcnkOff <= 0) { matrix[cy, cx] = -1; continue; }
                            long apos = (long)mcnkOff + 8 + 0x34;
                            if (apos + 4 > len) { matrix[cy, cx] = -1; continue; }
                            fs.Position = apos;
                            matrix[cy, cx] = (int)br.ReadUInt32();
                        }
                    }
                    catch { /* ignore */ }
                }
            }
        }

        // Aggregate counts for legend
        var counts = new Dictionary<int, int>();
        for (int y = 0; y < grid; y++) for (int x = 0; x < grid; x++)
        {
            int id = matrix[y, x]; counts[id] = counts.TryGetValue(id, out var c) ? c + 1 : 1;
        }

        // Prepare legend rows using names from CSVs (if present)
        var legendRows = new List<(int id, string name, string hex, int count)>();
        foreach (var kv in counts.OrderByDescending(k => k.Value))
        {
            int id = kv.Key; if (id < 0) continue; // skip missing
            int count = kv.Value;
            string nm = string.Empty;
            if (patchMap is not null && id > 0 && patchMap.TryGetTargetName(id, out var nm0)) nm = nm0;
            string hex = id == 0 ? "#FFFFFF" : ToHex(ColorForAreaId(id));
            legendRows.Add((id, nm, hex, count));
        }

        // Build stitched SVG
        var svg = new StringBuilder();
        svg.AppendLine($"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>");
        for (int y = 0; y < grid; y++)
        {
            for (int x = 0; x < grid; x++)
            {
                int id = matrix[y, x];
                string fill = id switch { -1 => "#FF00FF", 0 => "#FFFFFF", _ => ToHex(ColorForAreaId(id)) };
                int px = x * pxPerChunk; int py = y * pxPerChunk;
                svg.AppendLine($"<rect x='{px}' y='{py}' width='{pxPerChunk}' height='{pxPerChunk}' fill='{fill}' stroke='#777' stroke-width='0.2' />");
            }
        }
        // Tile grid every 16 chunks
        for (int t = 0; t <= grid; t += 16)
        {
            int p = t * pxPerChunk;
            svg.AppendLine($"<line x1='{p}' y1='0' x2='{p}' y2='{height}' stroke='#000' stroke-width='1' opacity='0.25' />");
            svg.AppendLine($"<line x1='0' y1='{p}' x2='{width}' y2='{p}' stroke='#000' stroke-width='1' opacity='0.25' />");
        }
        svg.AppendLine("</svg>");

        // Build HTML
        var html = new StringBuilder();
        html.AppendLine("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Map Visualization</title>");
        html.AppendLine("<style>body{font-family:sans-serif} .wrap{display:flex;gap:16px} .legend{max-height:95vh;overflow:auto} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px 8px} .sw{display:inline-block;width:16px;height:16px;border:1px solid #999;margin-right:6px;vertical-align:middle}</style></head><body>");
        html.AppendLine($"<h1>{mapName} AreaIDs</h1><div class='wrap'><div class='svg'>{svg}</div><div class='legend'><h2>Legend</h2><table><tr><th>AreaID</th><th>Name</th><th>Color</th><th>Count</th></tr>");
        foreach (var row in legendRows)
        {
            string sw = $"<span class='sw' style='background:{row.hex}'></span>";
            html.AppendLine($"<tr><td>{row.id}</td><td>{EscapeCsv(row.name)}</td><td>{sw}{row.hex}</td><td>{row.count}</td></tr>");
        }
        html.AppendLine("</table></div></div></body></html>");

        string root = !string.IsNullOrWhiteSpace(vizDir) ? vizDir! : Path.Combine(exportDir, "viz");
        string outDir = Path.Combine(root, "maps", mapName);
        Directory.CreateDirectory(outDir);
        File.WriteAllText(Path.Combine(outDir, "index.html"), html.ToString(), Encoding.UTF8);
    }
}
