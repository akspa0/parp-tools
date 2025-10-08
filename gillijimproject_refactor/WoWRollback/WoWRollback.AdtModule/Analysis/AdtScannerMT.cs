using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GillijimProject.WowFiles.Alpha;
using AlphaWdtAnalyzer.Core; // For WdtAlphaScanner and ListfileLoader

namespace WoWRollback.AdtModule.Analysis;

internal sealed class AdtScannerMT
{
    internal sealed class Result
    {
        public HashSet<string> WmoAssets { get; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> M2Assets { get; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> BlpAssets { get; } = new(StringComparer.OrdinalIgnoreCase);
        public List<PlacementRecord> Placements { get; } = new();
        public List<MapTile> Tiles { get; } = new();
    }

    public Result Scan(WdtAlphaScanner wdt, int degreeOfParallelism)
    {
        var result = new Result();
        var mdnm = wdt.MdnmFiles.Select(ListfileLoader.NormalizePath).ToList();
        var monm = wdt.MonmFiles.Select(ListfileLoader.NormalizePath).ToList();
        var baseDir = Path.GetDirectoryName(wdt.WdtPath) ?? ".";

        Console.WriteLine($"[AdtScannerMT] Scanning WDT: {wdt.MapName}, {wdt.AdtNumbers.Count} ADTs (dop={degreeOfParallelism})");

        var tilesBag = new ConcurrentBag<MapTile>();
        var wmoSet = new ConcurrentDictionary<string, byte>(StringComparer.OrdinalIgnoreCase);
        var m2Set = new ConcurrentDictionary<string, byte>(StringComparer.OrdinalIgnoreCase);
        var blpSet = new ConcurrentDictionary<string, byte>(StringComparer.OrdinalIgnoreCase);
        var placementsBag = new ConcurrentBag<PlacementRecord>();

        Parallel.ForEach(
            wdt.AdtNumbers,
            new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, degreeOfParallelism) },
            adtNum =>
            {
                var off = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
                if (off <= 0) return;

                var adt = new AdtAlpha(wdt.WdtPath, off, adtNum);
                var x = adt.GetXCoord();
                var y = adt.GetYCoord();

                var adtBasename = Path.GetFileNameWithoutExtension(wdt.WdtPath) + $"_{x}_{y}.adt";
                var adtPath = Path.Combine(baseDir, adtBasename);
                tilesBag.Add(new MapTile(x, y, adtPath));

                // MDDF (objects)
                var mddf = adt.GetMddfRaw();
                const int mddfEntrySize = 36;
                for (int start = 0; start + mddfEntrySize <= mddf.Length; start += mddfEntrySize)
                {
                    int nameIndex = BitConverter.ToInt32(mddf, start + 0);
                    int? uniqueId = null;
                    try { uniqueId = BitConverter.ToInt32(mddf, start + 4); } catch { uniqueId = null; }
                    float worldX = BitConverter.ToSingle(mddf, start + 8);
                    float worldZ = BitConverter.ToSingle(mddf, start + 12);
                    float worldY = BitConverter.ToSingle(mddf, start + 16);
                    float rotX = BitConverter.ToSingle(mddf, start + 20);
                    float rotY = BitConverter.ToSingle(mddf, start + 24);
                    float rotZ = BitConverter.ToSingle(mddf, start + 28);
                    ushort scaleRaw = BitConverter.ToUInt16(mddf, start + 32);
                    ushort flags = BitConverter.ToUInt16(mddf, start + 34);
                    float scale = scaleRaw > 0 ? scaleRaw / 1024.0f : 1.0f;

                    if (nameIndex >= 0 && nameIndex < mdnm.Count)
                    {
                        var p = mdnm[nameIndex];
                        m2Set.TryAdd(p, 1);
                        placementsBag.Add(new PlacementRecord(
                            AssetType.MdxOrM2,
                            p,
                            wdt.MapName,
                            x,
                            y,
                            uniqueId,
                            worldX,
                            worldY,
                            worldZ,
                            rotX,
                            rotY,
                            rotZ,
                            scale,
                            flags,
                            0,
                            0));
                    }
                }

                // MODF (WMOs)
                var modf = adt.GetModfRaw();
                const int modfEntrySize = 64;
                for (int start = 0; start + modfEntrySize <= modf.Length; start += modfEntrySize)
                {
                    int nameIndex = BitConverter.ToInt32(modf, start + 0);
                    int? uniqueId = null;
                    try { uniqueId = BitConverter.ToInt32(modf, start + 4); } catch { uniqueId = null; }
                    float worldX = BitConverter.ToSingle(modf, start + 8);
                    float worldZ = BitConverter.ToSingle(modf, start + 12);
                    float worldY = BitConverter.ToSingle(modf, start + 16);
                    float rotX = BitConverter.ToSingle(modf, start + 20);
                    float rotY = BitConverter.ToSingle(modf, start + 24);
                    float rotZ = BitConverter.ToSingle(modf, start + 28);
                    ushort flags = BitConverter.ToUInt16(modf, start + 56);
                    ushort doodadSet = BitConverter.ToUInt16(modf, start + 58);
                    ushort nameSet = BitConverter.ToUInt16(modf, start + 60);
                    ushort scaleRaw = BitConverter.ToUInt16(modf, start + 62);
                    float scale = scaleRaw > 0 ? scaleRaw / 1024.0f : 1.0f;

                    if (nameIndex >= 0 && nameIndex < monm.Count)
                    {
                        var p = monm[nameIndex];
                        wmoSet.TryAdd(p, 1);
                        placementsBag.Add(new PlacementRecord(
                            AssetType.Wmo,
                            p,
                            wdt.MapName,
                            x,
                            y,
                            uniqueId,
                            worldX,
                            worldY,
                            worldZ,
                            rotX,
                            rotY,
                            rotZ,
                            scale,
                            flags,
                            doodadSet,
                            nameSet));
                    }
                }

                // MTEX textures
                foreach (var tex in adt.GetMtexTextureNames())
                {
                    var norm = ListfileLoader.NormalizePath(tex);
                    if (!string.IsNullOrWhiteSpace(norm)) blpSet.TryAdd(norm, 1);
                }
            });

        result.Tiles.AddRange(tilesBag.OrderBy(t => t.Y).ThenBy(t => t.X));
        result.WmoAssets.UnionWith(wmoSet.Keys);
        result.M2Assets.UnionWith(m2Set.Keys);
        result.BlpAssets.UnionWith(blpSet.Keys);
        result.Placements.AddRange(placementsBag);
        return result;
    }
}
