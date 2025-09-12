using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.WowFiles.Alpha;

namespace AlphaWdtAnalyzer.Core;

public sealed class AdtScanner
{
    public sealed class Result
    {
        public HashSet<string> WmoAssets { get; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> M2Assets { get; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> BlpAssets { get; } = new(StringComparer.OrdinalIgnoreCase);
        public List<PlacementRecord> Placements { get; } = new();
        public List<MapTile> Tiles { get; } = new();
    }

    public Result Scan(WdtAlphaScanner wdt)
    {
        var result = new Result();
        var mdnm = wdt.MdnmFiles.Select(ListfileLoader.NormalizePath).ToList();
        var monm = wdt.MonmFiles.Select(ListfileLoader.NormalizePath).ToList();
        var baseDir = Path.GetDirectoryName(wdt.WdtPath) ?? ".";

        foreach (var adtNum in wdt.AdtNumbers)
        {
            var off = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (off <= 0) continue;

            var adt = new AdtAlpha(wdt.WdtPath, off, adtNum);
            var x = adt.GetXCoord();
            var y = adt.GetYCoord();

            // record tile path (expected next to WDT)
            var adtBasename = Path.GetFileNameWithoutExtension(wdt.WdtPath) + $"_{x}_{y}.adt";
            var adtPath = Path.Combine(baseDir, adtBasename);
            result.Tiles.Add(new MapTile(x, y, adtPath));

            // Parse MDDF entries (36 bytes each)
            var mddf = adt.GetMddfRaw();
            const int mddfEntrySize = 36;
            for (int start = 0; start + mddfEntrySize <= mddf.Length; start += mddfEntrySize)
            {
                int nameIndex = BitConverter.ToInt32(mddf, start + 0);
                int? uniqueId = null;
                try { uniqueId = BitConverter.ToInt32(mddf, start + 4); } catch { uniqueId = null; }

                if (nameIndex >= 0 && nameIndex < mdnm.Count)
                {
                    var p = mdnm[nameIndex];
                    result.M2Assets.Add(p);
                    result.Placements.Add(new PlacementRecord(AssetType.MdxOrM2, p, wdt.MapName, x, y, uniqueId));
                }
            }

            // Parse MODF entries (64 bytes each)
            var modf = adt.GetModfRaw();
            const int modfEntrySize = 64;
            for (int start = 0; start + modfEntrySize <= modf.Length; start += modfEntrySize)
            {
                int nameIndex = BitConverter.ToInt32(modf, start + 0);
                int? uniqueId = null;
                try { uniqueId = BitConverter.ToInt32(modf, start + 4); } catch { uniqueId = null; }

                if (nameIndex >= 0 && nameIndex < monm.Count)
                {
                    var p = monm[nameIndex];
                    result.WmoAssets.Add(p);
                    result.Placements.Add(new PlacementRecord(AssetType.Wmo, p, wdt.MapName, x, y, uniqueId));
                }
            }

            // Textures via MTEX string table
            foreach (var tex in adt.GetMtexTextureNames())
            {
                var norm = ListfileLoader.NormalizePath(tex);
                if (!string.IsNullOrWhiteSpace(norm)) result.BlpAssets.Add(norm);
            }
        }

        return result;
    }
}
