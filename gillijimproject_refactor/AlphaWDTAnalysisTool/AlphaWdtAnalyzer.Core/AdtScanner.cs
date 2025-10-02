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

        Console.WriteLine($"[AdtScanner] Scanning WDT: {wdt.MapName}, {wdt.AdtNumbers.Count} ADTs");

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
            // In Alpha 0.5.3, MDDF coordinates are ALWAYS (0,0,0) - they weren't implemented yet!
            // Blizzard didn't add world positions to MDDF/MODF until later Alpha builds.
            // We can only get tile coordinates, not actual world positions.
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
                    result.M2Assets.Add(p);
                    result.Placements.Add(new PlacementRecord(
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

            // Parse MODF entries (64 bytes each)
            // Layout: [0-3] nameId, [4-7] uniqueId, [8-11] X, [12-15] Z, [16-19] Y, [20-63] rotation/bbox/flags
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
                    result.WmoAssets.Add(p);
                    result.Placements.Add(new PlacementRecord(
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
