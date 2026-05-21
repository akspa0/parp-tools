using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static class Pm4ResearchCrossTileAnalyzer
{
    public static Pm4CrossTileReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<(string Path, Pm4ResearchDocument Doc)> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(path => (path, Pm4ResearchReader.ReadFile(path)))
            .ToList();

        int totalFiles = files.Count;
        int nonEmptyFiles = files.Count(f => f.Doc.KnownChunks.Msur is { Count: > 0 });

        Dictionary<uint, Pm4CrossTileCk24Accum> ck24Accum = new();
        Dictionary<string, Pm4CrossTileTileAccum> tileAccum = new(StringComparer.Ordinal);

        foreach ((string path, Pm4ResearchDocument doc) in files)
        {
            string? tileCoord = ExtractTileCoordinate(path);
            if (tileCoord is null)
                continue;

            IReadOnlyList<Pm4MsurEntry> msur = doc.KnownChunks.Msur;
            IReadOnlyList<Pm4MslkEntry> mslk = doc.KnownChunks.Mslk;
            IReadOnlyList<Pm4MprlEntry> mprl = doc.KnownChunks.Mprl;
            int mscnCount = doc.KnownChunks.Mscn?.Count ?? 0;

            int ck24GroupsInTile = 0;
            Dictionary<uint, int> tileCk24SurfaceCounts = new();

            for (int i = 0; i < msur.Count; i++)
            {
                Pm4MsurEntry surface = msur[i];
                uint ck24 = surface.Ck24;
                byte ck24Type = surface.Ck24Type;
                ushort ck24ObjectId = surface.Ck24ObjectId;

                if (!ck24Accum.TryGetValue(ck24, out Pm4CrossTileCk24Accum? accum))
                {
                    accum = new Pm4CrossTileCk24Accum(ck24, ck24Type, ck24ObjectId);
                    ck24Accum[ck24] = accum;
                }

                accum.TileCoordinates.Add(tileCoord);
                accum.SurfaceCount++;

                int mscnRef = (int)surface._0x18;
                if (mscnRef >= 0 && mscnRef < mscnCount)
                    accum.MscnRefCount++;

                tileCk24SurfaceCounts.TryGetValue(ck24, out int existing);
                tileCk24SurfaceCounts[ck24] = existing + 1;
            }

            ck24GroupsInTile = tileCk24SurfaceCounts.Count;

            var mshd = doc.KnownChunks.Mshd;
            tileAccum[tileCoord] = new Pm4CrossTileTileAccum(
                tileCoord,
                ck24GroupsInTile,
                msur.Count,
                mslk?.Count ?? 0,
                mscnCount,
                mprl?.Count ?? 0,
                mshd?.Field00 ?? 0,
                mshd?.Field04 ?? 0,
                mshd?.Field08 ?? 0);
        }

        int totalDistinctCk24 = ck24Accum.Count;
        int crossTileCk24Count = 0;
        List<Pm4CrossTileCk24Record> crossTileRecords = new();

        foreach (Pm4CrossTileCk24Accum accum in ck24Accum.Values.OrderByDescending(a => a.TileCoordinates.Count).ThenByDescending(a => a.SurfaceCount))
        {
            if (accum.TileCoordinates.Count > 1)
                crossTileCk24Count++;

            crossTileRecords.Add(new Pm4CrossTileCk24Record(
                accum.Ck24,
                accum.Ck24Type,
                accum.Ck24ObjectId,
                accum.TileCoordinates.OrderBy(c => c).ToList(),
                accum.SurfaceCount,
                accum.MscnRefCount,
                0));
        }

        List<Pm4CrossTileTileSummary> tileSummaries = tileAccum
            .OrderBy(kv => kv.Key)
            .Select(kv => new Pm4CrossTileTileSummary(
                kv.Value.TileCoordinate,
                kv.Value.Ck24GroupCount,
                kv.Value.SurfaceCount,
                kv.Value.MslkCount,
                kv.Value.MscnCount,
                kv.Value.MprlCount,
                kv.Value.Field00,
                kv.Value.Field04,
                kv.Value.Field08,
                false, 0, 0))
            .ToList();

        List<string> notes = new();
        notes.Add($"'{totalDistinctCk24}' distinct CK24 values across '{nonEmptyFiles}' non-empty tiles.");
        notes.Add($"'{crossTileCk24Count}' CK24 values span 2+ tiles ({(crossTileCk24Count * 100.0 / Math.Max(1, totalDistinctCk24)):F1}%).");
        notes.Add($"Top cross-tile CK24 spans '{crossTileRecords.FirstOrDefault()?.TileCoordinates.Count ?? 0}' tiles.");
        notes.Add($"This report reads all PM4 tiles as a single map object — cross-tile CK24 tracking shows which objects span tile boundaries.");
        notes.Add($"No ADT correlation attempted — ADT files in this corpus are not valid/populated.");

        return new Pm4CrossTileReport(
            resolvedDirectory,
            totalFiles,
            nonEmptyFiles,
            0,
            totalDistinctCk24,
            crossTileCk24Count,
            0,
            crossTileRecords.Take(40).ToList(),
            tileSummaries,
            notes);
    }

    private static string? ExtractTileCoordinate(string path)
    {
        string fileName = Path.GetFileNameWithoutExtension(path);
        if (Pm4CoordinateService.TryParseTileCoordinates(path, out int tileX, out int tileY))
            return $"{tileX}_{tileY}";
        return null;
    }

    private sealed class Pm4CrossTileCk24Accum
    {
        public uint Ck24 { get; }
        public byte Ck24Type { get; }
        public ushort Ck24ObjectId { get; }
        public HashSet<string> TileCoordinates { get; } = new(StringComparer.Ordinal);
        public int SurfaceCount { get; set; }
        public int MscnRefCount { get; set; }

        public Pm4CrossTileCk24Accum(uint ck24, byte ck24Type, ushort ck24ObjectId)
        {
            Ck24 = ck24;
            Ck24Type = ck24Type;
            Ck24ObjectId = ck24ObjectId;
        }
    }

    private sealed class Pm4CrossTileTileAccum
    {
        public string TileCoordinate { get; }
        public int Ck24GroupCount { get; }
        public int SurfaceCount { get; }
        public int MslkCount { get; }
        public int MscnCount { get; }
        public int MprlCount { get; }
        public uint Field00 { get; }
        public uint Field04 { get; }
        public uint Field08 { get; }

        public Pm4CrossTileTileAccum(string tileCoordinate, int ck24GroupCount, int surfaceCount, int mslkCount, int mscnCount, int mprlCount, uint field00, uint field04, uint field08)
        {
            TileCoordinate = tileCoordinate;
            Ck24GroupCount = ck24GroupCount;
            SurfaceCount = surfaceCount;
            MslkCount = mslkCount;
            MscnCount = mscnCount;
            MprlCount = mprlCount;
            Field00 = field00;
            Field04 = field04;
            Field08 = field08;
        }
    }
}
