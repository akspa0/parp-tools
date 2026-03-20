using System.Text;
using System.Text.RegularExpressions;
using WoWMapConverter.Core.Formats.PM4;

namespace WoWMapConverter.Core.Services;

public record DevelopmentTileAnalysisResult(
    string TileName,
    int TileX,
    int TileY,
    string TileClass,
    bool HasRootAdt,
    long RootLength,
    bool HasObj0Adt,
    long Obj0Length,
    bool HasTex0Adt,
    long Tex0Length,
    bool HasPm4,
    long Pm4Length,
    bool HasWlw,
    long WlwLength,
    bool HasWlm,
    long WlmLength,
    bool HasWlq,
    long WlqLength,
    bool HasWll,
    long WllLength,
    string RootStatus,
    bool HasMhdr,
    bool HasMcin,
    bool HasMh2o,
    int TopLevelChunkCount,
    int TopLevelMcnkCount,
    int McinEntryCount,
    int ValidRootChunkCount,
    int HeaderIndexMismatchCount,
    int ZeroIndexHeaderCount,
    int DuplicateHeaderIndexCount,
    string RecommendedAction);

public record DevelopmentMapAnalysisReport(
    string MapDirectory,
    string MapName,
    bool WdtExists,
    bool WdlExists,
    int TilesAnalyzed,
    int RootAdtCount,
    int Obj0Count,
    int Tex0Count,
    int Pm4Count,
    int WlwCount,
    int WlmCount,
    int WlqCount,
    int WllCount,
    int ZeroByteRootCount,
    int MissingMcinRootCount,
    int PartialMcnkRootCount,
    int TilesWithoutUsableTerrain,
    int TilesWithHeaderIndexMismatches,
    int TilesWithRepeatedZeroIndices,
    int HealthySplitCount,
    int IndexCorruptCount,
    int ScanOnlyRootCount,
    int WdlRebuildCount,
    int ManualReviewCount,
    List<DevelopmentTileAnalysisResult> Tiles);

public static partial class DevelopmentMapAnalyzer
{
    public const string DefaultDevelopmentMapDirectory = Pm4CoordinateService.DefaultDevelopmentMapDirectory;

    private static readonly HashSet<string> KnownAdtChunks = new(StringComparer.Ordinal)
    {
        "MVER", "MHDR", "MCIN", "MTEX", "MMDX", "MMID", "MWMO", "MWID", "MDDF", "MODF", "MH2O", "MCNK"
    };

    public static DevelopmentMapAnalysisReport Analyze(string? inputDirectory = null, int? tileLimit = null)
    {
        string mapDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory ?? DefaultDevelopmentMapDirectory);
        if (!Directory.Exists(mapDirectory))
            throw new DirectoryNotFoundException($"Development map directory not found: {mapDirectory}");

        string mapName = new DirectoryInfo(mapDirectory).Name;
        var tiles = new Dictionary<(int x, int y), TileFiles>();

        foreach (string path in Directory.EnumerateFiles(mapDirectory).OrderBy(Path.GetFileName))
        {
            if (!TryParseTileFile(path, out int tileX, out int tileY, out string extension, out string suffix))
                continue;

            if (!tiles.TryGetValue((tileX, tileY), out TileFiles? entry))
            {
                entry = new TileFiles(tileX, tileY);
                tiles[(tileX, tileY)] = entry;
            }

            switch (extension)
            {
                case ".adt" when suffix.Length == 0:
                    entry.RootAdtPath = path;
                    break;
                case ".adt" when suffix.Equals("_obj0", StringComparison.OrdinalIgnoreCase):
                    entry.Obj0Path = path;
                    break;
                case ".adt" when suffix.Equals("_tex0", StringComparison.OrdinalIgnoreCase):
                    entry.Tex0Path = path;
                    break;
                case ".pm4":
                    entry.Pm4Path = path;
                    break;
                case ".wlw":
                    entry.WlwPath = path;
                    break;
                case ".wlm":
                    entry.WlmPath = path;
                    break;
                case ".wlq":
                    entry.WlqPath = path;
                    break;
                case ".wll":
                    entry.WllPath = path;
                    break;
            }
        }

        var tileReports = new List<DevelopmentTileAnalysisResult>();
        foreach (TileFiles tile in tiles.Values.OrderBy(t => t.TileY).ThenBy(t => t.TileX))
        {
            tileReports.Add(AnalyzeTile(mapName, tile));
            if (tileLimit.HasValue && tileReports.Count >= tileLimit.Value)
                break;
        }

        return new DevelopmentMapAnalysisReport(
            MapDirectory: mapDirectory,
            MapName: mapName,
            WdtExists: File.Exists(Path.Combine(mapDirectory, $"{mapName}.wdt")),
            WdlExists: File.Exists(Path.Combine(mapDirectory, $"{mapName}.wdl")) || File.Exists(Path.Combine(mapDirectory, $"{mapName}.wdl.mpq")),
            TilesAnalyzed: tileReports.Count,
            RootAdtCount: tileReports.Count(t => t.HasRootAdt),
            Obj0Count: tileReports.Count(t => t.HasObj0Adt),
            Tex0Count: tileReports.Count(t => t.HasTex0Adt),
            Pm4Count: tileReports.Count(t => t.HasPm4),
            WlwCount: tileReports.Count(t => t.HasWlw),
            WlmCount: tileReports.Count(t => t.HasWlm),
            WlqCount: tileReports.Count(t => t.HasWlq),
            WllCount: tileReports.Count(t => t.HasWll),
            ZeroByteRootCount: tileReports.Count(t => t.HasRootAdt && t.RootLength == 0),
            MissingMcinRootCount: tileReports.Count(t => t.HasRootAdt && t.RootLength > 0 && !t.HasMcin),
            PartialMcnkRootCount: tileReports.Count(t => t.HasRootAdt && t.ValidRootChunkCount > 0 && t.ValidRootChunkCount < 256),
            TilesWithoutUsableTerrain: tileReports.Count(t => !t.HasRootAdt || t.ValidRootChunkCount == 0),
            TilesWithHeaderIndexMismatches: tileReports.Count(t => t.HeaderIndexMismatchCount > 0),
            TilesWithRepeatedZeroIndices: tileReports.Count(t => t.ZeroIndexHeaderCount > 1),
            HealthySplitCount: tileReports.Count(t => t.TileClass == "healthy-split"),
            IndexCorruptCount: tileReports.Count(t => t.TileClass == "index-corrupt"),
            ScanOnlyRootCount: tileReports.Count(t => t.TileClass == "scan-only-root"),
            WdlRebuildCount: tileReports.Count(t => t.TileClass == "wdl-rebuild"),
            ManualReviewCount: tileReports.Count(t => t.TileClass == "manual-review"),
            Tiles: tileReports);
    }

    private static DevelopmentTileAnalysisResult AnalyzeTile(string mapName, TileFiles tile)
    {
        long rootLength = GetFileLength(tile.RootAdtPath);
        long obj0Length = GetFileLength(tile.Obj0Path);
        long tex0Length = GetFileLength(tile.Tex0Path);
        long pm4Length = GetFileLength(tile.Pm4Path);
        long wlwLength = GetFileLength(tile.WlwPath);
        long wlmLength = GetFileLength(tile.WlmPath);
        long wlqLength = GetFileLength(tile.WlqPath);
        long wllLength = GetFileLength(tile.WllPath);

        var inspection = tile.RootAdtPath != null
            ? InspectRootAdt(tile.RootAdtPath)
            : RootInspection.Missing;

        string tileName = $"{mapName}_{tile.TileX}_{tile.TileY}";
        string tileClass = ClassifyTile(tile, inspection);
        string recommendedAction = RecommendAction(tileClass, tile, inspection);

        return new DevelopmentTileAnalysisResult(
            TileName: tileName,
            TileX: tile.TileX,
            TileY: tile.TileY,
            TileClass: tileClass,
            HasRootAdt: tile.RootAdtPath != null,
            RootLength: rootLength,
            HasObj0Adt: tile.Obj0Path != null,
            Obj0Length: obj0Length,
            HasTex0Adt: tile.Tex0Path != null,
            Tex0Length: tex0Length,
            HasPm4: tile.Pm4Path != null,
            Pm4Length: pm4Length,
            HasWlw: tile.WlwPath != null,
            WlwLength: wlwLength,
            HasWlm: tile.WlmPath != null,
            WlmLength: wlmLength,
            HasWlq: tile.WlqPath != null,
            WlqLength: wlqLength,
            HasWll: tile.WllPath != null,
            WllLength: wllLength,
            RootStatus: inspection.RootStatus,
            HasMhdr: inspection.HasMhdr,
            HasMcin: inspection.HasMcin,
            HasMh2o: inspection.HasMh2o,
            TopLevelChunkCount: inspection.TopLevelChunkCount,
            TopLevelMcnkCount: inspection.TopLevelMcnkCount,
            McinEntryCount: inspection.McinEntryCount,
            ValidRootChunkCount: inspection.ValidRootChunkCount,
            HeaderIndexMismatchCount: inspection.HeaderIndexMismatchCount,
            ZeroIndexHeaderCount: inspection.ZeroIndexHeaderCount,
            DuplicateHeaderIndexCount: inspection.DuplicateHeaderIndexCount,
                RecommendedAction: recommendedAction);
    }

    private static RootInspection InspectRootAdt(string rootAdtPath)
    {
        byte[] bytes = File.ReadAllBytes(rootAdtPath);
        if (bytes.Length == 0)
            return RootInspection.ZeroByte;

        List<TopLevelChunk> chunks = ReadTopLevelChunks(bytes);
        TopLevelChunk? mhdr = chunks.FirstOrDefault(chunk => chunk.Signature == "MHDR");
        TopLevelChunk? mcin = chunks.FirstOrDefault(chunk => chunk.Signature == "MCIN");
        int topLevelMcnkCount = chunks.Count(chunk => chunk.Signature == "MCNK");

        List<int> offsets = new();
        string rootStatus;
        bool hasMcin = false;
        int mcinEntryCount = 0;

        if (mcin != null)
        {
            hasMcin = true;
            offsets = ReadMcinOffsets(bytes, mcin.Offset, mcin.Size, out mcinEntryCount);
        }
        else
        {
            offsets = chunks
                .Where(chunk => chunk.Signature == "MCNK")
                .Select(chunk => chunk.Offset)
                .Take(256)
                .ToList();
        }

        OffsetInspection offsetInspection = InspectOffsets(bytes, offsets);

        if (bytes.Length == 0)
            rootStatus = "zero-byte";
        else if (offsetInspection.ValidRootChunkCount == 0)
            rootStatus = hasMcin ? "mcin-no-valid-mcnk" : (topLevelMcnkCount > 0 ? "scan-no-valid-mcnk" : "no-mcnk");
        else if (!hasMcin)
            rootStatus = "scan-only";
        else if (offsetInspection.ValidRootChunkCount < 256)
            rootStatus = "mcin-partial";
        else
            rootStatus = "mcin-valid";

        return new RootInspection(
            RootStatus: rootStatus,
            HasMhdr: mhdr != null,
            HasMcin: hasMcin,
            HasMh2o: chunks.Any(chunk => chunk.Signature == "MH2O"),
            TopLevelChunkCount: chunks.Count,
            TopLevelMcnkCount: topLevelMcnkCount,
            McinEntryCount: mcinEntryCount,
            ValidRootChunkCount: offsetInspection.ValidRootChunkCount,
            HeaderIndexMismatchCount: offsetInspection.HeaderIndexMismatchCount,
            ZeroIndexHeaderCount: offsetInspection.ZeroIndexHeaderCount,
            DuplicateHeaderIndexCount: offsetInspection.DuplicateHeaderIndexCount);
    }

    private static OffsetInspection InspectOffsets(byte[] bytes, List<int> offsets)
    {
        int validRootChunkCount = 0;
        int headerIndexMismatchCount = 0;
        int zeroIndexHeaderCount = 0;
        int duplicateHeaderIndexCount = 0;
        var seenIndices = new HashSet<(uint x, uint y)>();

        int entryCount = Math.Min(256, offsets.Count);
        for (int i = 0; i < entryCount; i++)
        {
            int offset = offsets[i];
            if (offset <= 0 || offset + 8 > bytes.Length)
                continue;

            string signature = NormalizeFourCc(Encoding.ASCII.GetString(bytes, offset, 4));
            if (signature != "MCNK")
                continue;

            int chunkSize = BitConverter.ToInt32(bytes, offset + 4);
            if (chunkSize < 128 || offset + 8 + chunkSize > bytes.Length)
                continue;

            uint indexX = BitConverter.ToUInt32(bytes, offset + 8 + 0x04);
            uint indexY = BitConverter.ToUInt32(bytes, offset + 8 + 0x08);

            validRootChunkCount++;
            if (indexX == 0 && indexY == 0)
                zeroIndexHeaderCount++;

            uint expectedX = (uint)(i % 16);
            uint expectedY = (uint)(i / 16);
            if (indexX != expectedX || indexY != expectedY)
                headerIndexMismatchCount++;

            if (!seenIndices.Add((indexX, indexY)))
                duplicateHeaderIndexCount++;
        }

        return new OffsetInspection(
            ValidRootChunkCount: validRootChunkCount,
            HeaderIndexMismatchCount: headerIndexMismatchCount,
            ZeroIndexHeaderCount: zeroIndexHeaderCount,
            DuplicateHeaderIndexCount: duplicateHeaderIndexCount);
    }

    private static string ClassifyTile(TileFiles tile, RootInspection inspection)
    {
        if (tile.RootAdtPath == null || inspection.RootStatus == "missing")
            return "wdl-rebuild";

        if (inspection.RootStatus == "zero-byte")
            return "wdl-rebuild";

        if (!inspection.HasMcin && tile.RootAdtPath != null && inspection.TopLevelMcnkCount > 0)
            return "scan-only-root";

        if (inspection.ValidRootChunkCount == 0)
            return "wdl-rebuild";

        if (inspection.RootStatus == "mcin-partial"
            || inspection.HeaderIndexMismatchCount > 0
            || inspection.ZeroIndexHeaderCount > 1
            || inspection.DuplicateHeaderIndexCount > 0)
            return "index-corrupt";

        if (inspection.ValidRootChunkCount > 0)
            return "healthy-split";

        return "manual-review";
    }

    private static string RecommendAction(string tileClass, TileFiles tile, RootInspection inspection)
    {
        return tileClass switch
        {
            "healthy-split" => (tile.Tex0Path != null || tile.Obj0Path != null) ? "merge-split" : "preserve-root",
            "index-corrupt" => "repair-mcnk-indices",
            "scan-only-root" => "scan-order-root",
            "wdl-rebuild" => "generate-from-wdl",
            "manual-review" when tile.RootAdtPath != null && inspection.ValidRootChunkCount > 0 => "manual-review-root",
            "manual-review" => "manual-review",
            _ => "manual-review"
        };
    }

    private static long GetFileLength(string? path) => path != null && File.Exists(path) ? new FileInfo(path).Length : 0L;

    private static List<TopLevelChunk> ReadTopLevelChunks(byte[] bytes)
    {
        var chunks = new List<TopLevelChunk>();
        int position = 0;

        while (position + 8 <= bytes.Length)
        {
            string rawSignature = Encoding.ASCII.GetString(bytes, position, 4);
            int size = BitConverter.ToInt32(bytes, position + 4);
            if (size < 0 || position + 8 + size > bytes.Length)
                break;

            chunks.Add(new TopLevelChunk(NormalizeFourCc(rawSignature), rawSignature, position, size));

            int next = position + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= position)
                break;

            position = next;
        }

        return chunks;
    }

    private static List<int> ReadMcinOffsets(byte[] bytes, int mcinOffset, int mcinSize, out int entryCount)
    {
        var offsets = new List<int>(256);
        entryCount = Math.Min(256, Math.Max(0, mcinSize / 16));
        int dataStart = mcinOffset + 8;

        for (int i = 0; i < entryCount; i++)
        {
            int entryOffset = dataStart + i * 16;
            if (entryOffset + 4 > bytes.Length)
                break;

            offsets.Add(BitConverter.ToInt32(bytes, entryOffset));
        }

        return offsets;
    }

    private static string NormalizeFourCc(string rawSignature)
    {
        if (KnownAdtChunks.Contains(rawSignature))
            return rawSignature;

        string reversed = new string(rawSignature.Reverse().ToArray());
        return KnownAdtChunks.Contains(reversed) ? reversed : rawSignature;
    }

    private static bool TryParseTileFile(string path, out int tileX, out int tileY, out string extension, out string suffix)
    {
        Match match = TileFilePattern().Match(Path.GetFileName(path));
        if (match.Success
            && int.TryParse(match.Groups[2].Value, out tileX)
            && int.TryParse(match.Groups[3].Value, out tileY))
        {
            extension = "." + match.Groups[5].Value.ToLowerInvariant();
            suffix = match.Groups[4].Value;
            return true;
        }

        tileX = 0;
        tileY = 0;
        extension = string.Empty;
        suffix = string.Empty;
        return false;
    }

    [GeneratedRegex(@"^(.+?)_(\d+)_(\d+)((?:_obj0|_tex0)?)\.(adt|pm4|wlw|wlm|wlq|wll)$", RegexOptions.IgnoreCase)]
    private static partial Regex TileFilePattern();

    private sealed class TileFiles
    {
        public TileFiles(int tileX, int tileY)
        {
            TileX = tileX;
            TileY = tileY;
        }

        public int TileX { get; }
        public int TileY { get; }
        public string? RootAdtPath { get; set; }
        public string? Obj0Path { get; set; }
        public string? Tex0Path { get; set; }
        public string? Pm4Path { get; set; }
        public string? WlwPath { get; set; }
        public string? WlmPath { get; set; }
        public string? WlqPath { get; set; }
        public string? WllPath { get; set; }
    }

    private sealed record TopLevelChunk(string Signature, string RawSignature, int Offset, int Size);

    private sealed record RootInspection(
        string RootStatus,
        bool HasMhdr,
        bool HasMcin,
        bool HasMh2o,
        int TopLevelChunkCount,
        int TopLevelMcnkCount,
        int McinEntryCount,
        int ValidRootChunkCount,
        int HeaderIndexMismatchCount,
        int ZeroIndexHeaderCount,
        int DuplicateHeaderIndexCount)
    {
        public static RootInspection Missing => new(
            RootStatus: "missing",
            HasMhdr: false,
            HasMcin: false,
            HasMh2o: false,
            TopLevelChunkCount: 0,
            TopLevelMcnkCount: 0,
            McinEntryCount: 0,
            ValidRootChunkCount: 0,
            HeaderIndexMismatchCount: 0,
            ZeroIndexHeaderCount: 0,
            DuplicateHeaderIndexCount: 0);

        public static RootInspection ZeroByte => new(
            RootStatus: "zero-byte",
            HasMhdr: false,
            HasMcin: false,
            HasMh2o: false,
            TopLevelChunkCount: 0,
            TopLevelMcnkCount: 0,
            McinEntryCount: 0,
            ValidRootChunkCount: 0,
            HeaderIndexMismatchCount: 0,
            ZeroIndexHeaderCount: 0,
            DuplicateHeaderIndexCount: 0);
    }

    private sealed record OffsetInspection(
        int ValidRootChunkCount,
        int HeaderIndexMismatchCount,
        int ZeroIndexHeaderCount,
        int DuplicateHeaderIndexCount);
}