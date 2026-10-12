using WowViewer.Core.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.IO.Maps;

/// <summary>Outcome of exporting a folder of DAT files, enough to drive both a CLI summary and a status line.</summary>
public sealed class DatFolderExportResult
{
    public required string SourceRoot { get; init; }
    public required string OutputDirectory { get; init; }
    public required string MapName { get; init; }
    public required bool Transposed { get; init; }
    public int FilesSeen { get; set; }
    public List<string> SkipReasons { get; } = [];
    public HashSet<(int TileX, int TileY)> TilesWritten { get; } = [];
    public SortedDictionary<uint, int> Versions { get; } = [];
    public DatToLkConversionReport Report { get; } = new();

    /// <summary>Formats actually written.</summary>
    public List<MapConversionTargetFormat> TargetsWritten { get; } = [];

    /// <summary>The Alpha 0.5.3 WDT written, when that target was requested.</summary>
    public string? AlphaWdtPath { get; set; }

    public string VersionSummary => string.Join(", ", Versions.Select(static kv => $"v{kv.Key} x{kv.Value}"));

    /// <summary>One line for a status bar or a log.</summary>
    public string Summary =>
        $"DAT -> {(TargetsWritten.Count == 0 ? "nothing" : string.Join(" + ", TargetsWritten.Select(MapConversionFormats.GetDisplayName)))}: " +
        $"{TilesWritten.Count} tile(s) ({VersionSummary}), {Report.ChunksWritten} chunks, " +
        $"{Report.LayersWritten} layers, {Report.AlphaMapsWritten} alpha maps, {Report.ObjectsPlaced} objects " +
        $"-> {OutputDirectory}";
}

/// <summary>
/// Spec 247 US3: exports a whole folder of AHDR-family DAT files as an LK v18 map (one ADT per tile, the WDT,
/// and a loss manifest). Lives in core so the CLI and the viewer run exactly the same conversion rather than
/// each carrying their own copy of the walk.
/// </summary>
public static class DatToLkAdtFolderExporter
{
    public const string ManifestFileName = "conversion-manifest.txt";

    /// <summary>Default output root for a map, relative to the project output directory (AGENTS.md 9.3).</summary>
    public static string DefaultOutputDirectory(string mapName)
        => Path.Combine("output", "dat-lk-export", mapName);

    /// <summary>Turns a folder name into something safe to use as a map name and file stem.</summary>
    public static string SanitizeName(string? name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return "DatExport";

        var buffer = new char[name.Length];
        int n = 0;
        foreach (char c in name)
            buffer[n++] = char.IsLetterOrDigit(c) || c is '_' or '-' ? c : '_';

        string cleaned = new string(buffer, 0, n).Trim('_');
        return cleaned.Length == 0 ? "DatExport" : cleaned;
    }

    /// <summary>
    /// Converts and writes every AHDR-family file in <paramref name="root"/>. Files that are not AHDR-family are
    /// ignored; files that are but carry no tile location are recorded in
    /// <see cref="DatFolderExportResult.SkipReasons"/>. Writes nothing and returns a result with no tiles when
    /// the folder yields none.
    /// </summary>
    public static DatFolderExportResult Export(
        string root,
        string? outputDirectory = null,
        string? mapName = null,
        DatToLkConversionOptions? options = null,
        IReadOnlyCollection<MapConversionTargetFormat>? targets = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);

        string map = SanitizeName(mapName ?? Path.GetFileName(Path.TrimEndingDirectorySeparator(root)));
        string outDir = string.IsNullOrWhiteSpace(outputDirectory) ? DefaultOutputDirectory(map) : outputDirectory;
        options ??= new DatToLkConversionOptions();

        var result = new DatFolderExportResult
        {
            SourceRoot = Path.GetFullPath(root),
            OutputDirectory = Path.GetFullPath(outDir),
            MapName = map,
            Transposed = options.TransposeChunks,
        };

        Directory.CreateDirectory(outDir);

        var wanted = targets is { Count: > 0 } ? targets : [MapConversionTargetFormat.LkAdtV18];
        foreach (MapConversionTargetFormat t in wanted)
        {
            if (!MapConversionFormats.HasWriter(t))
                throw new NotSupportedException(MapConversionFormats.GetUnavailableReason(t));
        }

        bool writeLk = wanted.Contains(MapConversionTargetFormat.LkAdtV18);
        bool writeAlpha = wanted.Contains(MapConversionTargetFormat.AlphaWdt053);

        // Keep the LK document per tile, not the Alpha one: LkToAlphaConverter allocates a 16 MB alpha
        // pack plus a 4 MB shadow mask per tile, so holding every AlphaTileData at once would be ~20 MB x
        // tile count (14 GB on the 699-file v26 corpus). The WDT writer takes a provider, so tiles are
        // converted one at a time on demand instead.
        var lkByTile = new Dictionary<(int TileX, int TileY), LkAdtData>();

        foreach (string path in Directory.EnumerateFiles(root).OrderBy(static p => p, StringComparer.OrdinalIgnoreCase))
        {
            byte[] data;
            try
            {
                data = File.ReadAllBytes(path);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                result.SkipReasons.Add($"{Path.GetFileName(path)}: {ex.Message}");
                continue;
            }

            if (!AdtAhdrReader.IsAhdrFamily(data))
                continue;

            result.FilesSeen++;
            AdtAhdrTile tile = AdtAhdrReader.Read(data, path);
            result.Versions[tile.Version] = result.Versions.GetValueOrDefault(tile.Version) + 1;

            if (!AdtAhdrReader.TryReadTileLocation(data, out int tx, out int ty)
                && !AdtAhdrReader.TryParseTileLocationFromName(path, out tx, out ty))
            {
                result.SkipReasons.Add($"{Path.GetFileName(path)}: no ALOC and no tile coordinates in the name");
                continue;
            }

            LkAdtData adt = DatToLkAdtConverter.Convert(tile, tx, ty, map, options, result.Report);
            if (writeLk)
                LkAdtWriter.Write(Path.Combine(outDir, $"{map}_{tx}_{ty}.adt"), adt);
            if (writeAlpha)
                lkByTile[(tx, ty)] = adt;
            result.TilesWritten.Add((tx, ty));
        }

        if (result.TilesWritten.Count == 0)
            return result;

        if (writeLk)
        {
            // MCCV is carried when the source has it and alpha is written as big (8-bit) MCAL, so both flags are set.
            LkWdtWriter.Write(
                Path.Combine(outDir, $"{map}.wdt"),
                result.TilesWritten,
                new LkWdtWriteOptions { HasMccv = true, HasBigAlpha = true });
            result.TargetsWritten.Add(MapConversionTargetFormat.LkAdtV18);
        }

        if (writeAlpha)
        {
            WriteAlphaWdt(result, outDir, map, lkByTile);
            result.TargetsWritten.Add(MapConversionTargetFormat.AlphaWdt053);
        }

        File.WriteAllLines(Path.Combine(outDir, ManifestFileName), BuildManifest(result));
        return result;
    }

    /// <summary>
    /// Alpha 0.5.3 is reached through the existing LK document: DAT -> LkAdtData -> AlphaTileData ->
    /// monolithic WDT, reusing <see cref="LkToAlphaConverter"/> and <see cref="AlphaWdtWriter"/> rather than
    /// adding a second conversion path. The Alpha WDT's model tables are map-global, so the names are unioned
    /// from every tile's placements first; tiles themselves are converted lazily by the provider.
    /// </summary>
    private static void WriteAlphaWdt(
        DatFolderExportResult result,
        string outDir,
        string map,
        Dictionary<(int TileX, int TileY), LkAdtData> lkByTile)
    {
        var mdxNames = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
        var wmoNames = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (LkAdtData adt in lkByTile.Values)
        {
            foreach (string n in adt.ModelNames)
                mdxNames.Add(n);
            foreach (string n in adt.WorldModelNames)
                wmoNames.Add(n);
        }

        var tileKeys = lkByTile.Keys.OrderBy(static k => k.TileX).ThenBy(static k => k.TileY).ToList();
        string path = Path.Combine(outDir, $"{map}.wdt");
        if (result.TargetsWritten.Contains(MapConversionTargetFormat.LkAdtV18))
            path = Path.Combine(outDir, $"{map}_alpha.wdt"); // the LK target already owns <map>.wdt

        AlphaWdtWriter.Write(
            path,
            map,
            tileKeys,
            (tx, ty) => LkToAlphaConverter.ConvertTile(lkByTile[(tx, ty)], tx, ty),
            mdxNames.ToList(),
            wmoNames.ToList());

        result.AlphaWdtPath = Path.GetFullPath(path);
    }

    /// <summary>The CARRIED/DROPPED manifest (spec 247 FR-013). Totals only: no per-tile running counts.</summary>
    public static IEnumerable<string> BuildManifest(DatFolderExportResult r)
    {
        DatToLkConversionReport report = r.Report;

        yield return "DAT -> LK v18 ADT conversion manifest";
        yield return $"Generated:  {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}Z";
        yield return $"Source:     {r.SourceRoot}";
        yield return $"Map name:   {r.MapName}";
        yield return $"Revisions:  {r.VersionSummary}";
        yield return $"Chunk axes: {(r.Transposed ? "transposed" : "source ACNK indices, untransposed")}";
        yield return $"Targets:    {(r.TargetsWritten.Count == 0 ? "none" : string.Join(", ", r.TargetsWritten.Select(MapConversionFormats.GetDisplayName)))}";
        if (r.AlphaWdtPath is not null)
            yield return $"Alpha WDT:  {r.AlphaWdtPath}";
        yield return "";
        yield return $"Files seen:        {r.FilesSeen}";
        yield return $"Files skipped:     {r.SkipReasons.Count}";
        yield return $"Tiles written:     {r.TilesWritten.Count}";
        yield return $"Chunks in source:  {report.ChunksInSource}";
        yield return $"Chunks written:    {report.ChunksWritten}";
        yield return $"Empty MCNK filled: {report.ChunksSynthesizedEmpty}";
        yield return "";
        yield return "CARRIED";
        yield return "  heights (MCVT), normals (MCNR)  every written chunk";
        yield return $"  texture layers (MCLY)           {report.LayersWritten}";
        yield return $"  alpha maps (MCAL, 8-bit)        {report.AlphaMapsWritten}";
        yield return $"  shadows (MCSH)                  {report.ShadowMapsCarried}";
        yield return $"  area ids                        {report.AreaIdsCarried}";
        yield return $"  vertex colours (MCCV)           {report.VertexColourChunks}";
        yield return $"  object placements (MDDF/MODF)   {report.ObjectsPlaced}";
        yield return "";
        yield return "DROPPED";
        yield return $"  layers with no usable alpha     {report.LayersDroppedNoAlpha}";
        yield return $"  objects with no model name      {report.ObjectsSkippedUnnamed}";
        yield return $"  ADST rows (no LK equivalent)    {report.AdstRowsDropped}";

        if (report.SawAoch)
            yield return "  AOCH                            present in source; 2048 B, all-zero and unexplained, no LK equivalent";

        if (report.ObjectsWithNegativeUniqueId > 0)
        {
            yield return $"  (carried, but note) {report.ObjectsWithNegativeUniqueId} placement(s) have a negative DAT uniqueId; "
                + "LK MDDF uniqueId is unsigned so they become large unsigned values";
        }

        foreach (string note in report.Notes)
            yield return $"  note: {note}";

        if (r.SkipReasons.Count > 0)
        {
            yield return "";
            yield return "SKIPPED FILES";
            foreach (string reason in r.SkipReasons)
                yield return $"  {reason}";
        }
    }
}
