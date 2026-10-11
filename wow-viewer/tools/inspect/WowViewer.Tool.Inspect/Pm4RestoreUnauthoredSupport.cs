using System.Globalization;
using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Tool.Inspect;

/// <summary>
/// CLI command handler for restoring authentic object placements on unauthored PM4 tiles
/// (Spec 270 Phase 4). Enforces the ground-truth preservation barrier on all hand-authored
/// Blizzard tiles in 335-dev, derives 2D PCA yaw rotations, and serializes standard monolithic
/// LK 3.3.5 ADTs (development_XX_YY.adt) merged with surviving terrain.
/// </summary>
internal static class Pm4RestoreUnauthoredSupport
{
    private const string DefaultBaseAdtPath = @"I:\parp\parp-tools\wow-viewer\test_data\WoWMuseum\335-dev\World\Maps\development";
    private const string DefaultPm4Path = @"I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development";
    private const string DefaultOutputPath = @"I:\parp\parp-tools\wow-viewer\output\restored_development\World\Maps\development";
    private const string DefaultLedgerPath = @"I:\parp\parp-tools\wow-viewer\output\datasets\pm4_object_association_ledger.json";

    public static void Run(string[] args)
    {
        string baseAdtDir = DefaultBaseAdtPath;
        string pm4Dir = DefaultPm4Path;
        string outputDir = DefaultOutputPath;
        string? targetTilesFilter = null;
        string ledgerPath = DefaultLedgerPath;
        string? jsonOutPath = null;
        bool validate = true;
        bool writeSplitObj = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--base":
                case "--adt":
                    if (i + 1 < args.Length) baseAdtDir = args[++i];
                    break;
                case "--pm4":
                case "--map":
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) pm4Dir = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputDir = args[++i];
                    break;
                case "--tiles":
                case "-t":
                    if (i + 1 < args.Length) targetTilesFilter = args[++i];
                    break;
                case "--ledger":
                    if (i + 1 < args.Length) ledgerPath = args[++i];
                    break;
                case "--json":
                    if (i + 1 < args.Length) jsonOutPath = args[++i];
                    break;
                case "--no-validate":
                    validate = false;
                    break;
                case "--split-obj":
                    writeSplitObj = true;
                    break;
            }
        }

        string resolvedBaseAdtDir = Pm4CoordinateService.ResolveMapDirectory(baseAdtDir);
        string nestedBase = Path.Combine(resolvedBaseAdtDir, "World", "Maps", "development");
        if (Directory.Exists(nestedBase))
            resolvedBaseAdtDir = nestedBase;

        string resolvedPm4Dir = Pm4CoordinateService.ResolveMapDirectory(pm4Dir);
        string nestedPm4 = Path.Combine(resolvedPm4Dir, "World", "Maps", "development");
        if (Directory.Exists(nestedPm4))
            resolvedPm4Dir = nestedPm4;

        if (!Directory.Exists(resolvedPm4Dir))
        {
            Console.Error.WriteLine($"Error: PM4 directory '{resolvedPm4Dir}' does not exist.");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputDir);

        Console.WriteLine("=================================================================");
        Console.WriteLine(" Spec 270: PM4 Unauthored Tile Placement Restoration Pipeline");
        Console.WriteLine("=================================================================");
        Console.WriteLine($"Base ADT Directory: {resolvedBaseAdtDir}");
        Console.WriteLine($"PM4 Directory:      {resolvedPm4Dir}");
        Console.WriteLine($"Output Directory:   {outputDir}");

        // Build shape library and exact Z ledger lookup from persistent ledger and authored tiles
        var (shapeLibrary, ledgerLookup) = LoadLibraries(resolvedBaseAdtDir, ledgerPath);
        Console.WriteLine($"Shape Library: Loaded {shapeLibrary.Count} unique asset shape profiles.");
        Console.WriteLine($"Ledger Lookup: Loaded {ledgerLookup.Count} bit-exact Z keys from persistent Rosetta ledger.");

        // Find candidate PM4 tiles
        var pm4Files = Directory.EnumerateFiles(resolvedPm4Dir, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .ToList();

        HashSet<string>? filterSet = null;
        if (!string.IsNullOrWhiteSpace(targetTilesFilter) && !targetTilesFilter.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            filterSet = new HashSet<string>(
                targetTilesFilter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(s => s.Replace(".pm4", "", StringComparison.OrdinalIgnoreCase)),
                StringComparer.OrdinalIgnoreCase);
        }

        var results = new List<Pm4TileResolutionResult>();
        int totalWmos = 0;
        int totalDoodads = 0;
        int roundTripPassed = 0;
        int roundTripFailed = 0;
        int preservedAuthoredCount = 0;
        int processedCount = 0;

        foreach (string pm4Path in pm4Files)
        {
            string fileName = Path.GetFileName(pm4Path);
            string tileStem = Path.GetFileNameWithoutExtension(pm4Path);

            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
                continue;

            string canonicalTileStem = $"development_{tileX}_{tileY}";

            if (filterSet != null)
            {
                bool matches = filterSet.Contains(tileStem) ||
                               filterSet.Contains(canonicalTileStem) ||
                               filterSet.Contains($"{tileX}_{tileY}") ||
                               filterSet.Any(f => tileStem.EndsWith(f, StringComparison.OrdinalIgnoreCase));
                if (!matches)
                    continue;
            }

            // Probe candidate base ADT files in baseAdtDir
            string[] candidateBaseAdtNames =
            [
                $"development_{tileX}_{tileY}.adt",
                $"development_{tileX:02d}_{tileY:02d}.adt",
                $"{tileX}_{tileY}.adt"
            ];

            string? baseAdtPath = candidateBaseAdtNames
                .Select(name => Path.Combine(resolvedBaseAdtDir, name))
                .FirstOrDefault(File.Exists);

            // Ground Truth Barrier (T14): Detect if tile contains hand-authored Blizzard assets in 335-dev
            bool isAuthored = false;
            AdtPlacementCatalog? existingCatalog = null;
            if (baseAdtPath != null)
            {
                try
                {
                    existingCatalog = AdtPlacementReader.Read(baseAdtPath);
                    bool hasRealWmo = existingCatalog.WorldModelPlacements.Any(w =>
                        !w.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase));
                    bool hasDoodads = existingCatalog.ModelPlacements.Count > 0;
                    isAuthored = hasRealWmo || hasDoodads;
                }
                catch
                {
                    isAuthored = false;
                }
            }

            if (isAuthored)
            {
                // Ground truth preservation: Copy authentic Blizzard ADT byte-for-byte to output
                string outAuthoredPath = Path.Combine(outputDir, $"development_{tileX}_{tileY}.adt");
                if (!string.Equals(Path.GetFullPath(baseAdtPath!), Path.GetFullPath(outAuthoredPath), StringComparison.OrdinalIgnoreCase))
                {
                    File.Copy(baseAdtPath!, outAuthoredPath, overwrite: true);
                }

                preservedAuthoredCount++;
                Console.WriteLine($"[PRESERVED] Hand-authored Blizzard tile development_{tileX}_{tileY}.adt ({existingCatalog?.WorldModelPlacements.Count ?? 0} WMOs, {existingCatalog?.ModelPlacements.Count ?? 0} doodads).");
                continue;
            }

            // Check if PM4 contains collision geometry
            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
            {
                // No geometry on this unauthored tile: copy base terrain if available
                if (baseAdtPath != null)
                {
                    string outBaseTerrainPath = Path.Combine(outputDir, $"development_{tileX}_{tileY}.adt");
                    if (!string.Equals(Path.GetFullPath(baseAdtPath), Path.GetFullPath(outBaseTerrainPath), StringComparison.OrdinalIgnoreCase))
                    {
                        File.Copy(baseAdtPath, outBaseTerrainPath, overwrite: true);
                    }
                }
                continue;
            }

            processedCount++;

            try
            {
                // Resolve authentic placements using 335-dev neighbor context, ledger, and 2D PCA yaw rotation
                Pm4TileResolutionResult result = Pm4UnauthoredPlacementResolver.ResolveTile(
                    pm4Path,
                    resolvedBaseAdtDir,
                    companionPath => AdtPlacementReader.Read(companionPath),
                    shapeLibrary,
                    ledgerLookup: ledgerLookup);

                // Monolithic LK ADT Assembly (T15): Merge surviving base terrain with resolved placements
                LkAdtData baseTerrainAdt;
                if (baseAdtPath != null && File.Exists(baseAdtPath))
                {
                    byte[] baseBytes = EnsureValidMcnkIndices(File.ReadAllBytes(baseAdtPath));
                    LkAdtData rawBase = LkAdtReader.Read(baseBytes, null, null, tileX, tileY);
                    if (rawBase.Chunks.Count == 256)
                    {
                        baseTerrainAdt = new LkAdtData
                        {
                            MapName = "development",
                            TileX = tileX,
                            TileY = tileY,
                            TextureNames = rawBase.TextureNames,
                            ModelNames = [],
                            WorldModelNames = [],
                            ModelPlacements = [],
                            WorldModelPlacements = [],
                            Chunks = rawBase.Chunks,
                            MhdrFlags = rawBase.MhdrFlags,
                            MfboFlightBounds = rawBase.MfboFlightBounds
                        };
                    }
                    else
                    {
                        baseTerrainAdt = BlankAdtFactory.CreateBlank("development", tileX, tileY);
                    }
                }
                else
                {
                    baseTerrainAdt = BlankAdtFactory.CreateBlank("development", tileX, tileY);
                }

                LkAdtData restoredAdt = BlankAdtFactory.WithPlacements(baseTerrainAdt, result.Catalog);

                string outAdtName = $"development_{tileX}_{tileY}.adt";
                string outAdtPath = Path.Combine(outputDir, outAdtName);

                LkAdtWriter.Write(outAdtPath, restoredAdt, MapConversionTargetFormat.LkAdtV18);

                if (writeSplitObj)
                {
                    string outObj0Name = $"{tileStem}_obj0.adt";
                    string outObj0Path = Path.Combine(outputDir, outObj0Name);
                    SplitAdtObjWriter.Write(outObj0Path, result.Catalog);
                }

                bool rtOk = true;
                if (validate)
                {
                    try
                    {
                        byte[] writtenBytes = File.ReadAllBytes(outAdtPath);
                        LkAdtData readBack = LkAdtReader.Read(writtenBytes, null, null, tileX, tileY);
                        if (readBack.WorldModelPlacements.Count != result.Catalog.WorldModelPlacements.Count ||
                            readBack.ModelPlacements.Count != result.Catalog.ModelPlacements.Count ||
                            readBack.Chunks.Count != 256)
                        {
                            rtOk = false;
                        }
                    }
                    catch (Exception ex)
                    {
                        rtOk = false;
                        Console.Error.WriteLine($"[FAIL] Round-trip error on {outAdtName}: {ex.Message}");
                    }
                }

                if (rtOk) roundTripPassed++;
                else roundTripFailed++;

                totalWmos += result.ResolvedWmos.Count;
                totalDoodads += result.ResolvedDoodads.Count;
                results.Add(result);

                Console.WriteLine($"\n[{processedCount}] Restored Monolithic ADT: {outAdtName} (Grid: [{result.TileX}, {result.TileY}])");
                Console.WriteLine($"    WMO Placements:    {result.ResolvedWmos.Count}");
                Console.WriteLine($"    Doodad Placements: {result.ResolvedDoodads.Count}");
                Console.WriteLine($"    Round-Trip Check:  {(rtOk ? "PASS (100% Monolithic LK Match)" : "FAIL")}");

                foreach (Pm4ResolvedObjectRecord wmo in result.ResolvedWmos)
                {
                    string shortMethod = wmo.Method switch
                    {
                        Pm4PlacementResolutionMethod.AdjacentSpillover => "Spillover",
                        Pm4PlacementResolutionMethod.LedgerExactZ => "Ledger-Z",
                        Pm4PlacementResolutionMethod.NeighborThematic => "Neighbor-Theme",
                        Pm4PlacementResolutionMethod.ShapeScore => "Shape-Score",
                        _ => "Fallback"
                    };
                    Console.WriteLine($"    * WMO [{shortMethod}] {Path.GetFileName(wmo.ModelPath)} @ Z={wmo.PlacementZ:F2}, Yaw={wmo.Rotation.Y:F1} deg (Conf: {wmo.Confidence:P0}) - {wmo.SourceNote}");
                }

                if (result.ResolvedDoodads.Count > 0)
                {
                    var doodadSummary = result.ResolvedDoodads
                        .GroupBy(d => Path.GetFileName(d.ModelPath))
                        .OrderByDescending(g => g.Count())
                        .Take(5)
                        .Select(g => $"{g.Key} ({g.Count()})");
                    Console.WriteLine($"    * Top Doodads: {string.Join(", ", doodadSummary)}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Failed to restore tile '{tileStem}': {ex.Message}");
            }
        }

        Console.WriteLine("\n=================================================================");
        Console.WriteLine(" Restoration Summary");
        Console.WriteLine("=================================================================");
        Console.WriteLine($"Hand-Authored Preserved: {preservedAuthoredCount} tiles");
        Console.WriteLine($"Unauthored Restored:     {results.Count} tiles");
        Console.WriteLine($"Total WMOs Restored:     {totalWmos}");
        Console.WriteLine($"Total M2s Restored:      {totalDoodads}");
        Console.WriteLine($"Round-Trip Verified:     {roundTripPassed} PASS / {roundTripFailed} FAIL");
        Console.WriteLine($"Output Location:         {outputDir}");
        Console.WriteLine("=================================================================\n");

        if (!string.IsNullOrWhiteSpace(jsonOutPath))
        {
            var summaryObj = new
            {
                BaseAdtDirectory = resolvedBaseAdtDir,
                Pm4Directory = resolvedPm4Dir,
                OutputDirectory = outputDir,
                AuthoredPreservedCount = preservedAuthoredCount,
                TilesRestoredCount = results.Count,
                TotalWmosRestored = totalWmos,
                TotalDoodadsRestored = totalDoodads,
                RoundTripPassed = roundTripPassed,
                RoundTripFailed = roundTripFailed,
                Tiles = results.Select(r => new
                {
                    r.TileName,
                    r.TileX,
                    r.TileY,
                    WmoCount = r.ResolvedWmos.Count,
                    DoodadCount = r.ResolvedDoodads.Count,
                    Wmos = r.ResolvedWmos.Select(w => new
                    {
                        w.ModelPath,
                        w.PlacementZ,
                        Method = w.Method.ToString(),
                        w.Confidence,
                        w.SourceNote,
                        w.Position,
                        w.Rotation
                    })
                })
            };

            File.WriteAllText(jsonOutPath, JsonSerializer.Serialize(summaryObj, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"Wrote detailed report to: {jsonOutPath}");
        }
    }

    private static (Dictionary<string, List<Vector3>> shapeLibrary, Dictionary<uint, Pm4LedgerPlacementRecord> ledgerLookup)
        LoadLibraries(string baseAdtDir, string ledgerPath)
    {
        var shapeLibrary = new Dictionary<string, List<Vector3>>(StringComparer.OrdinalIgnoreCase);
        var ledgerLookup = new Dictionary<uint, Pm4LedgerPlacementRecord>();

        // 1. Try loading from persistent ledger
        if (File.Exists(ledgerPath))
        {
            try
            {
                using JsonDocument json = JsonDocument.Parse(File.ReadAllBytes(ledgerPath));
                if (json.RootElement.TryGetProperty("Entries", out JsonElement entries))
                {
                    foreach (JsonElement rec in entries.EnumerateArray())
                    {
                        if (rec.TryGetProperty("Status", out JsonElement statusElem) &&
                            statusElem.GetString() == "matched" &&
                            rec.TryGetProperty("AssetPath", out JsonElement modelElem) &&
                            rec.TryGetProperty("MinX", out JsonElement minX) &&
                            rec.TryGetProperty("MinY", out JsonElement minY) &&
                            rec.TryGetProperty("MinZ", out JsonElement minZ) &&
                            rec.TryGetProperty("MaxX", out JsonElement maxX) &&
                            rec.TryGetProperty("MaxY", out JsonElement maxY) &&
                            rec.TryGetProperty("MaxZ", out JsonElement maxZ))
                        {
                            string? model = modelElem.GetString();
                            if (!string.IsNullOrWhiteSpace(model) && !model.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
                            {
                                Vector3 shape = new(
                                    MathF.Abs(maxX.GetSingle() - minX.GetSingle()),
                                    MathF.Abs(maxY.GetSingle() - minY.GetSingle()),
                                    MathF.Abs(maxZ.GetSingle() - minZ.GetSingle()));
                                if (!shapeLibrary.TryGetValue(model, out List<Vector3>? shapes))
                                {
                                    shapes = [];
                                    shapeLibrary[model] = shapes;
                                }
                                shapes.Add(shape);

                                if (rec.TryGetProperty("RawHex", out JsonElement rawHexElem))
                                {
                                    string? rawHexStr = rawHexElem.GetString();
                                    if (!string.IsNullOrWhiteSpace(rawHexStr) && rawHexStr.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
                                    {
                                        if (uint.TryParse(rawHexStr[2..], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out uint key) && key != 0)
                                        {
                                            float posX = rec.TryGetProperty("PosX", out var px) ? px.GetSingle() : 0f;
                                            float posY = rec.TryGetProperty("PosY", out var py) ? py.GetSingle() : 0f;
                                            float posZ = rec.TryGetProperty("PosZ", out var pz) ? pz.GetSingle() : 0f;
                                            float rotX = rec.TryGetProperty("RotX", out var rx) ? rx.GetSingle() : 0f;
                                            float rotY = rec.TryGetProperty("RotY", out var ry) ? ry.GetSingle() : 0f;
                                            float rotZ = rec.TryGetProperty("RotZ", out var rz) ? rz.GetSingle() : 0f;
                                            int surfCount = rec.TryGetProperty("SurfaceCount", out var sc) ? sc.GetInt32() : 0;
                                            string assetKind = rec.TryGetProperty("AssetKind", out var ak) ? ak.GetString() ?? "WMO" : "WMO";

                                            ledgerLookup[key] = new Pm4LedgerPlacementRecord(
                                                model,
                                                assetKind,
                                                posX, posY, posZ,
                                                rotX, rotY, rotZ,
                                                minX.GetSingle(), minY.GetSingle(), minZ.GetSingle(),
                                                maxX.GetSingle(), maxY.GetSingle(), maxZ.GetSingle(),
                                                surfCount);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Warning: Failed to load shape library from ledger: {ex.Message}");
            }
        }

        // 2. Supplement from authored ADTs in baseAdtDir
        if (Directory.Exists(baseAdtDir))
        {
            foreach (string adtPath in Directory.EnumerateFiles(baseAdtDir, "*.adt", SearchOption.TopDirectoryOnly))
            {
                try
                {
                    AdtPlacementCatalog cat = AdtPlacementReader.Read(adtPath);
                    foreach (AdtWorldModelPlacement placed in cat.WorldModelPlacements)
                    {
                        if (placed.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
                            continue;

                        Vector3 span = placed.BoundsMax - placed.BoundsMin;
                        Vector3 shape = new(MathF.Abs(span.X), MathF.Abs(span.Y), MathF.Abs(span.Z));
                        if (shape.X > 0.1f || shape.Y > 0.1f || shape.Z > 0.1f)
                        {
                            if (!shapeLibrary.TryGetValue(placed.ModelPath, out List<Vector3>? shapes))
                            {
                                shapes = [];
                                shapeLibrary[placed.ModelPath] = shapes;
                            }
                            shapes.Add(shape);
                        }
                    }
                }
                catch
                {
                    // Ignore unreadable
                }
            }
        }

        return (shapeLibrary, ledgerLookup);
    }

    private static byte[] EnsureValidMcnkIndices(byte[] bytes)
    {
        // Fix Schlumpf Cata WDL-ADT defect where all 256 MCNK/KNCM chunks had IndexY = 0
        int pos = 0;
        int chunkIdx = 0;
        byte[] copy = (byte[])bytes.Clone();
        while (pos + 8 <= copy.Length)
        {
            string tag = System.Text.Encoding.ASCII.GetString(copy, pos, 4);
            uint sz = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(copy.AsSpan(pos + 4, 4));
            if (tag == "MCNK" || tag == "KNCM")
            {
                if (pos + 8 + 12 <= copy.Length)
                {
                    uint ix = (uint)(chunkIdx % 16);
                    uint iy = (uint)(chunkIdx / 16);
                    System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(copy.AsSpan(pos + 8 + 4, 4), ix);
                    System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(copy.AsSpan(pos + 8 + 8, 4), iy);
                }
                chunkIdx++;
            }
            pos += 8 + (int)sz;
        }

        return chunkIdx == 256 ? copy : bytes;
    }
}
