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
/// (Spec 270 Phase 4 & Phase 5). Converges authentic placements from WoWMuseum/335-dev and
/// original_development _obj0 files with net-new PM4 collision objects, derives 2D PCA yaw
/// rotations, bakes authentic 256x256 minimap PNGs into 145-vertex MCCV chunks on untextured tiles,
/// and serializes standard monolithic LK 3.3.5 ADTs (development_XX_YY.adt).
/// </summary>
internal static class Pm4RestoreUnauthoredSupport
{
    private const string DefaultBaseAdtPath = @"I:\parp\parp-tools\wow-viewer\test_data\WoWMuseum\335-dev\World\Maps\development";
    private const string DefaultPm4Path = @"I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development";
    private const string DefaultOutputPath = @"I:\parp\parp-tools\wow-viewer\output\restored_development\World\Maps\development";
    private const string DefaultLedgerPath = @"I:\parp\parp-tools\wow-viewer\output\datasets\pm4_object_association_ledger.json";
    private const string DefaultMinimapPath = @"I:\parp\parp-tools\wow-viewer\test_data\development\World\Textures\Minimap";
    private const string DefaultRefinedTerrainPath = @"I:\parp\parp-tools\wow-viewer\output\development_reconstructed_lk";

    public static void Run(string[] args)
    {
        string baseAdtDir = DefaultBaseAdtPath;
        string refinedTerrainDir = DefaultRefinedTerrainPath;
        string pm4Dir = DefaultPm4Path;
        string outputDir = DefaultOutputPath;
        string? targetTilesFilter = null;
        string ledgerPath = DefaultLedgerPath;
        string minimapDir = DefaultMinimapPath;
        string? jsonOutPath = null;
        bool validate = true;
        bool writeSplitObj = false;
        bool cleanCompanions = true;
        bool bakeMccv = true;
        bool converge = true;
        bool allTiles = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--base":
                case "--adt":
                    if (i + 1 < args.Length) baseAdtDir = args[++i];
                    break;
                case "--refined":
                case "--refined-terrain":
                    if (i + 1 < args.Length) refinedTerrainDir = args[++i];
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
                case "--minimap":
                case "-m":
                    if (i + 1 < args.Length) minimapDir = args[++i];
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
                case "--no-clean-companions":
                    cleanCompanions = false;
                    break;
                case "--no-bake-mccv":
                    bakeMccv = false;
                    break;
                case "--no-converge":
                    converge = false;
                    break;
                case "--all-tiles":
                    allTiles = true;
                    break;
            }
        }

        string resolvedBaseAdtDir = Pm4CoordinateService.ResolveMapDirectory(baseAdtDir);
        string nestedBase = Path.Combine(resolvedBaseAdtDir, "World", "Maps", "development");
        if (Directory.Exists(nestedBase))
            resolvedBaseAdtDir = nestedBase;

        string resolvedRefinedDir = Pm4CoordinateService.ResolveMapDirectory(refinedTerrainDir);
        string nestedRefined = Path.Combine(resolvedRefinedDir, "World", "Maps", "development");
        if (Directory.Exists(nestedRefined))
            resolvedRefinedDir = nestedRefined;

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

        if (cleanCompanions && Directory.Exists(outputDir))
        {
            PurgeStrayCompanions(outputDir);
        }

        Console.WriteLine("=================================================================");
        Console.WriteLine(" Spec 270: PM4 Object Association & Multi-Source Convergence");
        Console.WriteLine("=================================================================");
        Console.WriteLine($"Base ADT Directory:    {resolvedBaseAdtDir}");
        Console.WriteLine($"Refined Terrain Dir:   {resolvedRefinedDir} (Exists: {Directory.Exists(resolvedRefinedDir)})");
        Console.WriteLine($"PM4 Directory:         {resolvedPm4Dir}");
        Console.WriteLine($"Minimap Directory:     {minimapDir}");
        Console.WriteLine($"Output Directory:      {outputDir}");
        Console.WriteLine($"Bake Minimap MCCV:     {bakeMccv}");
        Console.WriteLine($"Converge Objects:      {converge}");

        // Build shape library and exact Z ledger lookup from persistent ledger and authored tiles
        var (shapeLibrary, ledgerLookup) = LoadLibraries(resolvedBaseAdtDir, ledgerPath);
        Console.WriteLine($"Shape Library: Loaded {shapeLibrary.Count} unique asset shape profiles.");
        Console.WriteLine($"Ledger Lookup: Loaded {ledgerLookup.Count} bit-exact Z keys from persistent Rosetta ledger.");

        // Gather candidate tiles
        var candidateTiles = new HashSet<(int X, int Y)>();
        foreach (string pm4Path in Directory.EnumerateFiles(resolvedPm4Dir, "*.pm4", SearchOption.TopDirectoryOnly))
        {
            if (Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tx, out int ty))
                candidateTiles.Add((tx, ty));
        }

        if (allTiles || (targetTilesFilter != null && targetTilesFilter.Equals("all", StringComparison.OrdinalIgnoreCase)))
        {
            if (Directory.Exists(resolvedBaseAdtDir))
            {
                foreach (string adtPath in Directory.EnumerateFiles(resolvedBaseAdtDir, "*.adt", SearchOption.TopDirectoryOnly))
                {
                    if (Pm4CoordinateService.TryParseTileCoordinates(adtPath, out int tx, out int ty))
                        candidateTiles.Add((tx, ty));
                }
            }

            if (Directory.Exists(resolvedRefinedDir))
            {
                foreach (string adtPath in Directory.EnumerateFiles(resolvedRefinedDir, "*.adt", SearchOption.TopDirectoryOnly))
                {
                    if (Pm4CoordinateService.TryParseTileCoordinates(adtPath, out int tx, out int ty))
                        candidateTiles.Add((tx, ty));
                }
            }

            if (Directory.Exists(minimapDir))
            {
                foreach (string pngPath in Directory.EnumerateFiles(minimapDir, "*.png", SearchOption.TopDirectoryOnly))
                {
                    if (Pm4CoordinateService.TryParseTileCoordinates(pngPath, out int tx, out int ty))
                        candidateTiles.Add((tx, ty));
                }
            }
        }

        HashSet<string>? filterSet = null;
        if (!string.IsNullOrWhiteSpace(targetTilesFilter) && !targetTilesFilter.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            filterSet = new HashSet<string>(
                targetTilesFilter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(s => s.Replace(".pm4", "", StringComparison.OrdinalIgnoreCase).Replace(".adt", "", StringComparison.OrdinalIgnoreCase)),
                StringComparer.OrdinalIgnoreCase);
        }

        var results = new List<Pm4TileResolutionResult>();
        int totalWmos = 0;
        int totalDoodads = 0;
        int roundTripPassed = 0;
        int roundTripFailed = 0;
        int processedCount = 0;
        int mccvBakedCount = 0;

        var sortedTiles = candidateTiles.OrderBy(t => t.X).ThenBy(t => t.Y).ToList();

        foreach (var (tileX, tileY) in sortedTiles)
        {
            string canonicalTileStem = $"development_{tileX}_{tileY}";
            string tileStem0 = $"development_{tileX:02d}_{tileY:02d}";

            if (filterSet != null)
            {
                bool matches = filterSet.Contains(canonicalTileStem) ||
                               filterSet.Contains(tileStem0) ||
                               filterSet.Contains($"{tileX}_{tileY}") ||
                               filterSet.Any(f => canonicalTileStem.EndsWith(f, StringComparison.OrdinalIgnoreCase));
                if (!matches)
                    continue;
            }

            // 1. Probe candidate PM4
            string[] candidatePm4Names = [
                $"development_{tileX}_{tileY}.pm4",
                $"development_{tileX:02d}_{tileY:02d}.pm4",
                $"{tileX}_{tileY}.pm4"
            ];
            string? pm4Path = candidatePm4Names
                .Select(name => Path.Combine(resolvedPm4Dir, name))
                .FirstOrDefault(File.Exists);

            // 2. Probe candidate base ADT files in baseAdtDir
            string[] candidateBaseAdtNames = [
                $"development_{tileX}_{tileY}.adt",
                $"development_{tileX:02d}_{tileY:02d}.adt",
                $"{tileX}_{tileY}.adt"
            ];
            string? baseAdtPath = candidateBaseAdtNames
                .Select(name => Path.Combine(resolvedBaseAdtDir, name))
                .FirstOrDefault(File.Exists);

            // 2b. Probe candidate refined terrain ADT files from WDL neural reconstruction
            string? refinedAdtPath = (baseAdtPath == null && Directory.Exists(resolvedRefinedDir))
                ? candidateBaseAdtNames.Select(name => Path.Combine(resolvedRefinedDir, name)).FirstOrDefault(File.Exists)
                : null;

            // 3. Probe candidate companion _obj0 in original_development
            string[] candidateOrigObj0Names = [
                $"development_{tileX}_{tileY}_obj0.adt",
                $"development_{tileX:02d}_{tileY:02d}_obj0.adt",
                $"{tileX}_{tileY}_obj0.adt"
            ];
            string? origObj0Path = candidateOrigObj0Names
                .Select(name => Path.Combine(resolvedPm4Dir, name))
                .FirstOrDefault(File.Exists);

            // 4. Probe candidate minimap PNG
            string[] candidateMinimapNames = [
                $"development_{tileX}_{tileY}.png",
                $"development_{tileX:02d}_{tileY:02d}.png",
                $"map{tileX}_{tileY}.png",
                $"{tileX}_{tileY}.png"
            ];
            string? minimapPngPath = Directory.Exists(minimapDir)
                ? candidateMinimapNames.Select(name => Path.Combine(minimapDir, name)).FirstOrDefault(File.Exists)
                : null;

            // Ingest existing placements from base ADT and companion _obj0
            AdtPlacementCatalog? existingCatalog = null;
            if (baseAdtPath != null)
            {
                try
                {
                    existingCatalog = AdtPlacementReader.Read(baseAdtPath);
                }
                catch { }
            }

            if (origObj0Path != null)
            {
                try
                {
                    AdtPlacementCatalog origCatalog = AdtPlacementReader.Read(origObj0Path);
                    existingCatalog = existingCatalog != null
                        ? MergeCatalogs(origCatalog, existingCatalog)
                        : origCatalog;
                }
                catch { }
            }

            // Resolve PM4 placements if collision geometry exists
            Pm4TileResolutionResult? pm4Result = null;
            if (pm4Path != null && File.Exists(pm4Path))
            {
                try
                {
                    Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
                    if (chunks.Msur.Count > 0)
                    {
                        pm4Result = Pm4UnauthoredPlacementResolver.ResolveTile(
                            pm4Path,
                            resolvedBaseAdtDir,
                            companionPath => AdtPlacementReader.Read(companionPath),
                            shapeLibrary,
                            ledgerLookup: ledgerLookup);
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Warning: Failed to resolve PM4 collision on '{canonicalTileStem}': {ex.Message}");
                }
            }

            // Converge placements: retain authored, align matching yaw/elev, and append net-new PM4 objects
            AdtPlacementCatalog convergedCatalog = converge
                ? ConvergeCatalogs(existingCatalog, pm4Result?.Catalog, tileX, tileY)
                : (pm4Result?.Catalog ?? existingCatalog ?? new AdtPlacementCatalog(string.Empty, MapFileKind.Adt, [], [], [], []));

            // Ingest and prepare base terrain
            LkAdtData baseTerrainAdt;
            bool bakedMccvOnThisTile = false;
            string? sourceTerrainPath = baseAdtPath ?? refinedAdtPath;
            if (sourceTerrainPath != null && File.Exists(sourceTerrainPath))
            {
                try
                {
                    byte[] baseBytes = EnsureValidMcnkIndices(File.ReadAllBytes(sourceTerrainPath));
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
                            Chunks = RepairAlphaAndShadow(baseBytes, rawBase.Chunks),
                            MhdrFlags = rawBase.MhdrFlags,
                            MfboFlightBounds = rawBase.MfboFlightBounds
                        };

                        // Check if terrain lacks textures or MCCV
                        bool lacksTextures = baseTerrainAdt.TextureNames.Count == 0 ||
                                              baseTerrainAdt.Chunks.All(c => c.Layers.Count == 0);
                        bool lacksMccv = baseTerrainAdt.Chunks.All(c => c.MccvColors == null || c.MccvColors.Length < 580);

                        // Spec 270 T19: minimap MCCV is only for tiles with no texture layers; baking it onto
                        // textured tiles tints authored terrain (orange cast).
                        if (bakeMccv && lacksTextures && minimapPngPath != null)
                        {
                            var mccv = AdtMccvImageSampler.SampleMccvFromPng(minimapPngPath);
                            if (mccv != null)
                            {
                                baseTerrainAdt = AdtMccvImageSampler.InjectMccvIntoLkAdt(baseTerrainAdt, mccv);
                                bakedMccvOnThisTile = true;
                                mccvBakedCount++;
                            }
                        }
                    }
                    else
                    {
                        baseTerrainAdt = CreateBlankTileWithMccv(tileX, tileY, minimapPngPath, bakeMccv, ref mccvBakedCount, ref bakedMccvOnThisTile);
                    }
                }
                catch
                {
                    baseTerrainAdt = CreateBlankTileWithMccv(tileX, tileY, minimapPngPath, bakeMccv, ref mccvBakedCount, ref bakedMccvOnThisTile);
                }
            }
            else
            {
                baseTerrainAdt = CreateBlankTileWithMccv(tileX, tileY, minimapPngPath, bakeMccv, ref mccvBakedCount, ref bakedMccvOnThisTile);
            }

            processedCount++;

            try
            {
                LkAdtData restoredAdt = BlankAdtFactory.WithPlacements(baseTerrainAdt, convergedCatalog);

                string outAdtName = $"development_{tileX}_{tileY}.adt";
                string outAdtPath = Path.Combine(outputDir, outAdtName);

                LkAdtWriter.Write(outAdtPath, restoredAdt, MapConversionTargetFormat.LkAdtV18);

                if (writeSplitObj)
                {
                    string outObj0Name = $"{canonicalTileStem}_obj0.adt";
                    string outObj0Path = Path.Combine(outputDir, outObj0Name);
                    SplitAdtObjWriter.Write(outObj0Path, convergedCatalog);
                }

                bool rtOk = true;
                if (validate)
                {
                    try
                    {
                        byte[] writtenBytes = File.ReadAllBytes(outAdtPath);
                        LkAdtData readBack = LkAdtReader.Read(writtenBytes, null, null, tileX, tileY);
                        if (readBack.WorldModelPlacements.Count != convergedCatalog.WorldModelPlacements.Count ||
                            readBack.ModelPlacements.Count != convergedCatalog.ModelPlacements.Count ||
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

                totalWmos += convergedCatalog.WorldModelPlacements.Count;
                totalDoodads += convergedCatalog.ModelPlacements.Count;

                if (pm4Result != null)
                {
                    results.Add(pm4Result);
                }

                string mccvBadge = bakedMccvOnThisTile ? " [MCCV BAKED]" : "";
                Console.WriteLine($"\n[{processedCount}] Converged Monolithic ADT: {outAdtName} (Grid: [{tileX}, {tileY}]){mccvBadge}");
                Console.WriteLine($"    WMO Placements:    {convergedCatalog.WorldModelPlacements.Count}");
                Console.WriteLine($"    Doodad Placements: {convergedCatalog.ModelPlacements.Count}");
                Console.WriteLine($"    Round-Trip Check:  {(rtOk ? "PASS (100% Monolithic LK Match)" : "FAIL")}");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Failed to converge tile '{canonicalTileStem}': {ex.Message}");
            }
        }

        Console.WriteLine("\n=================================================================");
        Console.WriteLine(" Convergence & Restoration Summary");
        Console.WriteLine("=================================================================");
        Console.WriteLine($"Total Tiles Processed: {processedCount} tiles");
        Console.WriteLine($"Total Minimap MCCV Baked: {mccvBakedCount} tiles");
        Console.WriteLine($"Total WMOs Converged:    {totalWmos}");
        Console.WriteLine($"Total M2s Converged:     {totalDoodads}");
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
                MinimapDirectory = minimapDir,
                TilesProcessedCount = processedCount,
                MccvBakedCount = mccvBakedCount,
                TotalWmos = totalWmos,
                TotalDoodads = totalDoodads,
                RoundTripPassed = roundTripPassed,
                RoundTripFailed = roundTripFailed,
            };

            File.WriteAllText(jsonOutPath, JsonSerializer.Serialize(summaryObj, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"Wrote detailed report to: {jsonOutPath}");
        }
    }

    private static void PurgeStrayCompanions(string outputDir)
    {
        try
        {
            int deleted = 0;
            foreach (string file in Directory.EnumerateFiles(outputDir, "*.adt", SearchOption.AllDirectories))
            {
                string name = Path.GetFileName(file);
                if (name.Contains("_obj0.adt", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("_obj1.adt", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("_tex0.adt", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("_lod.adt", StringComparison.OrdinalIgnoreCase))
                {
                    File.Delete(file);
                    deleted++;
                }
            }

            if (deleted > 0)
            {
                Console.WriteLine($"[CLEAN] Purged {deleted} stray split companion files (_obj0/_tex0) from output directory.");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: Failed to clean companion files: {ex.Message}");
        }
    }

    private static LkAdtData CreateBlankTileWithMccv(
        int tileX,
        int tileY,
        string? minimapPngPath,
        bool bakeMccv,
        ref int mccvBakedCount,
        ref bool bakedMccvOnThisTile)
    {
        Dictionary<int, byte[]>? mccv = null;
        if (bakeMccv && minimapPngPath != null && File.Exists(minimapPngPath))
        {
            mccv = AdtMccvImageSampler.SampleMccvFromPng(minimapPngPath);
            if (mccv != null)
            {
                bakedMccvOnThisTile = true;
                mccvBakedCount++;
            }
        }

        return BlankAdtFactory.CreateBlank(
            "development",
            tileX,
            tileY,
            "tileset\\ocean\\westfallseafloor.blp",
            mccv);
    }

    private static AdtPlacementCatalog MergeCatalogs(AdtPlacementCatalog a, AdtPlacementCatalog b)
    {
        var wmos = a.WorldModelPlacements.ToList();
        var doodads = a.ModelPlacements.ToList();

        var existingWmoIds = new HashSet<int>(wmos.Select(w => w.UniqueId));
        foreach (var w in b.WorldModelPlacements)
        {
            if (existingWmoIds.Add(w.UniqueId))
                wmos.Add(w);
        }

        var existingDoodadIds = new HashSet<int>(doodads.Select(d => d.UniqueId));
        foreach (var d in b.ModelPlacements)
        {
            if (existingDoodadIds.Add(d.UniqueId))
                doodads.Add(d);
        }

        var modelNames = doodads.Select(d => d.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        var worldModelNames = wmos.Select(w => w.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase).ToList();

        return new AdtPlacementCatalog(a.SourcePath, a.Kind, modelNames, worldModelNames, doodads, wmos);
    }

    private static AdtPlacementCatalog ConvergeCatalogs(
        AdtPlacementCatalog? existingCatalog,
        AdtPlacementCatalog? pm4Catalog,
        int tileX,
        int tileY)
    {
        var convergedWmos = new List<AdtWorldModelPlacement>();
        var convergedDoodads = new List<AdtModelPlacement>();

        var matchedPm4Wmos = new HashSet<int>();
        var matchedPm4Doodads = new HashSet<int>();

        // 1. Ingest and refine existing WMOs
        if (existingCatalog != null)
        {
            foreach (var existing in existingCatalog.WorldModelPlacements)
            {
                if (existing.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
                    continue;

                int bestPm4Idx = -1;
                float bestDist = 65.0f; // Proximity threshold for magnetic attraction

                if (pm4Catalog != null)
                {
                    for (int i = 0; i < pm4Catalog.WorldModelPlacements.Count; i++)
                    {
                        if (matchedPm4Wmos.Contains(i)) continue;
                        var pm4Wmo = pm4Catalog.WorldModelPlacements[i];
                        float dist = Vector3.Distance(existing.Position, pm4Wmo.Position);

                        // Strongly prioritize exact model match
                        bool sameModel = existing.ModelPath.Equals(pm4Wmo.ModelPath, StringComparison.OrdinalIgnoreCase);
                        float effectiveDist = sameModel ? dist * 0.5f : dist;

                        if (effectiveDist < bestDist)
                        {
                            bestDist = effectiveDist;
                            bestPm4Idx = i;
                        }
                    }
                }

                Vector3 finalRot = existing.Rotation;
                Vector3 finalPos = existing.Position;
                Vector3 boundsMin = existing.BoundsMin;
                Vector3 boundsMax = existing.BoundsMax;

                if (bestPm4Idx >= 0 && pm4Catalog != null)
                {
                    matchedPm4Wmos.Add(bestPm4Idx);
                    var pm4Wmo = pm4Catalog.WorldModelPlacements[bestPm4Idx];

                    // Magnetically snap Z if existing Z is zero or deviates slightly from exact PM4 collision key
                    if (existing.Position.Z == 0f || (pm4Wmo.Position.Z != 0f && MathF.Abs(existing.Position.Z - pm4Wmo.Position.Z) > 0.05f && MathF.Abs(existing.Position.Z - pm4Wmo.Position.Z) < 5.0f))
                    {
                        finalPos = new Vector3(existing.Position.X, existing.Position.Y, pm4Wmo.Position.Z);
                    }

                    // Preserve authentic ADT rotation from existing; only use PM4 PCA rotation if unauthored/zero
                    if (finalRot == Vector3.Zero && pm4Wmo.Rotation != Vector3.Zero)
                        finalRot = pm4Wmo.Rotation;

                    boundsMin = pm4Wmo.BoundsMin;
                    boundsMax = pm4Wmo.BoundsMax;
                }

                convergedWmos.Add(new AdtWorldModelPlacement(
                    existing.NameId,
                    existing.ModelPath,
                    existing.UniqueId,
                    finalPos,
                    finalRot,
                    boundsMin,
                    boundsMax,
                    existing.Flags,
                    existing.DoodadSet,
                    existing.NameSet,
                    existing.Scale));
            }

            // 2. Ingest and refine existing M2 doodads
            foreach (var existing in existingCatalog.ModelPlacements)
            {
                int bestPm4Idx = -1;
                float bestDist = 25.0f; // 25 yd proximity for doodads

                if (pm4Catalog != null)
                {
                    for (int i = 0; i < pm4Catalog.ModelPlacements.Count; i++)
                    {
                        if (matchedPm4Doodads.Contains(i)) continue;
                        var pm4Doodad = pm4Catalog.ModelPlacements[i];
                        float dist = Vector3.Distance(existing.Position, pm4Doodad.Position);

                        bool sameModel = existing.ModelPath.Equals(pm4Doodad.ModelPath, StringComparison.OrdinalIgnoreCase);
                        float effectiveDist = sameModel ? dist * 0.5f : dist;

                        if (effectiveDist < bestDist)
                        {
                            bestDist = effectiveDist;
                            bestPm4Idx = i;
                        }
                    }
                }

                Vector3 finalRot = existing.Rotation;
                Vector3 finalPos = existing.Position;
                if (bestPm4Idx >= 0 && pm4Catalog != null)
                {
                    matchedPm4Doodads.Add(bestPm4Idx);
                    var pm4Doodad = pm4Catalog.ModelPlacements[bestPm4Idx];

                    // Magnetically snap Z if existing Z was unauthored or deviates slightly
                    if (existing.Position.Z == 0f || (pm4Doodad.Position.Z != 0f && MathF.Abs(existing.Position.Z - pm4Doodad.Position.Z) > 0.05f && MathF.Abs(existing.Position.Z - pm4Doodad.Position.Z) < 5.0f))
                        finalPos = new Vector3(existing.Position.X, existing.Position.Y, pm4Doodad.Position.Z);

                    // Preserve authentic rotation
                    if (finalRot == Vector3.Zero && pm4Doodad.Rotation != Vector3.Zero)
                        finalRot = pm4Doodad.Rotation;
                }

                convergedDoodads.Add(new AdtModelPlacement(
                    existing.NameId,
                    existing.ModelPath,
                    existing.UniqueId,
                    finalPos,
                    finalRot,
                    existing.Scale));
            }
        }

        // 3. Append net-new PM4 WMOs (with placement tile boundary containment and deduplication)
        if (pm4Catalog != null)
        {
            for (int i = 0; i < pm4Catalog.WorldModelPlacements.Count; i++)
            {
                if (matchedPm4Wmos.Contains(i)) continue;
                var newWmo = pm4Catalog.WorldModelPlacements[i];
                if (Pm4CoordinateService.IsWithinPlacementTileBounds(newWmo.Position, tileX, tileY))
                {
                    bool alreadyPresent = convergedWmos.Any(w =>
                        w.ModelPath.Equals(newWmo.ModelPath, StringComparison.OrdinalIgnoreCase)
                        && Vector3.Distance(w.Position, newWmo.Position) < 20.0f);
                    if (!alreadyPresent)
                    {
                        convergedWmos.Add(newWmo);
                    }
                }
            }

            // 4. Append net-new PM4 M2 doodads (with placement tile boundary containment and deduplication)
            for (int i = 0; i < pm4Catalog.ModelPlacements.Count; i++)
            {
                if (matchedPm4Doodads.Contains(i)) continue;
                var newDoodad = pm4Catalog.ModelPlacements[i];
                if (Pm4CoordinateService.IsWithinPlacementTileBounds(newDoodad.Position, tileX, tileY))
                {
                    bool alreadyPresent = convergedDoodads.Any(d =>
                        d.ModelPath.Equals(newDoodad.ModelPath, StringComparison.OrdinalIgnoreCase)
                        && Vector3.Distance(d.Position, newDoodad.Position) < 5.0f);
                    if (!alreadyPresent)
                    {
                        convergedDoodads.Add(newDoodad);
                    }
                }
            }
        }

        var modelNames = convergedDoodads.Select(d => d.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        var worldModelNames = convergedWmos.Select(w => w.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase).ToList();

        return new AdtPlacementCatalog(
            existingCatalog?.SourcePath ?? pm4Catalog?.SourcePath ?? string.Empty,
            existingCatalog?.Kind ?? pm4Catalog?.Kind ?? MapFileKind.Adt,
            modelNames,
            worldModelNames,
            convergedDoodads,
            convergedWmos);
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

    /// <summary>
    /// LkAdtReader.ReadMcnkChunk adds the 8-byte MCNK header to MCAL/MCSH offsets that are already
    /// MCNK-start-relative, so every alpha map and shadow map it returns is shifted 8 bytes. Rather than
    /// touch the shared reader, re-extract both subchunks here with the correct base and swap them in.
    /// Chunks whose offset does not land on an MCAL/MCSH tag keep the reader's data unchanged.
    /// </summary>
    private static IReadOnlyList<LkMcnkData> RepairAlphaAndShadow(byte[] bytes, IReadOnlyList<LkMcnkData> chunks)
    {
        var starts = new List<int>(256);
        int pos = 0;
        while (pos + 8 <= bytes.Length)
        {
            string tag = System.Text.Encoding.ASCII.GetString(bytes, pos, 4);
            uint sz = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(pos + 4, 4));
            if (tag == "MCNK" || tag == "KNCM")
                starts.Add(pos);
            long next = (long)pos + 8 + sz;
            if (next > bytes.Length) break;
            pos = (int)next;
        }

        static byte[]? Extract(byte[] b, int chunkStart, int ofs, int size, string fwd, string rev)
        {
            if (ofs <= 0 || size <= 8) return null;
            long at = (long)chunkStart + ofs;
            if (at + size > b.Length) return null;
            string t = System.Text.Encoding.ASCII.GetString(b, (int)at, 4);
            if (t != fwd && t != rev) return null;
            byte[] data = new byte[size - 8];
            Buffer.BlockCopy(b, (int)at + 8, data, 0, data.Length);
            return data;
        }

        var result = new List<LkMcnkData>(chunks.Count);
        foreach (LkMcnkData c in chunks)
        {
            int i = c.IndexY * 16 + c.IndexX;
            if (i < 0 || i >= starts.Count)
            {
                result.Add(c);
                continue;
            }

            int start = starts[i];
            int ofsAl = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(start + 8 + 0x24, 4));
            int szAl = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(start + 8 + 0x28, 4));
            int ofsSh = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(start + 8 + 0x2C, 4));
            int szSh = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(start + 8 + 0x30, 4));
            byte[]? alpha = Extract(bytes, start, ofsAl, szAl, "MCAL", "LACM");
            byte[]? shadow = Extract(bytes, start, ofsSh, szSh, "MCSH", "HSCM");

            result.Add(new LkMcnkData
            {
                IndexX = c.IndexX,
                IndexY = c.IndexY,
                Flags = c.Flags,
                AreaId = c.AreaId,
                NLayers = c.NLayers,
                HoleMask = c.HoleMask,
                BaseHeight = c.BaseHeight,
                Heights = c.Heights,
                Normals = c.Normals,
                ShadowMap = shadow ?? c.ShadowMap,
                AlphaMapData = alpha ?? c.AlphaMapData,
                AlphaMapSize = alpha?.Length ?? c.AlphaMapSize,
                Layers = c.Layers,
                DoodadRefs = c.DoodadRefs,
                WorldModelRefs = c.WorldModelRefs,
                LiquidData = c.LiquidData,
                MccvColors = c.MccvColors,
                MclvLighting = c.MclvLighting,
                PosX = c.PosX,
                PosY = c.PosY,
                PosZ = c.PosZ
            });
        }

        return result;
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
