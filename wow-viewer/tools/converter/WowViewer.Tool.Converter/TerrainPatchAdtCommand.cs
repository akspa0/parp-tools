using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Tool.Converter;

internal static class TerrainPatchAdtCommand
{
    private const int TileHeightmapSize = 257;
    private const float DefaultTileWorldSize = 533.33333f;
    private static readonly Regex TileNameRegex = new(@"^(?<map>.+)_(?<x>\d+)_(?<y>\d+)(?:_.+)?$", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public static void Run(string[] args)
    {
        string? inputAdtDir = GetOption(args, "--input-adt-dir", "-i");
        string? inferenceDir = GetOption(args, "--inference-dir", "-p") ?? GetOption(args, "--prediction-dir");
        string? outputDir = GetOption(args, "--output-dir", "-o");
        bool copyFamily = !HasFlag(args, "--no-copy-family");
        bool exportGlb = HasFlag(args, "--export-glb") && !HasFlag(args, "--no-export-glb");
        bool exportGuideTextures = !HasFlag(args, "--no-export-guide-textures");
        bool exportTextureSupervision = !HasFlag(args, "--no-export-texture-supervision");
        bool centerMesh = HasFlag(args, "--center-mesh");
        float tileWorldSize = GetSingleOption(args, "--tile-world-size") ?? DefaultTileWorldSize;
        float heightOffset = GetSingleOption(args, "--height-offset") ?? 0f;

        if (string.IsNullOrWhiteSpace(inputAdtDir) || string.IsNullOrWhiteSpace(inferenceDir) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: terrain-patch-adt requires --input-adt-dir <dir>, --inference-dir <dir>, and --output-dir <dir>.");
            Environment.ExitCode = 1;
            return;
        }

        string inputRoot = Path.GetFullPath(inputAdtDir);
        string inferenceRoot = Path.GetFullPath(inferenceDir);
        string outputRoot = Path.GetFullPath(outputDir);
        if (!Directory.Exists(inputRoot))
        {
            Console.Error.WriteLine($"Error: input ADT directory not found: {inputRoot}");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(inferenceRoot))
        {
            Console.Error.WriteLine($"Error: inference directory not found: {inferenceRoot}");
            Environment.ExitCode = 1;
            return;
        }

        Dictionary<string, string> rootAdts = IndexRootAdts(inputRoot);
        string[] summaryPaths = Directory.GetFiles(inferenceRoot, "inference_summary.json", SearchOption.AllDirectories);
        if (summaryPaths.Length == 0)
        {
            Console.Error.WriteLine($"Error: no inference_summary.json files were found under {inferenceRoot}.");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputRoot);
        List<TerrainPatchResult> results = [];
        HashSet<string> patchedTiles = new(StringComparer.OrdinalIgnoreCase);
        List<PendingTerrainPatch> pendingPatches = [];
        int patchedCount = 0;
        int copiedCount = 0;
        int skippedCount = 0;

        foreach ((string tileName, string inputAdtPath) in rootAdts)
        {
            string relativeRootPath = Path.GetRelativePath(inputRoot, inputAdtPath);
            string outputAdtPath = Path.Combine(outputRoot, relativeRootPath);
            CopyRootAdt(inputAdtPath, outputAdtPath);
            if (copyFamily)
                CopyTileFamily(inputAdtPath, outputAdtPath);

            results.Add(new TerrainPatchResult(null, tileName, outputAdtPath, false, null, null, null, null, null, [], null, true, null, null));
            copiedCount++;
        }

        foreach (string summaryPath in summaryPaths)
        {
            TerrainInferenceSummary? summary;
            try
            {
                summary = JsonSerializer.Deserialize<TerrainInferenceSummary>(File.ReadAllText(summaryPath), CreateJsonOptions());
            }
            catch (Exception ex)
            {
                results.Add(new TerrainPatchResult(summaryPath, null, null, false, null, null, null, null, null, [], $"Failed to parse inference summary: {ex.Message}", false, null, null));
                skippedCount++;
                continue;
            }

            if (summary is null || string.IsNullOrWhiteSpace(summary.PredictedHeight257Path))
            {
                results.Add(new TerrainPatchResult(summaryPath, null, null, false, null, null, null, null, null, [], "Summary is missing predicted_height_257_path.", false, null, null));
                skippedCount++;
                continue;
            }

            string? tileName = ResolveTileName(summary, summaryPath);
            if (string.IsNullOrWhiteSpace(tileName))
            {
                results.Add(new TerrainPatchResult(summaryPath, null, null, false, null, null, null, null, null, [], "Could not derive a tile name from the inference summary.", false, null, null));
                skippedCount++;
                continue;
            }

            if (!rootAdts.TryGetValue(tileName, out string? inputAdtPath))
            {
                results.Add(new TerrainPatchResult(summaryPath, tileName, null, false, null, null, null, null, null, [], "Matching root ADT was not found in the input directory.", false, null, null));
                skippedCount++;
                continue;
            }

            string predictedHeightPath = ResolveReferencedPath(summaryPath, summary.PredictedHeight257Path);
            if (!File.Exists(predictedHeightPath))
            {
                results.Add(new TerrainPatchResult(summaryPath, tileName, inputAdtPath, false, null, null, null, null, null, [], $"Predicted heightmap not found: {predictedHeightPath}", false, null, null));
                skippedCount++;
                continue;
            }

            try
            {
                float[] heightmap = NpyFloatArrayReader.ReadMatrix(predictedHeightPath, out int rows, out int cols);
                if (rows != TileHeightmapSize || cols != TileHeightmapSize)
                    throw new InvalidDataException($"Expected a {TileHeightmapSize}x{TileHeightmapSize} terrain heightmap, but found {rows}x{cols}.");

                if (!TryParseTileIdentity(tileName, out string mapName, out int tileX, out int tileY))
                    throw new InvalidDataException($"Could not parse tile coordinates from '{tileName}'.");

                string relativeRootPath = Path.GetRelativePath(inputRoot, inputAdtPath);
                string outputAdtPath = Path.Combine(outputRoot, relativeRootPath);
                pendingPatches.Add(new PendingTerrainPatch(
                    summaryPath,
                    tileName,
                    mapName,
                    tileX,
                    tileY,
                    inputAdtPath,
                    outputAdtPath,
                    ResolveTexturePath(summary, summaryPath),
                    ResolveSourceMinimapPath(summary, summaryPath),
                    heightmap,
                    heightmap.ToArray(),
                    []));
            }
            catch (Exception ex)
            {
                results.Add(new TerrainPatchResult(summaryPath, tileName, inputAdtPath, false, null, null, null, null, null, [], ex.Message, false, null, null));
                skippedCount++;
            }
        }

        if (pendingPatches.Count > 0)
        {
            Dictionary<(int TileX, int TileY), float[]> seamInputs = new();
            foreach (PendingTerrainPatch patch in pendingPatches)
                seamInputs.Add((patch.TileX, patch.TileY), patch.Heightmap);

            Dictionary<(int TileX, int TileY), float[]> neighborAnchors = LoadNeighborAnchors(rootAdts, pendingPatches);
            AdtHeightmapSeamStitcher.StitchSharedEdges(seamInputs);
            AdtHeightmapSeamStitcher.AnchorPredictedEdgesToNeighbors(seamInputs, neighborAnchors);

            Dictionary<(int TileX, int TileY), float[]> seamReferences = new(seamInputs);
            foreach (((int tileX, int tileY), float[] heightmap) in neighborAnchors)
                seamReferences[(tileX, tileY)] = heightmap;

            foreach (PendingTerrainPatch patch in pendingPatches)
            {
                patch.ReferenceHeightmaps.Clear();
                foreach (((int tileX, int tileY), float[] heightmap) in seamReferences)
                    patch.ReferenceHeightmaps[(tileX, tileY)] = heightmap;
            }
        }

        foreach (PendingTerrainPatch patch in pendingPatches)
        {
            try
            {
                AdtTerrainWriter.Write(patch.InputAdtPath, patch.OutputAdtPath, patch.Heightmap);
                AdtChunkChangeAudit chunkChangeAudit = AdtTerrainPatchAudit.AnalyzeChunkChanges(patch.InputAdtPath, patch.OutputAdtPath);
                AdtSeamAudit seamAudit = AdtTerrainPatchAudit.CreateSeamAudit(
                    (patch.TileX, patch.TileY),
                    patch.OriginalHeightmap,
                    patch.Heightmap,
                    patch.ReferenceHeightmaps);
                if (copyFamily)
                    CopyTileFamily(patch.InputAdtPath, patch.OutputAdtPath);

                TerrainMccvGuideTextureBuilder.TerrainMccvGuideOutputs? guideOutputs = null;
                if (exportGuideTextures)
                    guideOutputs = TerrainMccvGuideTextureBuilder.TryWriteOutputs(patch.OutputAdtPath, patch.TileName, patch.SourceMinimapPath);

                AdtTextureTrainingSupervisionExporter.AdtTextureTrainingSupervisionExport? textureSupervision = null;
                if (exportTextureSupervision)
                {
                    string outputDirectory = Path.GetDirectoryName(patch.OutputAdtPath) ?? ".";
                    textureSupervision = AdtTextureTrainingSupervisionExporter.Export(patch.InputAdtPath, patch.TileName, outputDirectory);
                }

                string? outputGlbPath = null;
                if (exportGlb)
                {
                    outputGlbPath = Path.ChangeExtension(patch.OutputAdtPath, ".glb");
                    TerrainHeightmapGlbExporter.Export(
                        outputGlbPath,
                        patch.Heightmap,
                        guideOutputs?.GuideTexturePath ?? patch.TexturePath,
                        tileWorldSize,
                        centerMesh,
                        heightOffset);
                }

                patchedTiles.Add(patch.TileName);
                results.Add(new TerrainPatchResult(
                    patch.SummaryPath,
                    patch.TileName,
                    patch.OutputAdtPath,
                    true,
                    outputGlbPath,
                    guideOutputs?.RawMccvPngPath,
                    guideOutputs?.GuideTexturePath,
                    textureSupervision?.Status,
                    textureSupervision?.MetadataPath,
                    textureSupervision?.TilesetIndexPath,
                    textureSupervision?.TextureMaskPaths ?? [],
                    null,
                    false,
                    chunkChangeAudit,
                    seamAudit));
                patchedCount++;
                Console.WriteLine($"Patched {patch.TileName}: {patch.OutputAdtPath}");
            }
            catch (Exception ex)
            {
                results.Add(new TerrainPatchResult(patch.SummaryPath, patch.TileName, patch.InputAdtPath, false, null, null, null, null, null, [], ex.Message, false, null, null));
                skippedCount++;
            }
        }

        for (int index = 0; index < results.Count; index++)
        {
            TerrainPatchResult result = results[index];
            if (!result.CopiedFromInput || string.IsNullOrWhiteSpace(result.TileName) || !patchedTiles.Contains(result.TileName))
                continue;

            results.RemoveAt(index);
            index--;
        }

        string reportPath = Path.Combine(outputRoot, "terrain_patch_report.json");
        File.WriteAllText(reportPath, JsonSerializer.Serialize(results, CreateJsonOptions()));
        Console.WriteLine($"terrain-patch-adt complete: patched={patchedCount} copied={copiedCount - patchedCount} skipped={skippedCount} report={reportPath}");
    }

    private static Dictionary<string, string> IndexRootAdts(string inputRoot)
    {
        Dictionary<string, string> indexed = new(StringComparer.OrdinalIgnoreCase);
        foreach (string path in Directory.EnumerateFiles(inputRoot, "*.adt", SearchOption.AllDirectories))
        {
            string fileName = Path.GetFileName(path);
            if (fileName.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
                || fileName.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
                || fileName.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string tileName = Path.GetFileNameWithoutExtension(path);
            if (!indexed.TryAdd(tileName, path))
                throw new InvalidOperationException($"Duplicate root ADT tile name '{tileName}' found under {inputRoot}. Use a narrower input directory.");
        }

        return indexed;
    }

    private static void CopyTileFamily(string inputAdtPath, string outputAdtPath)
    {
        string inputDirectory = Path.GetDirectoryName(inputAdtPath) ?? string.Empty;
        string outputDirectory = Path.GetDirectoryName(outputAdtPath) ?? string.Empty;
        string tileName = Path.GetFileNameWithoutExtension(inputAdtPath);

        foreach (string suffix in new[] { "_obj0.adt", "_tex0.adt", "_lod.adt" })
        {
            string source = Path.Combine(inputDirectory, tileName + suffix);
            if (!File.Exists(source))
                continue;

            Directory.CreateDirectory(outputDirectory);
            File.Copy(source, Path.Combine(outputDirectory, Path.GetFileName(source)), overwrite: true);
        }
    }

    private static void CopyRootAdt(string inputAdtPath, string outputAdtPath)
    {
        string? outputDirectory = Path.GetDirectoryName(outputAdtPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        File.Copy(inputAdtPath, outputAdtPath, overwrite: true);
    }

    private static string? ResolveTexturePath(TerrainInferenceSummary summary, string summaryPath)
    {
        if (!string.IsNullOrWhiteSpace(summary.PredictedMeshTexturePath))
        {
            string resolved = ResolveReferencedPath(summaryPath, summary.PredictedMeshTexturePath);
            if (File.Exists(resolved))
                return resolved;
        }

        if (!string.IsNullOrWhiteSpace(summary.SourceMinimapPath))
        {
            string resolved = ResolveReferencedPath(summaryPath, summary.SourceMinimapPath);
            if (File.Exists(resolved))
                return resolved;
        }

        return null;
    }

    private static string? ResolveSourceMinimapPath(TerrainInferenceSummary summary, string summaryPath)
    {
        if (string.IsNullOrWhiteSpace(summary.SourceMinimapPath))
            return null;

        string resolved = ResolveReferencedPath(summaryPath, summary.SourceMinimapPath);
        return File.Exists(resolved) ? resolved : null;
    }

    private static string? ResolveTileName(TerrainInferenceSummary summary, string summaryPath)
    {
        foreach (string? candidate in new[]
                 {
                     summary.TileName,
                     summary.SourceMinimapPath,
                     summary.Shard,
                     summary.PredictedHeight257Path,
                     Path.GetFileName(Path.GetDirectoryName(summaryPath))
                 })
        {
            string? tileName = TryParseTileName(candidate);
            if (!string.IsNullOrWhiteSpace(tileName))
                return tileName;
        }

        return null;
    }

    private static string? TryParseTileName(string? candidate)
    {
        if (string.IsNullOrWhiteSpace(candidate))
            return null;

        string stem = Path.GetFileNameWithoutExtension(candidate.Trim());
        Match match = TileNameRegex.Match(stem);
        if (!match.Success)
            return null;

        return $"{match.Groups["map"].Value}_{match.Groups["x"].Value}_{match.Groups["y"].Value}";
    }

    private static bool TryParseTileIdentity(string tileName, out string mapName, out int tileX, out int tileY)
    {
        mapName = string.Empty;
        tileX = 0;
        tileY = 0;

        Match match = TileNameRegex.Match(tileName);
        if (!match.Success)
            return false;

        mapName = match.Groups["map"].Value;
        return !string.IsNullOrWhiteSpace(mapName)
            && int.TryParse(match.Groups["x"].Value, out tileX)
            && int.TryParse(match.Groups["y"].Value, out tileY);
    }

    private static Dictionary<(int TileX, int TileY), float[]> LoadNeighborAnchors(
        IReadOnlyDictionary<string, string> rootAdts,
        IReadOnlyList<PendingTerrainPatch> pendingPatches)
    {
        Dictionary<(int TileX, int TileY), float[]> anchors = [];
        HashSet<(int TileX, int TileY)> predictedTiles = pendingPatches
            .Select(static patch => (patch.TileX, patch.TileY))
            .ToHashSet();

        foreach (PendingTerrainPatch patch in pendingPatches)
        {
            for (int offsetY = -1; offsetY <= 1; offsetY++)
            {
                for (int offsetX = -1; offsetX <= 1; offsetX++)
                {
                    if (offsetX == 0 && offsetY == 0)
                        continue;

                    (int TileX, int TileY) neighborCoord = (patch.TileX + offsetX, patch.TileY + offsetY);
                    if (predictedTiles.Contains(neighborCoord) || anchors.ContainsKey(neighborCoord))
                        continue;

                    string neighborTileName = $"{patch.MapName}_{neighborCoord.TileX}_{neighborCoord.TileY}";
                    if (!rootAdts.TryGetValue(neighborTileName, out string? neighborAdtPath))
                        continue;

                    WorldTerrainHeightmapData? neighborHeightmap = WorldTerrainTileBuilder.Read(neighborAdtPath).Heightmap;
                    if (neighborHeightmap?.Heights is null || neighborHeightmap.Heights.Length != TileHeightmapSize * TileHeightmapSize)
                        continue;

                    anchors.Add(neighborCoord, neighborHeightmap.Heights.ToArray());
                }
            }
        }

        return anchors;
    }

    private static string ResolveReferencedPath(string summaryPath, string referencedPath)
    {
        if (Path.IsPathRooted(referencedPath))
            return Path.GetFullPath(referencedPath);

        string summaryDirectory = Path.GetDirectoryName(summaryPath) ?? Environment.CurrentDirectory;
        return Path.GetFullPath(Path.Combine(summaryDirectory, referencedPath));
    }

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string arg = args[index];
            if (!names.Contains(arg, StringComparer.OrdinalIgnoreCase))
                continue;

            if (index + 1 >= args.Length)
                return null;

            return args[index + 1];
        }

        return null;
    }

    private static bool HasFlag(string[] args, string name)
    {
        return args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
    }

    private static float? GetSingleOption(string[] args, string name)
    {
        string? value = GetOption(args, name);
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return float.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
    }

    private static JsonSerializerOptions CreateJsonOptions()
    {
        return new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            WriteIndented = true,
        };
    }

    private sealed record TerrainPatchResult(
        string? SummaryPath,
        string? TileName,
        string? OutputAdtPath,
        bool Patched,
        string? OutputGlbPath,
        string? OutputMccvPath,
        string? OutputGuideTexturePath,
        string? TextureSupervisionStatus,
        string? OutputTextureMetadataPath,
        string? OutputTilesetIndexPath,
        IReadOnlyList<string> OutputTextureMaskPaths,
        string? Error,
        bool CopiedFromInput,
        AdtChunkChangeAudit? ChunkChangeAudit,
        AdtSeamAudit? SeamAudit);

    private sealed record PendingTerrainPatch(
        string SummaryPath,
        string TileName,
        string MapName,
        int TileX,
        int TileY,
        string InputAdtPath,
        string OutputAdtPath,
        string? TexturePath,
        string? SourceMinimapPath,
        float[] Heightmap,
        float[] OriginalHeightmap,
        Dictionary<(int TileX, int TileY), float[]> ReferenceHeightmaps);

    private sealed class TerrainInferenceSummary
    {
        [JsonPropertyName("tile_name")]
        public string? TileName { get; set; }

        [JsonPropertyName("source_minimap_path")]
        public string? SourceMinimapPath { get; set; }

        [JsonPropertyName("shard")]
        public string? Shard { get; set; }

        [JsonPropertyName("predicted_height_257_path")]
        public string? PredictedHeight257Path { get; set; }

        [JsonPropertyName("predicted_mesh_texture_path")]
        public string? PredictedMeshTexturePath { get; set; }
    }
}