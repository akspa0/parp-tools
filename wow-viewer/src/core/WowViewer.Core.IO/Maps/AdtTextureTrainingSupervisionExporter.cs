using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtTextureTrainingSupervisionExporter
{
    public const int ChunkAlphaSize = 64;
    public const int TileChunks = 16;
    public const int TileAlphaSize = ChunkAlphaSize * TileChunks;
    public const byte NoDataTilesetIndex = byte.MaxValue;

    public static AdtTextureTrainingSupervisionExport Export(string sourceAdtPath, string tileName, string outputDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceAdtPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);

        AdtTileFamily family = AdtTileFamilyResolver.Resolve(sourceAdtPath);
        string? textureSourcePath = family.TextureSourcePath;
        if (string.IsNullOrWhiteSpace(textureSourcePath) || family.TextureSourceKind is null)
            return new AdtTextureTrainingSupervisionExport("no-texture-source", null, null, [], 0, 0, 0, 0);

        AdtTextureFile textureFile = AdtTextureReader.Read(textureSourcePath);
        return Export(textureFile, tileName, outputDirectory);
    }

    public static AdtTextureTrainingSupervisionExport Export(AdtTextureFile textureFile, string tileName, string outputDirectory)
    {
        ArgumentNullException.ThrowIfNull(textureFile);
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);

        if (textureFile.TextureNames.Count == 0)
            return new AdtTextureTrainingSupervisionExport("no-texture-names", null, null, [], 0, 0, 0, 0);

        int chunksWithLayers = textureFile.Chunks.Count(static chunk => chunk.Layers.Count > 0);
        if (chunksWithLayers == 0)
            return new AdtTextureTrainingSupervisionExport("no-layer-data", null, null, [], 0, 0, 0, 0);

        List<TextureEntryState> textures = CreateTextureEntries(textureFile);
        if (textures.Count == 0)
            return new AdtTextureTrainingSupervisionExport("no-used-textures", null, null, [], 0, chunksWithLayers, 0, 0);

        byte[] tilesetIndex = Enumerable.Repeat(NoDataTilesetIndex, TileAlphaSize * TileAlphaSize).ToArray();
        int missingDecodedAlphaLayers = 0;

        foreach (AdtTextureChunk chunk in textureFile.Chunks)
        {
            if (chunk.Layers.Count == 0)
                continue;

            int[] perTextureContributions = new int[textures.Count];
            byte[][] alphaMaps = chunk.Layers
                .Select(static layer => layer.DecodedAlpha?.AlphaMap ?? Array.Empty<byte>())
                .ToArray();

            for (int localY = 0; localY < ChunkAlphaSize; localY++)
            {
                for (int localX = 0; localX < ChunkAlphaSize; localX++)
                {
                    Array.Clear(perTextureContributions, 0, perTextureContributions.Length);
                    int localIndex = (localY * ChunkAlphaSize) + localX;
                    int remaining = 255;

                    for (int layerIndex = chunk.Layers.Count - 1; layerIndex >= 1; layerIndex--)
                    {
                        AdtTextureChunkLayer layer = chunk.Layers[layerIndex];
                        byte[] alphaMap = alphaMaps[layerIndex];
                        if (alphaMap.Length != ChunkAlphaSize * ChunkAlphaSize)
                        {
                            missingDecodedAlphaLayers++;
                            continue;
                        }

                        int alpha = alphaMap[localIndex];
                        int contribution = ((alpha * remaining) + 127) / 255;
                        if (contribution > 0 && TryResolveTextureOrdinal(textures, layer.TextureId, out int ordinal))
                            perTextureContributions[ordinal] = Math.Min(255, perTextureContributions[ordinal] + contribution);

                        remaining = ((remaining * (255 - alpha)) + 127) / 255;
                    }

                    AdtTextureChunkLayer baseLayer = chunk.Layers[0];
                    if (remaining > 0 && TryResolveTextureOrdinal(textures, baseLayer.TextureId, out int baseOrdinal))
                        perTextureContributions[baseOrdinal] = Math.Min(255, perTextureContributions[baseOrdinal] + remaining);

                    int globalX = (chunk.ChunkX * ChunkAlphaSize) + localX;
                    int globalY = (chunk.ChunkY * ChunkAlphaSize) + localY;
                    int globalIndex = (globalY * TileAlphaSize) + globalX;

                    int dominantOrdinal = -1;
                    int dominantContribution = -1;
                    for (int ordinal = 0; ordinal < textures.Count; ordinal++)
                    {
                        int contribution = perTextureContributions[ordinal];
                        if (contribution <= 0)
                            continue;

                        textures[ordinal].Mask[globalIndex] = (byte)contribution;
                        textures[ordinal].CoveredPixelCount++;
                        if (contribution > dominantContribution)
                        {
                            dominantContribution = contribution;
                            dominantOrdinal = ordinal;
                        }
                    }

                    if (dominantOrdinal >= 0)
                        tilesetIndex[globalIndex] = (byte)dominantOrdinal;
                }
            }

            HashSet<int> coveredOrdinals = [];
            foreach (AdtTextureChunkLayer layer in chunk.Layers)
            {
                if (TryResolveTextureOrdinal(textures, layer.TextureId, out int ordinal))
                    coveredOrdinals.Add(ordinal);
            }

            foreach (int ordinal in coveredOrdinals)
                textures[ordinal].CoveredChunkCount++;
        }

        Directory.CreateDirectory(outputDirectory);
        string tilesetIndexPath = Path.Combine(outputDirectory, $"{tileName}_tileset_index.png");
        SaveGrayPng(tilesetIndexPath, tilesetIndex);

        List<TextureMaskManifestEntry> manifestEntries = [];
        List<string> maskPaths = [];
        for (int ordinal = 0; ordinal < textures.Count; ordinal++)
        {
            TextureEntryState texture = textures[ordinal];
            string maskPath = Path.Combine(outputDirectory, $"{tileName}_tileset_{ordinal:D2}_mask.png");
            SaveGrayPng(maskPath, texture.Mask);
            maskPaths.Add(maskPath);
            manifestEntries.Add(new TextureMaskManifestEntry(
                ordinal,
                texture.TextureId,
                texture.TexturePath,
                maskPath,
                texture.CoveredChunkCount,
                texture.CoveredPixelCount));
        }

        string metadataPath = Path.Combine(outputDirectory, $"{tileName}_texture_supervision.json");
        TextureSupervisionManifest manifest = new(
            tileName,
            textureFile.SourcePath,
            textureFile.Kind.ToString(),
            TileAlphaSize,
            tilesetIndexPath,
            chunksWithLayers,
            textureFile.Chunks.Count,
            missingDecodedAlphaLayers,
            manifestEntries);
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));

        return new AdtTextureTrainingSupervisionExport(
            missingDecodedAlphaLayers > 0 ? "exported-partial" : "exported",
            metadataPath,
            tilesetIndexPath,
            maskPaths,
            textures.Count,
            chunksWithLayers,
            textureFile.Chunks.Count,
            missingDecodedAlphaLayers);
    }

    private static List<TextureEntryState> CreateTextureEntries(AdtTextureFile textureFile)
    {
        List<TextureEntryState> textures = [];
        HashSet<uint> seenTextureIds = [];
        foreach (AdtTextureChunk chunk in textureFile.Chunks)
        {
            foreach (AdtTextureChunkLayer layer in chunk.Layers)
            {
                if (!seenTextureIds.Add(layer.TextureId))
                    continue;

                textures.Add(new TextureEntryState(layer.TextureId, layer.TexturePath));
            }
        }

        return textures;
    }

    private static bool TryResolveTextureOrdinal(IReadOnlyList<TextureEntryState> textures, uint textureId, out int ordinal)
    {
        for (int index = 0; index < textures.Count; index++)
        {
            if (textures[index].TextureId != textureId)
                continue;

            ordinal = index;
            return true;
        }

        ordinal = -1;
        return false;
    }

    private static void SaveGrayPng(string path, byte[] pixels)
    {
        using Image<L8> image = Image.LoadPixelData<L8>(pixels, TileAlphaSize, TileAlphaSize);
        image.SaveAsPng(path);
    }

    public sealed record AdtTextureTrainingSupervisionExport(
        string Status,
        string? MetadataPath,
        string? TilesetIndexPath,
        IReadOnlyList<string> TextureMaskPaths,
        int TextureCount,
        int ChunksWithLayers,
        int TotalChunkCount,
        int MissingDecodedAlphaLayerCount);

    private sealed class TextureEntryState
    {
        public TextureEntryState(uint textureId, string? texturePath)
        {
            TextureId = textureId;
            TexturePath = texturePath;
            Mask = new byte[TileAlphaSize * TileAlphaSize];
        }

        public uint TextureId { get; }

        public string? TexturePath { get; }

        public byte[] Mask { get; }

        public int CoveredChunkCount { get; set; }

        public int CoveredPixelCount { get; set; }
    }

    private sealed record TextureSupervisionManifest(
        string TileName,
        string TextureSourcePath,
        string TextureSourceKind,
        int TileAlphaSize,
        string TilesetIndexPath,
        int ChunksWithLayers,
        int TotalChunkCount,
        int MissingDecodedAlphaLayerCount,
        IReadOnlyList<TextureMaskManifestEntry> Textures);

    private sealed record TextureMaskManifestEntry(
        int Ordinal,
        uint TextureId,
        string? TexturePath,
        string MaskPath,
        int CoveredChunkCount,
        int CoveredPixelCount);
}