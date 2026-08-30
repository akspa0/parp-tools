namespace WowViewer.Core.IO.Terrain;

/// <summary>
/// Result of multi-layer chunk texture allocation, containing mapped texture paths and normalized 64x64 alpha splats.
/// </summary>
public sealed class AllocatedChunkLayers
{
    public string[] TexturePaths { get; init; } = [];
    public List<byte[]> AlphaSplats { get; init; } = []; // Layer 0 has no alpha splat (base); Layers 1..N-1 have 64x64 splats
    public uint[] EffectIds { get; init; } = [];
    public uint[] Flags { get; init; } = [];
}

/// <summary>
/// Manages multi-layer texture allocation, merging, and budget enforcement for ADT terrain chunks.
/// Strictly enforces the engine's 4-layer-per-chunk limit using energy-based pruning and normalization.
/// </summary>
public static class TerrainLayerAllocator
{
    public const int MaxLayersPerChunk = 4;
    public const int AlphaResolution = 64;
    public const int AlphaPixelCount = AlphaResolution * AlphaResolution;

    /// <summary>
    /// Merges stamp layers onto an existing chunk's layer stack, enforcing the 4-layer ceiling.
    /// </summary>
    public static AllocatedChunkLayers MergeLayers(
        IReadOnlyList<string> existingTextures,
        IReadOnlyList<byte[]>? existingAlphas,
        IReadOnlyList<TerrainPasteLayer> incomingLayers,
        float stampWeight = 1.0f)
    {
        stampWeight = Math.Clamp(stampWeight, 0f, 1f);

        // Gather all candidate layers with their combined 64x64 alpha buffers
        var layerMap = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);

        // 1. Ingest existing layers
        for (int i = 0; i < existingTextures.Count; i++)
        {
            string tex = existingTextures[i];
            if (string.IsNullOrWhiteSpace(tex))
                continue;

            if (!layerMap.TryGetValue(tex, out float[]? alphaBuf))
            {
                alphaBuf = new float[AlphaPixelCount];
                layerMap[tex] = alphaBuf;
            }

            if (i == 0)
            {
                // Base layer has full coverage initially
                for (int p = 0; p < AlphaPixelCount; p++)
                    alphaBuf[p] = 255f;
            }
            else if (existingAlphas != null && i - 1 < existingAlphas.Count)
            {
                byte[] srcAlpha = existingAlphas[i - 1];
                for (int p = 0; p < Math.Min(AlphaPixelCount, srcAlpha.Length); p++)
                    alphaBuf[p] = Math.Max(alphaBuf[p], srcAlpha[p]);
            }
        }

        // Ensure at least one base texture exists
        if (layerMap.Count == 0 && incomingLayers.Count > 0)
        {
            string firstTex = incomingLayers[0].TexturePath;
            if (!string.IsNullOrWhiteSpace(firstTex))
            {
                var buf = new float[AlphaPixelCount];
                Array.Fill(buf, 255f);
                layerMap[firstTex] = buf;
            }
        }

        // 2. Ingest incoming stamp layers
        foreach (TerrainPasteLayer inc in incomingLayers)
        {
            if (string.IsNullOrWhiteSpace(inc.TexturePath))
                continue;

            if (!layerMap.TryGetValue(inc.TexturePath, out float[]? alphaBuf))
            {
                alphaBuf = new float[AlphaPixelCount];
                layerMap[inc.TexturePath] = alphaBuf;
            }

            if (inc.AlphaMask.Length == AlphaPixelCount)
            {
                for (int p = 0; p < AlphaPixelCount; p++)
                {
                    float incVal = inc.AlphaMask[p] * stampWeight;
                    alphaBuf[p] = Math.Clamp(alphaBuf[p] + incVal, 0f, 255f);
                }
            }
            else if (inc.AlphaMask.Length > 0 && inc.Resolution > 0)
            {
                // Resample to 64x64
                for (int y = 0; y < AlphaResolution; y++)
                {
                    float v = (float)y / (AlphaResolution - 1);
                    int sy = Math.Clamp((int)Math.Round(v * (inc.Resolution - 1)), 0, inc.Resolution - 1);

                    for (int x = 0; x < AlphaResolution; x++)
                    {
                        float u = (float)x / (AlphaResolution - 1);
                        int sx = Math.Clamp((int)Math.Round(u * (inc.Resolution - 1)), 0, inc.Resolution - 1);

                        byte val = inc.AlphaMask[sy * inc.Resolution + sx];
                        alphaBuf[y * AlphaResolution + x] = Math.Clamp(alphaBuf[y * AlphaResolution + x] + val * stampWeight, 0f, 255f);
                    }
                }
            }
        }

        // 3. If layers exceed MaxLayersPerChunk (4), prune the lowest energy layers
        List<KeyValuePair<string, float[]>> sortedLayers;
        if (layerMap.Count > MaxLayersPerChunk)
        {
            // Base layer is preserved, sort other layers by sum of alpha energy
            string baseTex = existingTextures.Count > 0 ? existingTextures[0] : layerMap.Keys.First();

            var otherLayers = layerMap
                .Where(kvp => !string.Equals(kvp.Key, baseTex, StringComparison.OrdinalIgnoreCase))
                .Select(kvp => new { kvp.Key, kvp.Value, TotalEnergy = kvp.Value.Sum() })
                .OrderByDescending(x => x.TotalEnergy)
                .Take(MaxLayersPerChunk - 1)
                .Select(x => new KeyValuePair<string, float[]>(x.Key, x.Value))
                .ToList();

            sortedLayers = [new KeyValuePair<string, float[]>(baseTex, layerMap[baseTex]), .. otherLayers];
        }
        else
        {
            // Keep base texture at index 0 if available
            string? baseTex = existingTextures.Count > 0 && layerMap.ContainsKey(existingTextures[0])
                ? existingTextures[0]
                : layerMap.Keys.FirstOrDefault();

            if (baseTex != null)
            {
                sortedLayers = [new KeyValuePair<string, float[]>(baseTex, layerMap[baseTex])];
                foreach (var kvp in layerMap)
                {
                    if (!string.Equals(kvp.Key, baseTex, StringComparison.OrdinalIgnoreCase))
                        sortedLayers.Add(kvp);
                }
            }
            else
            {
                sortedLayers = layerMap.ToList();
            }
        }

        // 4. Build allocated arrays
        string[] outTextures = new string[sortedLayers.Count];
        var outSplats = new List<byte[]>();

        for (int i = 0; i < sortedLayers.Count; i++)
        {
            outTextures[i] = sortedLayers[i].Key;
            if (i > 0) // Layer 0 has no alpha splat in MCAL
            {
                float[] rawBuf = sortedLayers[i].Value;
                byte[] splat = new byte[AlphaPixelCount];
                for (int p = 0; p < AlphaPixelCount; p++)
                    splat[p] = (byte)Math.Clamp((int)Math.Round(rawBuf[p]), 0, 255);
                outSplats.Add(splat);
            }
        }

        return new AllocatedChunkLayers
        {
            TexturePaths = outTextures,
            AlphaSplats = outSplats,
            EffectIds = new uint[outTextures.Length],
            Flags = new uint[outTextures.Length]
        };
    }
}
