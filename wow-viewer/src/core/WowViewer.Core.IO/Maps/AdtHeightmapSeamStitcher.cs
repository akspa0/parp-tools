namespace WowViewer.Core.IO.Maps;

public static class AdtHeightmapSeamStitcher
{
    public const int TileHeightmapSize = 257;

    public static void StitchSharedEdges(IReadOnlyDictionary<(int TileX, int TileY), float[]> tileHeightmaps)
    {
        ArgumentNullException.ThrowIfNull(tileHeightmaps);

        ValidateHeightmaps(tileHeightmaps, nameof(tileHeightmaps));

        foreach (((int tileX, int tileY), float[] heightmap) in tileHeightmaps)
        {
            if (tileHeightmaps.TryGetValue((tileX + 1, tileY), out float[]? rightNeighbor))
                StitchVerticalEdge(heightmap, rightNeighbor);

            if (tileHeightmaps.TryGetValue((tileX, tileY + 1), out float[]? bottomNeighbor))
                StitchHorizontalEdge(heightmap, bottomNeighbor);
        }

        HashSet<(int CornerX, int CornerY)> processedCorners = [];
        foreach ((int tileX, int tileY) in tileHeightmaps.Keys)
        {
            StitchCorner(tileHeightmaps, processedCorners, tileX, tileY);
            StitchCorner(tileHeightmaps, processedCorners, tileX + 1, tileY);
            StitchCorner(tileHeightmaps, processedCorners, tileX, tileY + 1);
            StitchCorner(tileHeightmaps, processedCorners, tileX + 1, tileY + 1);
        }
    }

    public static void AnchorPredictedEdgesToNeighbors(
        IReadOnlyDictionary<(int TileX, int TileY), float[]> predictedHeightmaps,
        IReadOnlyDictionary<(int TileX, int TileY), float[]> neighborHeightmaps)
    {
        ArgumentNullException.ThrowIfNull(predictedHeightmaps);
        ArgumentNullException.ThrowIfNull(neighborHeightmaps);

        ValidateHeightmaps(predictedHeightmaps, nameof(predictedHeightmaps));
        ValidateHeightmaps(neighborHeightmaps, nameof(neighborHeightmaps));

        foreach (((int tileX, int tileY), float[] heightmap) in predictedHeightmaps)
        {
            if (!predictedHeightmaps.ContainsKey((tileX - 1, tileY))
                && neighborHeightmaps.TryGetValue((tileX - 1, tileY), out float[]? leftNeighbor))
            {
                CopyLeftEdgeFromNeighbor(heightmap, leftNeighbor);
            }

            if (!predictedHeightmaps.ContainsKey((tileX + 1, tileY))
                && neighborHeightmaps.TryGetValue((tileX + 1, tileY), out float[]? rightNeighbor))
            {
                CopyRightEdgeFromNeighbor(heightmap, rightNeighbor);
            }

            if (!predictedHeightmaps.ContainsKey((tileX, tileY - 1))
                && neighborHeightmaps.TryGetValue((tileX, tileY - 1), out float[]? topNeighbor))
            {
                CopyTopEdgeFromNeighbor(heightmap, topNeighbor);
            }

            if (!predictedHeightmaps.ContainsKey((tileX, tileY + 1))
                && neighborHeightmaps.TryGetValue((tileX, tileY + 1), out float[]? bottomNeighbor))
            {
                CopyBottomEdgeFromNeighbor(heightmap, bottomNeighbor);
            }
        }

        HashSet<(int CornerX, int CornerY)> processedCorners = [];
        foreach ((int tileX, int tileY) in predictedHeightmaps.Keys)
        {
            AnchorCorner(predictedHeightmaps, neighborHeightmaps, processedCorners, tileX, tileY);
            AnchorCorner(predictedHeightmaps, neighborHeightmaps, processedCorners, tileX + 1, tileY);
            AnchorCorner(predictedHeightmaps, neighborHeightmaps, processedCorners, tileX, tileY + 1);
            AnchorCorner(predictedHeightmaps, neighborHeightmaps, processedCorners, tileX + 1, tileY + 1);
        }
    }

    private static void StitchVerticalEdge(float[] leftHeightmap, float[] rightHeightmap)
    {
        for (int sampleY = 0; sampleY < TileHeightmapSize; sampleY++)
        {
            int leftIndex = (sampleY * TileHeightmapSize) + (TileHeightmapSize - 1);
            int rightIndex = sampleY * TileHeightmapSize;
            float averaged = (leftHeightmap[leftIndex] + rightHeightmap[rightIndex]) * 0.5f;
            leftHeightmap[leftIndex] = averaged;
            rightHeightmap[rightIndex] = averaged;
        }
    }

    private static void StitchHorizontalEdge(float[] topHeightmap, float[] bottomHeightmap)
    {
        int topRowOffset = (TileHeightmapSize - 1) * TileHeightmapSize;
        for (int sampleX = 0; sampleX < TileHeightmapSize; sampleX++)
        {
            int topIndex = topRowOffset + sampleX;
            int bottomIndex = sampleX;
            float averaged = (topHeightmap[topIndex] + bottomHeightmap[bottomIndex]) * 0.5f;
            topHeightmap[topIndex] = averaged;
            bottomHeightmap[bottomIndex] = averaged;
        }
    }

    private static void CopyLeftEdgeFromNeighbor(float[] heightmap, float[] leftNeighbor)
    {
        for (int sampleY = 0; sampleY < TileHeightmapSize; sampleY++)
        {
            int targetIndex = sampleY * TileHeightmapSize;
            int sourceIndex = (sampleY * TileHeightmapSize) + (TileHeightmapSize - 1);
            heightmap[targetIndex] = leftNeighbor[sourceIndex];
        }
    }

    private static void CopyRightEdgeFromNeighbor(float[] heightmap, float[] rightNeighbor)
    {
        for (int sampleY = 0; sampleY < TileHeightmapSize; sampleY++)
        {
            int targetIndex = (sampleY * TileHeightmapSize) + (TileHeightmapSize - 1);
            int sourceIndex = sampleY * TileHeightmapSize;
            heightmap[targetIndex] = rightNeighbor[sourceIndex];
        }
    }

    private static void CopyTopEdgeFromNeighbor(float[] heightmap, float[] topNeighbor)
    {
        int sourceRowOffset = (TileHeightmapSize - 1) * TileHeightmapSize;
        for (int sampleX = 0; sampleX < TileHeightmapSize; sampleX++)
            heightmap[sampleX] = topNeighbor[sourceRowOffset + sampleX];
    }

    private static void CopyBottomEdgeFromNeighbor(float[] heightmap, float[] bottomNeighbor)
    {
        int targetRowOffset = (TileHeightmapSize - 1) * TileHeightmapSize;
        for (int sampleX = 0; sampleX < TileHeightmapSize; sampleX++)
            heightmap[targetRowOffset + sampleX] = bottomNeighbor[sampleX];
    }

    private static void StitchCorner(
        IReadOnlyDictionary<(int TileX, int TileY), float[]> tileHeightmaps,
        ISet<(int CornerX, int CornerY)> processedCorners,
        int cornerX,
        int cornerY)
    {
        if (!processedCorners.Add((cornerX, cornerY)))
            return;

        List<(float[] Heightmap, int Index)> samples = [];
        AddCornerSample(tileHeightmaps, samples, cornerX - 1, cornerY - 1, TileHeightmapSize - 1, TileHeightmapSize - 1);
        AddCornerSample(tileHeightmaps, samples, cornerX, cornerY - 1, 0, TileHeightmapSize - 1);
        AddCornerSample(tileHeightmaps, samples, cornerX - 1, cornerY, TileHeightmapSize - 1, 0);
        AddCornerSample(tileHeightmaps, samples, cornerX, cornerY, 0, 0);

        if (samples.Count <= 1)
            return;

        float sum = 0f;
        foreach ((float[] heightmap, int index) in samples)
            sum += heightmap[index];

        float averaged = sum / samples.Count;
        foreach ((float[] heightmap, int index) in samples)
            heightmap[index] = averaged;
    }

    private static void AddCornerSample(
        IReadOnlyDictionary<(int TileX, int TileY), float[]> tileHeightmaps,
        ICollection<(float[] Heightmap, int Index)> samples,
        int tileX,
        int tileY,
        int sampleX,
        int sampleY)
    {
        if (!tileHeightmaps.TryGetValue((tileX, tileY), out float[]? heightmap))
            return;

        samples.Add((heightmap, (sampleY * TileHeightmapSize) + sampleX));
    }

    private static void AnchorCorner(
        IReadOnlyDictionary<(int TileX, int TileY), float[]> predictedHeightmaps,
        IReadOnlyDictionary<(int TileX, int TileY), float[]> neighborHeightmaps,
        ISet<(int CornerX, int CornerY)> processedCorners,
        int cornerX,
        int cornerY)
    {
        if (!processedCorners.Add((cornerX, cornerY)))
            return;

        List<(float[] Heightmap, int Index)> predictedSamples = [];
        AddCornerSample(predictedHeightmaps, predictedSamples, cornerX - 1, cornerY - 1, TileHeightmapSize - 1, TileHeightmapSize - 1);
        AddCornerSample(predictedHeightmaps, predictedSamples, cornerX, cornerY - 1, 0, TileHeightmapSize - 1);
        AddCornerSample(predictedHeightmaps, predictedSamples, cornerX - 1, cornerY, TileHeightmapSize - 1, 0);
        AddCornerSample(predictedHeightmaps, predictedSamples, cornerX, cornerY, 0, 0);
        if (predictedSamples.Count == 0)
            return;

        List<(float[] Heightmap, int Index)> anchorSamples = [];
        AddCornerAnchorSample(predictedHeightmaps, neighborHeightmaps, anchorSamples, cornerX - 1, cornerY - 1, TileHeightmapSize - 1, TileHeightmapSize - 1);
        AddCornerAnchorSample(predictedHeightmaps, neighborHeightmaps, anchorSamples, cornerX, cornerY - 1, 0, TileHeightmapSize - 1);
        AddCornerAnchorSample(predictedHeightmaps, neighborHeightmaps, anchorSamples, cornerX - 1, cornerY, TileHeightmapSize - 1, 0);
        AddCornerAnchorSample(predictedHeightmaps, neighborHeightmaps, anchorSamples, cornerX, cornerY, 0, 0);
        if (anchorSamples.Count == 0)
            return;

        float sum = 0f;
        foreach ((float[] heightmap, int index) in anchorSamples)
            sum += heightmap[index];

        float anchored = sum / anchorSamples.Count;
        foreach ((float[] heightmap, int index) in predictedSamples)
            heightmap[index] = anchored;
    }

    private static void AddCornerAnchorSample(
        IReadOnlyDictionary<(int TileX, int TileY), float[]> predictedHeightmaps,
        IReadOnlyDictionary<(int TileX, int TileY), float[]> neighborHeightmaps,
        ICollection<(float[] Heightmap, int Index)> samples,
        int tileX,
        int tileY,
        int sampleX,
        int sampleY)
    {
        if (predictedHeightmaps.ContainsKey((tileX, tileY)))
            return;

        AddCornerSample(neighborHeightmaps, samples, tileX, tileY, sampleX, sampleY);
    }

    private static void ValidateHeightmaps(IReadOnlyDictionary<(int TileX, int TileY), float[]> tileHeightmaps, string paramName)
    {
        foreach (((int tileX, int tileY), float[] heightmap) in tileHeightmaps)
        {
            ArgumentNullException.ThrowIfNull(heightmap, $"Tile ({tileX},{tileY}) heightmap");
            if (heightmap.Length != TileHeightmapSize * TileHeightmapSize)
                throw new ArgumentException($"Tile ({tileX},{tileY}) heightmap must contain exactly {TileHeightmapSize * TileHeightmapSize} samples.", paramName);
        }
    }
}