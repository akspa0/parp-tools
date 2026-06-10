using WowViewer.Core.IO.Maps;
namespace WoWViewer.Terrain.Vlm;

public static class VlmShadowAssociationService
{
    private const int ShadowDimension = 64;
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16f;
    private const float PixelSize = ChunkSize / ShadowDimension;
    private const float BaseObjectMarginPixels = 6f;
    private const float FootprintPaddingPixels = 1.5f;
    private const int MaxRegionCandidateCount = 8;
    private const int MaxChunkCandidateCount = 32;
    private const int MinScarRegionPixels = 8;
    private const float ExplainedOverlapThreshold = 0.60f;
    private const float AmbiguousOverlapThreshold = 0.20f;
    private const float ScarCandidateScoreThreshold = 0.55f;

    public static VlmChunkShadowAnalysis[] AnalyzeTile(
        IReadOnlyList<VlmChunkShadowBits> shadowBits,
        float[] chunkPositions,
        IReadOnlyList<VlmObjectPlacement> objects)
    {
        List<VlmChunkShadowAnalysis> analyses = new(shadowBits.Count);
        foreach (VlmChunkShadowBits shadowBit in shadowBits.OrderBy(static item => item.ChunkIndex))
        {
            int positionIndex = shadowBit.ChunkIndex * 3;
            if (positionIndex + 2 >= chunkPositions.Length)
                continue;

            byte[] rawShadow = Convert.FromBase64String(shadowBit.BitsBase64);
            analyses.Add(AnalyzeChunk(
                shadowBit.ChunkIndex,
                rawShadow,
                chunkPositions[positionIndex],
                chunkPositions[positionIndex + 1],
                chunkPositions[positionIndex + 2],
                objects));
        }

        return analyses.ToArray();
    }

    private static VlmChunkShadowAnalysis AnalyzeChunk(
        int chunkIndex,
        byte[] rawShadowBits,
        float chunkWorldX,
        float chunkWorldY,
        float chunkWorldZ,
        IReadOnlyList<VlmObjectPlacement> objects)
    {
        byte[] shadow = ShadowMapService.ReadShadow(rawShadowBits);
        List<ShadowRegionAccumulator> regions = ExtractRegions(shadow);
        int shadowedPixelCount = regions.Sum(static region => region.PixelCount);
        int largestRegionPixelCount = regions.Count == 0 ? 0 : regions.Max(static region => region.PixelCount);
        bool[] currentObjectMask = new bool[shadow.Length];

        List<ShadowCandidateAccumulator> candidates = [];
        foreach (VlmObjectPlacement obj in objects)
        {
            (float pixelX, float pixelY) = ProjectObjectToChunkPixels(obj, chunkWorldX, chunkWorldY);
            bool insideChunk = pixelX >= 0f && pixelX < ShadowDimension && pixelY >= 0f && pixelY < ShadowDimension;
            float radiusPixels = EstimateHorizontalRadiusPixels(obj);
            float chunkDistance = DistanceToRectangle(pixelX, pixelY, 0f, 0f, ShadowDimension - 1, ShadowDimension - 1);
            float candidateThreshold = MathF.Max(BaseObjectMarginPixels, radiusPixels + 2f);
            if (!insideChunk && chunkDistance > candidateThreshold)
                continue;

            int? nearestRegionId = null;
            float nearestDistance = float.PositiveInfinity;
            bool insideRegionBounds = false;
            foreach (ShadowRegionAccumulator region in regions)
            {
                float regionDistance = DistanceToRectangle(pixelX, pixelY, region.MinX, region.MinY, region.MaxX, region.MaxY);
                if (regionDistance < nearestDistance)
                {
                    nearestDistance = regionDistance;
                    nearestRegionId = region.RegionId;
                    insideRegionBounds = regionDistance <= 0f;
                }
            }

            if (nearestRegionId.HasValue && nearestDistance > candidateThreshold && !insideRegionBounds)
                continue;

            if (!nearestRegionId.HasValue && !insideChunk)
                continue;

            float explanationRadiusPixels = MathF.Max(1.5f, radiusPixels + FootprintPaddingPixels);
            RasterizeObjectFootprint(currentObjectMask, pixelX, pixelY, explanationRadiusPixels);

            candidates.Add(new ShadowCandidateAccumulator(
                obj,
                pixelX,
                pixelY,
                nearestRegionId,
                nearestRegionId.HasValue ? nearestDistance : chunkDistance,
                insideChunk,
                insideRegionBounds));
        }

        List<ShadowCandidateAccumulator> topCandidates = candidates
            .OrderBy(static candidate => candidate.PixelDistanceToRegion)
            .ThenByDescending(static candidate => candidate.InsideRegionBounds)
            .ThenByDescending(static candidate => candidate.InsideChunk)
            .ThenBy(static candidate => candidate.Object.UniqueId)
            .Take(MaxChunkCandidateCount)
            .ToList();

        VlmShadowRegion[] regionReports = regions
            .Select(region => region.ToReport(chunkWorldX, chunkWorldY, chunkWorldZ, currentObjectMask, topCandidates))
            .ToArray();

        VlmShadowObjectCandidate[] candidateReports = topCandidates
            .Select(static candidate => candidate.ToReport())
            .ToArray();

        int explainedShadowPixelCount = regionReports.Sum(static region => region.ExplainedShadowPixelCount);
        int residualShadowPixelCount = regionReports.Sum(static region => region.ResidualShadowPixelCount);
        int scarCandidateRegionCount = regionReports.Count(static region => string.Equals(region.ScarType, "unexplained_scar", StringComparison.Ordinal));
        float scarCandidateScore = shadowedPixelCount == 0
            ? 0f
            : regionReports.Sum(region => region.ScarCandidateScore * region.ResidualShadowPixelCount) / shadowedPixelCount;

        return new VlmChunkShadowAnalysis(
            chunkIndex,
            shadowedPixelCount,
            shadowedPixelCount / (float)(ShadowDimension * ShadowDimension),
            regions.Count,
            largestRegionPixelCount,
            explainedShadowPixelCount,
            residualShadowPixelCount,
            shadowedPixelCount == 0 ? 0f : explainedShadowPixelCount / (float)shadowedPixelCount,
            shadowedPixelCount == 0 ? 0f : residualShadowPixelCount / (float)shadowedPixelCount,
            scarCandidateRegionCount,
            scarCandidateScore,
            regionReports,
            candidateReports);
    }

    private static (float PixelX, float PixelY) ProjectObjectToChunkPixels(VlmObjectPlacement obj, float chunkWorldX, float chunkWorldY)
    {
        float pixelX = ((chunkWorldY - obj.Y) / ChunkSize) * ShadowDimension;
        float pixelY = ((chunkWorldX - obj.X) / ChunkSize) * ShadowDimension;
        return (pixelX, pixelY);
    }

    private static float EstimateHorizontalRadiusPixels(VlmObjectPlacement obj)
    {
        if (obj.BoundsMin is null || obj.BoundsMax is null || obj.BoundsMin.Length < 2 || obj.BoundsMax.Length < 2)
            return 0f;

        float minX = obj.BoundsMin[0];
        float minY = obj.BoundsMin[1];
        float maxX = obj.BoundsMax[0];
        float maxY = obj.BoundsMax[1];

        float radiusWorld = MathF.Max(
            MathF.Max(MathF.Sqrt((minX * minX) + (minY * minY)), MathF.Sqrt((minX * minX) + (maxY * maxY))),
            MathF.Max(MathF.Sqrt((maxX * maxX) + (minY * minY)), MathF.Sqrt((maxX * maxX) + (maxY * maxY))));

        return (radiusWorld * MathF.Max(obj.Scale, 0.1f)) / PixelSize;
    }

    private static void RasterizeObjectFootprint(bool[] mask, float pixelX, float pixelY, float radiusPixels)
    {
        if (mask.Length == 0 || radiusPixels <= 0f)
            return;

        int minX = Math.Max(0, (int)MathF.Floor(pixelX - radiusPixels));
        int maxX = Math.Min(ShadowDimension - 1, (int)MathF.Ceiling(pixelX + radiusPixels));
        int minY = Math.Max(0, (int)MathF.Floor(pixelY - radiusPixels));
        int maxY = Math.Min(ShadowDimension - 1, (int)MathF.Ceiling(pixelY + radiusPixels));
        float radiusSquared = radiusPixels * radiusPixels;

        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                float deltaX = x - pixelX;
                float deltaY = y - pixelY;
                if ((deltaX * deltaX) + (deltaY * deltaY) > radiusSquared)
                    continue;

                mask[(y * ShadowDimension) + x] = true;
            }
        }
    }

    private static List<ShadowRegionAccumulator> ExtractRegions(byte[] shadow)
    {
        bool[] visited = new bool[shadow.Length];
        List<ShadowRegionAccumulator> regions = [];
        Queue<int> queue = new();

        for (int y = 0; y < ShadowDimension; y++)
        {
            for (int x = 0; x < ShadowDimension; x++)
            {
                int index = (y * ShadowDimension) + x;
                if (visited[index] || shadow[index] >= 128)
                    continue;

                ShadowRegionAccumulator region = new(regions.Count);
                visited[index] = true;
                queue.Enqueue(index);

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    int currentX = current % ShadowDimension;
                    int currentY = current / ShadowDimension;
                    region.Add(currentX, currentY);

                    EnqueueNeighbor(currentX - 1, currentY);
                    EnqueueNeighbor(currentX + 1, currentY);
                    EnqueueNeighbor(currentX, currentY - 1);
                    EnqueueNeighbor(currentX, currentY + 1);
                }

                regions.Add(region);

                void EnqueueNeighbor(int neighborX, int neighborY)
                {
                    if (neighborX < 0 || neighborX >= ShadowDimension || neighborY < 0 || neighborY >= ShadowDimension)
                        return;

                    int neighborIndex = (neighborY * ShadowDimension) + neighborX;
                    if (visited[neighborIndex] || shadow[neighborIndex] >= 128)
                        return;

                    visited[neighborIndex] = true;
                    queue.Enqueue(neighborIndex);
                }
            }
        }

        return regions;
    }

    private static float DistanceToRectangle(float x, float y, float minX, float minY, float maxX, float maxY)
    {
        float deltaX = x < minX ? minX - x : x > maxX ? x - maxX : 0f;
        float deltaY = y < minY ? minY - y : y > maxY ? y - maxY : 0f;
        return MathF.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
    }

    private sealed class ShadowRegionAccumulator
    {
        private readonly List<int> _pixelIndices = [];

        public ShadowRegionAccumulator(int regionId)
        {
            RegionId = regionId;
        }

        public int RegionId { get; }

        public int PixelCount { get; private set; }

        public int MinX { get; private set; } = int.MaxValue;

        public int MinY { get; private set; } = int.MaxValue;

        public int MaxX { get; private set; } = int.MinValue;

        public int MaxY { get; private set; } = int.MinValue;

        public float SumX { get; private set; }

        public float SumY { get; private set; }

        public void Add(int x, int y)
        {
            PixelCount++;
            MinX = Math.Min(MinX, x);
            MinY = Math.Min(MinY, y);
            MaxX = Math.Max(MaxX, x);
            MaxY = Math.Max(MaxY, y);
            SumX += x;
            SumY += y;
            _pixelIndices.Add((y * ShadowDimension) + x);
        }

        public VlmShadowRegion ToReport(
            float chunkWorldX,
            float chunkWorldY,
            float chunkWorldZ,
            bool[] currentObjectMask,
            IReadOnlyList<ShadowCandidateAccumulator> candidates)
        {
            float centroidX = PixelCount == 0 ? 0f : SumX / PixelCount;
            float centroidY = PixelCount == 0 ? 0f : SumY / PixelCount;
            float centroidWorldX = chunkWorldX - ((centroidY + 0.5f) * PixelSize);
            float centroidWorldY = chunkWorldY - ((centroidX + 0.5f) * PixelSize);
            float worldX0 = chunkWorldX - (MinY * PixelSize);
            float worldX1 = chunkWorldX - ((MaxY + 1) * PixelSize);
            float worldY0 = chunkWorldY - (MinX * PixelSize);
            float worldY1 = chunkWorldY - ((MaxX + 1) * PixelSize);

            uint[] candidateObjectIds = candidates
                .Where(candidate => candidate.NearestRegionId == RegionId)
                .OrderBy(candidate => candidate.PixelDistanceToRegion)
                .ThenBy(candidate => candidate.Object.UniqueId)
                .Take(MaxRegionCandidateCount)
                .Select(candidate => candidate.Object.UniqueId)
                .ToArray();

            int explainedShadowPixelCount = CountExplainedPixels(currentObjectMask);
            int residualShadowPixelCount = Math.Max(0, PixelCount - explainedShadowPixelCount);
            float explainedOverlapRatio = PixelCount == 0 ? 0f : explainedShadowPixelCount / (float)PixelCount;
            float? nearestCandidateDistancePixels = candidates
                .Where(candidate => candidate.NearestRegionId == RegionId)
                .Select(candidate => (float?)candidate.PixelDistanceToRegion)
                .Min();
            float scarCandidateScore = ComputeScarCandidateScore(
                PixelCount,
                residualShadowPixelCount,
                explainedOverlapRatio,
                nearestCandidateDistancePixels,
                candidateObjectIds.Length);
            string scarType = ClassifyRegion(
                PixelCount,
                explainedOverlapRatio,
                scarCandidateScore,
                candidateObjectIds.Length);
            bool explainedByCurrentObjects = string.Equals(scarType, "explained_current", StringComparison.Ordinal);

            return new VlmShadowRegion(
                RegionId,
                PixelCount,
                PixelCount / (float)(ShadowDimension * ShadowDimension),
                [MinX, MinY],
                [MaxX, MaxY],
                [centroidX, centroidY],
                [centroidWorldX, centroidWorldY, chunkWorldZ],
                [MathF.Min(worldX0, worldX1), MathF.Min(worldY0, worldY1)],
                [MathF.Max(worldX0, worldX1), MathF.Max(worldY0, worldY1)],
                candidateObjectIds,
                explainedByCurrentObjects,
                explainedShadowPixelCount,
                residualShadowPixelCount,
                explainedOverlapRatio,
                nearestCandidateDistancePixels,
                scarCandidateScore,
                scarType);
        }

        private int CountExplainedPixels(bool[] currentObjectMask)
        {
            int explainedShadowPixelCount = 0;
            foreach (int pixelIndex in _pixelIndices)
            {
                if (pixelIndex >= 0 && pixelIndex < currentObjectMask.Length && currentObjectMask[pixelIndex])
                    explainedShadowPixelCount++;
            }

            return explainedShadowPixelCount;
        }

        private static float ComputeScarCandidateScore(
            int pixelCount,
            int residualShadowPixelCount,
            float explainedOverlapRatio,
            float? nearestCandidateDistancePixels,
            int candidateObjectCount)
        {
            if (pixelCount <= 0)
                return 0f;

            float residualRatio = residualShadowPixelCount / (float)pixelCount;
            float sizeScore = Math.Clamp((pixelCount - MinScarRegionPixels) / 48f, 0f, 1f);
            float distanceScore = !nearestCandidateDistancePixels.HasValue
                ? 1f
                : Math.Clamp(nearestCandidateDistancePixels.Value / (BaseObjectMarginPixels + 4f), 0f, 1f);
            float crowdingPenalty = candidateObjectCount > 1 ? 0.10f : 0f;
            float overlapPenalty = explainedOverlapRatio * 0.35f;

            return Math.Clamp(
                (residualRatio * 0.70f) + (distanceScore * 0.20f) + (sizeScore * 0.10f) - crowdingPenalty - overlapPenalty,
                0f,
                1f);
        }

        private static string ClassifyRegion(
            int pixelCount,
            float explainedOverlapRatio,
            float scarCandidateScore,
            int candidateObjectCount)
        {
            if (explainedOverlapRatio >= ExplainedOverlapThreshold)
                return "explained_current";

            if (explainedOverlapRatio >= AmbiguousOverlapThreshold || candidateObjectCount > 1)
                return "ambiguous_mixed";

            if (pixelCount >= MinScarRegionPixels && scarCandidateScore >= ScarCandidateScoreThreshold)
                return "unexplained_scar";

            return "non_object_shadow";
        }
    }

    private sealed class ShadowCandidateAccumulator(
        VlmObjectPlacement obj,
        float pixelX,
        float pixelY,
        int? nearestRegionId,
        float pixelDistanceToRegion,
        bool insideChunk,
        bool insideRegionBounds)
    {
        public VlmObjectPlacement Object { get; } = obj;

        public float PixelX { get; } = pixelX;

        public float PixelY { get; } = pixelY;

        public int? NearestRegionId { get; } = nearestRegionId;

        public float PixelDistanceToRegion { get; } = pixelDistanceToRegion;

        public bool InsideChunk { get; } = insideChunk;

        public bool InsideRegionBounds { get; } = insideRegionBounds;

        public VlmShadowObjectCandidate ToReport()
        {
            return new VlmShadowObjectCandidate(
                Object.UniqueId,
                Object.Name,
                Object.Category,
                [Object.X, Object.Y, Object.Z],
                [PixelX, PixelY],
                NearestRegionId,
                PixelDistanceToRegion,
                InsideChunk,
                InsideRegionBounds);
        }
    }
}
