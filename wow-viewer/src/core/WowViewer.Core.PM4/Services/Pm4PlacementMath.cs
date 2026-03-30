using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4PlacementMath
{
    private const float ConnectorQuantizationUnits = 2f;
    private const float ConnectorMergeBoundsPadding = 32f;
    private const float ConnectorMergeMaxCenterDistance = 256f;
    private const float ConnectorMergeCloseCenterDistance = 128f;

    public static Pm4CoordinateModeResolution ResolveCoordinateMode(
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        IReadOnlyList<Pm4MprlEntry> positionRefs,
        IReadOnlyList<Pm4MprlEntry>? anchorPositionRefs,
        int tileX,
        int tileY,
        Pm4AxisConvention axisConvention,
        Pm4CoordinateMode fallbackCoordinateMode)
    {
        Pm4PlanarTransform fallbackTransform = Pm4PlacementContract.GetDefaultPlanarTransform(fallbackCoordinateMode);
        IReadOnlyList<Pm4MprlEntry> scoringRefs = anchorPositionRefs is { Count: > 0 }
            ? anchorPositionRefs
            : positionRefs;

        if (surfaces.Count == 0 || scoringRefs.Count == 0)
            return new Pm4CoordinateModeResolution(fallbackCoordinateMode, fallbackTransform, float.PositiveInfinity, float.PositiveInfinity, true);

        List<Vector3> objectVertices = CollectSurfaceVertices(meshVertices, meshIndices, surfaces);
        if (objectVertices.Count == 0)
            return new Pm4CoordinateModeResolution(fallbackCoordinateMode, fallbackTransform, float.PositiveInfinity, float.PositiveInfinity, true);

        List<Vector3> sampledObjectVertices = SampleObjectVertices(objectVertices, 192);
        List<Vector2> referencePlanarPoints = BuildMprlPlanarPoints(scoringRefs);
        if (sampledObjectVertices.Count == 0 || referencePlanarPoints.Count == 0)
            return new Pm4CoordinateModeResolution(fallbackCoordinateMode, fallbackTransform, float.PositiveInfinity, float.PositiveInfinity, true);

        float tileLocalScore = EvaluateCoordinateMode(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            scoringRefs,
            sampledObjectVertices,
            referencePlanarPoints,
            tileX,
            tileY,
            axisConvention,
            Pm4CoordinateMode.TileLocal,
            out Pm4PlanarTransform tileLocalTransform);

        float worldSpaceScore = EvaluateCoordinateMode(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            scoringRefs,
            sampledObjectVertices,
            referencePlanarPoints,
            tileX,
            tileY,
            axisConvention,
            Pm4CoordinateMode.WorldSpace,
            out Pm4PlanarTransform worldSpaceTransform);

        if (!float.IsFinite(tileLocalScore) && !float.IsFinite(worldSpaceScore))
            return new Pm4CoordinateModeResolution(fallbackCoordinateMode, fallbackTransform, tileLocalScore, worldSpaceScore, true);

        if (!float.IsFinite(tileLocalScore))
            return new Pm4CoordinateModeResolution(Pm4CoordinateMode.WorldSpace, worldSpaceTransform, tileLocalScore, worldSpaceScore, false);

        if (!float.IsFinite(worldSpaceScore))
            return new Pm4CoordinateModeResolution(Pm4CoordinateMode.TileLocal, tileLocalTransform, tileLocalScore, worldSpaceScore, false);

        const float decisiveMargin = 512f;
        if (tileLocalScore + decisiveMargin < worldSpaceScore)
            return new Pm4CoordinateModeResolution(Pm4CoordinateMode.TileLocal, tileLocalTransform, tileLocalScore, worldSpaceScore, false);

        if (worldSpaceScore + decisiveMargin < tileLocalScore)
            return new Pm4CoordinateModeResolution(Pm4CoordinateMode.WorldSpace, worldSpaceTransform, tileLocalScore, worldSpaceScore, false);

        return fallbackCoordinateMode == Pm4CoordinateMode.TileLocal
            ? new Pm4CoordinateModeResolution(Pm4CoordinateMode.TileLocal, tileLocalTransform, tileLocalScore, worldSpaceScore, true)
            : new Pm4CoordinateModeResolution(Pm4CoordinateMode.WorldSpace, worldSpaceTransform, tileLocalScore, worldSpaceScore, true);
    }

    public static Pm4PlacementSolution ResolvePlacementSolution(
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        IReadOnlyList<Pm4MprlEntry> positionRefs,
        IReadOnlyList<Pm4MprlEntry>? anchorPositionRefs,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention)
    {
        Pm4PlanarTransform planarTransform = ResolvePlanarTransform(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            anchorPositionRefs,
            tileX,
            tileY,
            coordinateMode,
            axisConvention);

        Vector3 worldPivot = ComputeSurfaceWorldCentroid(
            meshVertices,
            meshIndices,
            surfaces,
            tileX,
            tileY,
            coordinateMode,
            axisConvention,
            planarTransform);

        IReadOnlyList<Pm4MprlEntry> scoringRefs = anchorPositionRefs is { Count: > 0 }
            ? anchorPositionRefs
            : positionRefs;

        float worldYawCorrectionRadians = TryComputeWorldYawCorrectionRadians(
            meshVertices,
            meshIndices,
            surfaces,
            scoringRefs,
            tileX,
            tileY,
            coordinateMode,
            axisConvention,
            planarTransform,
            out float resolvedYawCorrection)
            ? resolvedYawCorrection
            : 0f;

        return new Pm4PlacementSolution(
            tileX,
            tileY,
            coordinateMode,
            axisConvention,
            planarTransform,
            worldPivot,
            worldYawCorrectionRadians);
    }

    public static Pm4LinkedPositionRefSummary SummarizeLinkedPositionRefs(IReadOnlyList<Pm4MprlEntry> positionRefs)
    {
        if (positionRefs.Count == 0)
            return new Pm4LinkedPositionRefSummary(0, 0, 0, 0, 0, float.NaN, float.NaN, float.NaN);

        int normalCount = 0;
        int terminatorCount = 0;
        int floorMin = int.MaxValue;
        int floorMax = int.MinValue;
        float headingMinDegrees = float.PositiveInfinity;
        float headingMaxDegrees = float.NegativeInfinity;
        double sumSin = 0d;
        double sumCos = 0d;

        for (int index = 0; index < positionRefs.Count; index++)
        {
            Pm4MprlEntry positionRef = positionRefs[index];
            if (positionRef.Unk16 != 0)
            {
                terminatorCount++;
                continue;
            }

            normalCount++;
            floorMin = Math.Min(floorMin, positionRef.Unk14);
            floorMax = Math.Max(floorMax, positionRef.Unk14);

            float headingDegrees = DecodeRawMprlPackedAngleRadians(positionRef) * (180f / MathF.PI);
            headingMinDegrees = Math.Min(headingMinDegrees, headingDegrees);
            headingMaxDegrees = Math.Max(headingMaxDegrees, headingDegrees);

            float headingRadians = headingDegrees * (MathF.PI / 180f);
            sumSin += Math.Sin(headingRadians);
            sumCos += Math.Cos(headingRadians);
        }

        if (normalCount == 0)
            return new Pm4LinkedPositionRefSummary(positionRefs.Count, 0, terminatorCount, 0, 0, float.NaN, float.NaN, float.NaN);

        float headingMeanDegrees = (float)(Math.Atan2(sumSin, sumCos) * (180d / Math.PI));
        if (headingMeanDegrees < 0f)
            headingMeanDegrees += 360f;

        return new Pm4LinkedPositionRefSummary(
            positionRefs.Count,
            normalCount,
            terminatorCount,
            floorMin,
            floorMax,
            headingMinDegrees,
            headingMaxDegrees,
            headingMeanDegrees);
    }

    public static bool TryComputeWorldYawCorrectionRadians(
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        IReadOnlyList<Pm4MprlEntry> scoringRefs,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        out float yawCorrectionRadians)
    {
        yawCorrectionRadians = 0f;
        if (surfaces.Count == 0 || scoringRefs.Count < 2)
            return false;

        List<Vector3> objectVertices = CollectSurfaceVertices(meshVertices, meshIndices, surfaces);
        if (objectVertices.Count < 3)
            return false;

        if (!TryComputeExpectedMprlYawRadians(scoringRefs, out float expectedYaw))
            return false;

        if (!TryComputePlanarPrincipalYaw(objectVertices, tileX, tileY, coordinateMode, axisConvention, planarTransform, out float candidateYaw))
            return false;

        float delta = ComputeBestSignedYawDeltaWithBasisFallback(candidateYaw, expectedYaw);

        const float minimumMeaningfulYawCorrectionRadians = 12f * MathF.PI / 180f;
        if (MathF.Abs(delta) < minimumMeaningfulYawCorrectionRadians)
            return false;

        yawCorrectionRadians = delta;
        return true;
    }

    public static Vector3 ComputeSurfaceWorldCentroid(
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform)
    {
        List<Vector3> objectVertices = CollectSurfaceVertices(meshVertices, meshIndices, surfaces);
        if (objectVertices.Count == 0)
            return Vector3.Zero;

        Vector3 centroid = Vector3.Zero;
        for (int index = 0; index < objectVertices.Count; index++)
            centroid += objectVertices[index];
        centroid /= objectVertices.Count;

        return ConvertPm4VertexToWorld(centroid, tileX, tileY, coordinateMode, axisConvention, planarTransform);
    }

    public static Pm4PlanarTransform ResolvePlanarTransform(
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        IReadOnlyList<Pm4MprlEntry> positionRefs,
        IReadOnlyList<Pm4MprlEntry>? anchorPositionRefs,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention)
    {
        Pm4PlanarTransform defaultTransform = Pm4PlacementContract.GetDefaultPlanarTransform(coordinateMode);
        if (positionRefs.Count == 0 || surfaces.Count == 0)
            return defaultTransform;

        List<Vector3> objectVertices = CollectSurfaceVertices(meshVertices, meshIndices, surfaces);
        if (objectVertices.Count == 0)
            return defaultTransform;

        Vector3 centroid = Vector3.Zero;
        for (int index = 0; index < objectVertices.Count; index++)
            centroid += objectVertices[index];
        centroid /= objectVertices.Count;

        IReadOnlyList<Pm4MprlEntry> scoringRefs = anchorPositionRefs is { Count: > 0 }
            ? anchorPositionRefs
            : positionRefs;
        bool useFootprintScoring = anchorPositionRefs is { Count: >= 2 };
        List<Vector3> sampledObjectVertices = useFootprintScoring
            ? SampleObjectVertices(objectVertices, 256)
            : new List<Vector3>();
        List<Vector2> referencePlanarPoints = useFootprintScoring
            ? BuildMprlPlanarPoints(scoringRefs)
            : new List<Vector2>();
        bool hasExpectedYaw = TryComputeExpectedMprlYawRadians(scoringRefs, out float expectedYaw);

        Pm4PlanarTransform bestTransform = defaultTransform;
        float bestScore = float.MaxValue;
        float bestYawDelta = float.MaxValue;

        foreach (Pm4PlanarTransform candidate in Pm4PlacementContract.EnumeratePlanarTransforms(coordinateMode))
        {
            Vector3 candidateWorld = ConvertPm4VertexToWorld(centroid, tileX, tileY, coordinateMode, axisConvention, candidate);
            float centroidScore = NearestPositionRefDistanceSquared(scoringRefs, candidateWorld);
            float score = centroidScore;

            if (useFootprintScoring)
            {
                float footprintScore = ComputeMprlFootprintScore(
                    referencePlanarPoints,
                    sampledObjectVertices,
                    tileX,
                    tileY,
                    coordinateMode,
                    axisConvention,
                    candidate);
                if (float.IsFinite(footprintScore))
                    score = footprintScore * 0.85f + centroidScore * 0.15f;
            }

            float yawDelta = float.MaxValue;
            if (hasExpectedYaw
                && TryComputePlanarPrincipalYaw(objectVertices, tileX, tileY, coordinateMode, axisConvention, candidate, out float candidateYaw))
            {
                yawDelta = ComputeMprlYawDeltaWithQuarterTurnFallback(candidateYaw, expectedYaw);
            }

            if (candidate.InvertsWinding)
            {
                score += coordinateMode == Pm4CoordinateMode.TileLocal
                    ? (useFootprintScoring ? 4096f : 1024f)
                    : (useFootprintScoring ? 8192f : 4096f);
            }

            if (IsCandidateBetter(score, yawDelta, bestScore, bestYawDelta, useFootprintScoring, hasExpectedYaw))
            {
                bestScore = score;
                bestYawDelta = yawDelta;
                bestTransform = candidate;
            }
        }

        return bestTransform;
    }

    public static Pm4AxisConvention DetectAxisConventionByTriangleNormals(IReadOnlyList<Vector3> meshVertices, IReadOnlyList<uint> meshIndices)
    {
        Pm4AxisConvention[] candidates =
        [
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        ];

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionByTriangleNormals(meshVertices, meshIndices, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        return bestScore > 0f
            ? bestConvention
            : DetectAxisConventionByRanges(meshVertices);
    }

    public static Pm4AxisConvention DetectAxisConventionBySurfaceNormals(IReadOnlyList<Vector3> meshVertices, IReadOnlyList<uint> meshIndices, IReadOnlyList<Pm4MsurEntry> surfaces)
    {
        if (surfaces.Count == 0)
            return DetectAxisConventionByTriangleNormals(meshVertices, meshIndices);

        Pm4AxisConvention[] candidates =
        [
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        ];

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionBySurfaceNormals(meshVertices, meshIndices, surfaces, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        if (bestScore > 0f)
            return bestConvention;

        List<Vector3> groupVertices = CollectSurfaceVertices(meshVertices, meshIndices, surfaces);
        return groupVertices.Count > 0
            ? DetectAxisConventionByRanges(groupVertices)
            : DetectAxisConventionByRanges(meshVertices);
    }

    public static float ScoreAxisConventionByTriangleNormals(IReadOnlyList<Vector3> meshVertices, IReadOnlyList<uint> meshIndices, Pm4AxisConvention convention)
    {
        if (meshVertices.Count == 0 || meshIndices.Count < 3)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        Pm4PlanarTransform defaultTransform = Pm4PlacementContract.GetDefaultPlanarTransform(Pm4CoordinateMode.WorldSpace);

        for (int index = 0; index + 2 < meshIndices.Count && samples < maxSamples; index += 3)
        {
            int i0 = (int)meshIndices[index];
            int i1 = (int)meshIndices[index + 1];
            int i2 = (int)meshIndices[index + 2];
            if ((uint)i0 >= (uint)meshVertices.Count || (uint)i1 >= (uint)meshVertices.Count || (uint)i2 >= (uint)meshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(meshVertices[i0], 0, 0, Pm4CoordinateMode.WorldSpace, convention, defaultTransform);
            Vector3 b = ConvertPm4VertexToWorld(meshVertices[i1], 0, 0, Pm4CoordinateMode.WorldSpace, convention, defaultTransform);
            Vector3 c = ConvertPm4VertexToWorld(meshVertices[i2], 0, 0, Pm4CoordinateMode.WorldSpace, convention, defaultTransform);

            Vector3 normal = Vector3.Cross(b - a, c - a);
            float length = normal.Length();
            if (length < 1e-5f)
                continue;

            sum += MathF.Abs(normal.Z / length);
            samples++;
        }

        return samples > 0 ? sum / samples : 0f;
    }

    public static float ScoreAxisConventionBySurfaceNormals(IReadOnlyList<Vector3> meshVertices, IReadOnlyList<uint> meshIndices, IReadOnlyList<Pm4MsurEntry> surfaces, Pm4AxisConvention convention)
    {
        if (meshVertices.Count == 0 || meshIndices.Count < 3 || surfaces.Count == 0)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        Pm4PlanarTransform defaultTransform = Pm4PlacementContract.GetDefaultPlanarTransform(Pm4CoordinateMode.WorldSpace);

        for (int surfaceIndex = 0; surfaceIndex < surfaces.Count && samples < maxSamples; surfaceIndex++)
        {
            Pm4MsurEntry surface = surfaces[surfaceIndex];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            if (surface.IndexCount < 3 || firstIndex < 0 || endExclusive - firstIndex < 3)
                continue;

            int i0 = (int)meshIndices[firstIndex];
            if ((uint)i0 >= (uint)meshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(meshVertices[i0], 0, 0, Pm4CoordinateMode.WorldSpace, convention, defaultTransform);
            for (int index = firstIndex + 1; index + 1 < endExclusive && samples < maxSamples; index++)
            {
                int i1 = (int)meshIndices[index];
                int i2 = (int)meshIndices[index + 1];
                if ((uint)i1 >= (uint)meshVertices.Count || (uint)i2 >= (uint)meshVertices.Count)
                    continue;

                Vector3 b = ConvertPm4VertexToWorld(meshVertices[i1], 0, 0, Pm4CoordinateMode.WorldSpace, convention, defaultTransform);
                Vector3 c = ConvertPm4VertexToWorld(meshVertices[i2], 0, 0, Pm4CoordinateMode.WorldSpace, convention, defaultTransform);

                Vector3 normal = Vector3.Cross(b - a, c - a);
                float length = normal.Length();
                if (length < 1e-5f)
                    continue;

                sum += MathF.Abs(normal.Z / length);
                samples++;
            }
        }

        return samples > 0 ? sum / samples : 0f;
    }

    public static Pm4AxisConvention DetectAxisConventionByRanges(IReadOnlyList<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return Pm4AxisConvention.XYPlaneZUp;

        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int index = 0; index < vertices.Count; index++)
        {
            Vector3 value = vertices[index];
            if (value.X < minX) minX = value.X;
            if (value.Y < minY) minY = value.Y;
            if (value.Z < minZ) minZ = value.Z;
            if (value.X > maxX) maxX = value.X;
            if (value.Y > maxY) maxY = value.Y;
            if (value.Z > maxZ) maxZ = value.Z;
        }

        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        float rangeZ = maxZ - minZ;
        const float tieTolerance = 8f;

        if (rangeY + tieTolerance < rangeX && rangeY + tieTolerance < rangeZ)
            return Pm4AxisConvention.XZPlaneYUp;
        if (rangeZ + tieTolerance < rangeX && rangeZ + tieTolerance < rangeY)
            return Pm4AxisConvention.XYPlaneZUp;
        if (rangeX + tieTolerance < rangeY && rangeX + tieTolerance < rangeZ)
            return Pm4AxisConvention.YZPlaneXUp;

        return Pm4AxisConvention.XYPlaneZUp;
    }

    public static bool IsLikelyTileLocal(IReadOnlyList<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return false;

        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int index = 0; index < vertices.Count; index++)
        {
            Vector3 value = vertices[index];
            if (value.X < minX) minX = value.X;
            if (value.Y < minY) minY = value.Y;
            if (value.Z < minZ) minZ = value.Z;
            if (value.X > maxX) maxX = value.X;
            if (value.Y > maxY) maxY = value.Y;
            if (value.Z > maxZ) maxZ = value.Z;
        }

        const float tolerance = 64f;
        float tileSpan = Pm4CoordinateService.TileSize;

        bool xyLocal = minX >= -tolerance && minY >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxY <= tileSpan + tolerance;
        bool xzLocal = minX >= -tolerance && minZ >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;
        bool yzLocal = minY >= -tolerance && minZ >= -tolerance &&
                       maxY <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;

        return xyLocal || xzLocal || yzLocal;
    }

    public static Vector3 ConvertPm4VertexToWorld(
        Vector3 pm4Vertex,
        Pm4PlacementSolution placement)
    {
        return ConvertPm4VertexToWorld(
            pm4Vertex,
            placement.TileX,
            placement.TileY,
            placement.CoordinateMode,
            placement.AxisConvention,
            placement.PlanarTransform,
            placement.WorldPivot,
            placement.WorldYawCorrectionRadians);
    }

    public static Vector3 ConvertPm4VertexToWorld(
        Vector3 pm4Vertex,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform)
    {
        return ConvertPm4VertexToWorld(
            pm4Vertex,
            tileX,
            tileY,
            coordinateMode,
            axisConvention,
            planarTransform,
            worldPivot: null,
            worldYawCorrectionRadians: 0f);
    }

    public static Vector3 ConvertPm4VertexToWorld(
        Vector3 pm4Vertex,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3? worldPivot,
        float worldYawCorrectionRadians)
    {
        float localU;
        float localV;
        float localUp;

        switch (axisConvention)
        {
            case Pm4AxisConvention.XZPlaneYUp:
                localU = pm4Vertex.X;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.Y;
                break;
            case Pm4AxisConvention.YZPlaneXUp:
                localU = pm4Vertex.Y;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.X;
                break;
            case Pm4AxisConvention.XYPlaneZUp:
            default:
                localU = pm4Vertex.Y;
                localV = pm4Vertex.X;
                localUp = pm4Vertex.Z;
                break;
        }

        if (planarTransform.SwapPlanarAxes)
            (localU, localV) = (localV, localU);

        float tileSpan = Pm4CoordinateService.TileSize;
        float worldX;
        float worldY;

        if (coordinateMode == Pm4CoordinateMode.TileLocal)
        {
            float mappedU = planarTransform.InvertU ? tileSpan - localU : localU;
            float mappedV = planarTransform.InvertV ? tileSpan - localV : localV;

            worldX = tileY * tileSpan + mappedU;
            worldY = tileX * tileSpan + mappedV;
        }
        else
        {
            if (planarTransform.InvertU)
                localU = -localU;
            if (planarTransform.InvertV)
                localV = -localV;

            worldX = localU;
            worldY = localV;
        }

        Vector3 world = new(worldX, worldY, localUp);
        return worldPivot.HasValue
            ? RotateWorldAroundPivot(world, worldPivot.Value, worldYawCorrectionRadians)
            : world;
    }

    public static Vector3 RotateWorldAroundPivot(Vector3 world, Vector3 pivot, float yawRadians)
    {
        if (MathF.Abs(yawRadians) < 1e-6f)
            return world;

        float sin = MathF.Sin(yawRadians);
        float cos = MathF.Cos(yawRadians);
        float dx = world.X - pivot.X;
        float dy = world.Y - pivot.Y;

        float rx = dx * cos - dy * sin;
        float ry = dx * sin + dy * cos;
        return new Vector3(pivot.X + rx, pivot.Y + ry, world.Z);
    }

    public static IReadOnlyList<Pm4ConnectorKey> BuildConnectorKeys(
        IReadOnlyList<Vector3> exteriorVertices,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        Pm4PlacementSolution placement)
    {
        if (surfaces.Count == 0 || exteriorVertices.Count == 0)
            return Array.Empty<Pm4ConnectorKey>();

        HashSet<uint> distinctMdosIndices = new();
        HashSet<Pm4ConnectorKey> connectorKeys = new();
        List<Pm4ConnectorKey> ordered = new();

        for (int index = 0; index < surfaces.Count; index++)
        {
            uint mdosIndex = surfaces[index].MdosIndex;
            if (!distinctMdosIndices.Add(mdosIndex) || mdosIndex >= exteriorVertices.Count)
                continue;

            Vector3 connectorPoint = ConvertPm4VertexToWorld(exteriorVertices[(int)mdosIndex], placement);
            if (!float.IsFinite(connectorPoint.X) || !float.IsFinite(connectorPoint.Y) || !float.IsFinite(connectorPoint.Z))
                continue;

            Pm4ConnectorKey connectorKey = QuantizeConnectorKey(connectorPoint);
            if (connectorKeys.Add(connectorKey))
                ordered.Add(connectorKey);
        }

        ordered.Sort(static (a, b) =>
        {
            int compareX = a.X.CompareTo(b.X);
            if (compareX != 0)
                return compareX;

            int compareY = a.Y.CompareTo(b.Y);
            if (compareY != 0)
                return compareY;

            return a.Z.CompareTo(b.Z);
        });

        return ordered;
    }

    public static IReadOnlyDictionary<Pm4ObjectGroupKey, Pm4ObjectGroupKey> BuildMergedGroupMap(
        IReadOnlyList<Pm4ConnectorMergeCandidate> groups)
    {
        Dictionary<Pm4ObjectGroupKey, Pm4ObjectGroupKey> mergedGroupMap = new(groups.Count);
        if (groups.Count == 0)
            return mergedGroupMap;

        Dictionary<(int tileX, int tileY), List<int>> groupsByTile = new();
        for (int index = 0; index < groups.Count; index++)
        {
            Pm4ConnectorMergeCandidate group = groups[index];
            mergedGroupMap[group.Key] = group.Key;

            (int tileX, int tileY) tileKey = (group.Key.TileX, group.Key.TileY);
            if (!groupsByTile.TryGetValue(tileKey, out List<int>? tileGroupIndices))
            {
                tileGroupIndices = new List<int>();
                groupsByTile[tileKey] = tileGroupIndices;
            }

            tileGroupIndices.Add(index);
        }

        if (groups.Count <= 1)
            return mergedGroupMap;

        int[] parent = new int[groups.Count];
        for (int index = 0; index < parent.Length; index++)
            parent[index] = index;

        static int Find(int[] parentArray, int index)
        {
            while (parentArray[index] != index)
            {
                parentArray[index] = parentArray[parentArray[index]];
                index = parentArray[index];
            }

            return index;
        }

        static void Union(int[] parentArray, int a, int b)
        {
            int rootA = Find(parentArray, a);
            int rootB = Find(parentArray, b);
            if (rootA != rootB)
                parentArray[rootB] = rootA;
        }

        (int deltaX, int deltaY)[] neighborOffsets =
        [
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1)
        ];

        foreach ((int tileX, int tileY) tileKey in groupsByTile.Keys)
        {
            List<int> currentTileGroupIndices = groupsByTile[tileKey];
            for (int offsetIndex = 0; offsetIndex < neighborOffsets.Length; offsetIndex++)
            {
                (int deltaX, int deltaY) offset = neighborOffsets[offsetIndex];
                (int tileX, int tileY) neighborTileKey = (tileKey.tileX + offset.deltaX, tileKey.tileY + offset.deltaY);
                if (!groupsByTile.TryGetValue(neighborTileKey, out List<int>? neighborTileGroupIndices))
                    continue;

                for (int currentIndex = 0; currentIndex < currentTileGroupIndices.Count; currentIndex++)
                {
                    int currentGroupIndex = currentTileGroupIndices[currentIndex];
                    for (int neighborIndex = 0; neighborIndex < neighborTileGroupIndices.Count; neighborIndex++)
                    {
                        int neighborGroupIndex = neighborTileGroupIndices[neighborIndex];
                        if (ShouldMergeConnectorGroups(groups[currentGroupIndex], groups[neighborGroupIndex]))
                            Union(parent, currentGroupIndex, neighborGroupIndex);
                    }
                }
            }
        }

        Dictionary<int, List<int>> components = new();
        for (int index = 0; index < groups.Count; index++)
        {
            int root = Find(parent, index);
            if (!components.TryGetValue(root, out List<int>? members))
            {
                members = new List<int>();
                components[root] = members;
            }

            members.Add(index);
        }

        foreach (List<int> members in components.Values)
        {
            if (members.Count <= 1)
                continue;

            Pm4ObjectGroupKey canonicalKey = members
                .Select(index => groups[index].Key)
                .OrderBy(static key => key.TileX)
                .ThenBy(static key => key.TileY)
                .ThenBy(static key => key.Ck24)
                .First();

            for (int memberIndex = 0; memberIndex < members.Count; memberIndex++)
                mergedGroupMap[groups[members[memberIndex]].Key] = canonicalKey;
        }

        return mergedGroupMap;
    }

    public static List<Vector3> CollectSurfaceVertices(IReadOnlyList<Vector3> meshVertices, IReadOnlyList<uint> meshIndices, IReadOnlyList<Pm4MsurEntry> surfaces)
    {
        List<Vector3> vertices = new();
        HashSet<int> seen = new();

        for (int surfaceIndex = 0; surfaceIndex < surfaces.Count; surfaceIndex++)
        {
            Pm4MsurEntry surface = surfaces[surfaceIndex];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            if (surface.IndexCount <= 0 || firstIndex < 0 || endExclusive <= firstIndex)
                continue;

            for (int index = firstIndex; index < endExclusive; index++)
            {
                int vertexIndex = (int)meshIndices[index];
                if ((uint)vertexIndex >= (uint)meshVertices.Count)
                    continue;
                if (!seen.Add(vertexIndex))
                    continue;

                vertices.Add(meshVertices[vertexIndex]);
            }
        }

        return vertices;
    }

    private static bool IsCandidateBetter(float score, float yawDelta, float bestScore, float bestYawDelta, bool useFootprintScoring, bool hasExpectedYaw)
    {
        bool isBetterDistance = score < bestScore - 0.001f;
        float tieDistanceThreshold = useFootprintScoring ? 256f : 4096f;
        bool isNearDistance = MathF.Abs(score - bestScore) <= tieDistanceThreshold;
        bool isBetterYaw = yawDelta + 0.01f < bestYawDelta;
        bool yawCanOverrideDistance = useFootprintScoring
            && hasExpectedYaw
            && yawDelta + 0.02f < bestYawDelta
            && score <= bestScore + 1024f;

        return isBetterDistance || (isNearDistance && isBetterYaw) || yawCanOverrideDistance;
    }

    private static float EvaluateCoordinateMode(
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        IReadOnlyList<Pm4MprlEntry> positionRefs,
        IReadOnlyList<Pm4MprlEntry> scoringRefs,
        IReadOnlyList<Vector3> sampledObjectVertices,
        IReadOnlyList<Vector2> referencePlanarPoints,
        int tileX,
        int tileY,
        Pm4AxisConvention axisConvention,
        Pm4CoordinateMode coordinateMode,
        out Pm4PlanarTransform planarTransform)
    {
        planarTransform = ResolvePlanarTransform(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            scoringRefs,
            tileX,
            tileY,
            coordinateMode,
            axisConvention);

        float footprintScore = ComputeMprlFootprintScore(
            referencePlanarPoints,
            sampledObjectVertices,
            tileX,
            tileY,
            coordinateMode,
            axisConvention,
            planarTransform);

        if (!float.IsFinite(footprintScore))
            return float.MaxValue;

        Vector3 centroid = Vector3.Zero;
        for (int index = 0; index < sampledObjectVertices.Count; index++)
            centroid += sampledObjectVertices[index];
        centroid /= sampledObjectVertices.Count;

        Vector3 centroidWorld = ConvertPm4VertexToWorld(
            centroid,
            tileX,
            tileY,
            coordinateMode,
            axisConvention,
            planarTransform);
        float centroidScore = NearestPositionRefDistanceSquared(scoringRefs, centroidWorld);

        return footprintScore * 0.85f + centroidScore * 0.15f;
    }

    private static List<Vector3> SampleObjectVertices(IReadOnlyList<Vector3> objectVertices, int maxSamples)
    {
        List<Vector3> sampled = new();
        if (objectVertices.Count == 0)
            return sampled;

        int sampleCount = Math.Min(maxSamples, objectVertices.Count);
        int stride = Math.Max(1, objectVertices.Count / sampleCount);
        for (int index = 0; index < objectVertices.Count; index += stride)
            sampled.Add(objectVertices[index]);

        if (sampled.Count == 0)
            sampled.Add(objectVertices[0]);

        return sampled;
    }

    private static List<Vector2> BuildMprlPlanarPoints(IReadOnlyList<Pm4MprlEntry> positionRefs)
    {
        List<Vector2> points = new(positionRefs.Count);
        for (int index = 0; index < positionRefs.Count; index++)
        {
            Vector3 refWorld = ConvertMprlPositionToWorld(positionRefs[index].Position);
            points.Add(new Vector2(refWorld.X, refWorld.Y));
        }

        return points;
    }

    private static Vector3 ConvertMprlPositionToWorld(Vector3 refPos)
    {
        return new Vector3(refPos.X, refPos.Z, refPos.Y);
    }

    private static float NearestDistanceSquared(IReadOnlyList<Vector2> points, in Vector2 target)
    {
        float best = float.MaxValue;
        for (int index = 0; index < points.Count; index++)
        {
            Vector2 delta = points[index] - target;
            float distSq = delta.LengthSquared();
            if (distSq < best)
                best = distSq;
        }

        return best;
    }

    private static float ComputeMprlFootprintScore(
        IReadOnlyList<Vector2> referencePoints,
        IReadOnlyList<Vector3> sampledVertices,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform candidate)
    {
        if (referencePoints.Count == 0 || sampledVertices.Count == 0)
            return float.MaxValue;

        List<Vector2> candidatePoints = new(sampledVertices.Count);
        for (int index = 0; index < sampledVertices.Count; index++)
        {
            Vector3 world = ConvertPm4VertexToWorld(sampledVertices[index], tileX, tileY, coordinateMode, axisConvention, candidate);
            candidatePoints.Add(new Vector2(world.X, world.Y));
        }

        float sumObjectToRef = 0f;
        for (int index = 0; index < candidatePoints.Count; index++)
            sumObjectToRef += NearestDistanceSquared(referencePoints, candidatePoints[index]);

        float sumRefToObject = 0f;
        for (int index = 0; index < referencePoints.Count; index++)
            sumRefToObject += NearestDistanceSquared(candidatePoints, referencePoints[index]);

        float avgObjectToRef = sumObjectToRef / Math.Max(1, candidatePoints.Count);
        float avgRefToObject = sumRefToObject / Math.Max(1, referencePoints.Count);
        return avgObjectToRef + avgRefToObject;
    }

    private static float DecodeRawMprlPackedAngleRadians(Pm4MprlEntry positionRef)
    {
        return positionRef.Unk04 * (2f * MathF.PI / 65536f);
    }

    private static bool TryComputeExpectedMprlYawRadians(IReadOnlyList<Pm4MprlEntry> positionRefs, out float yawRadians)
    {
        yawRadians = 0f;
        if (positionRefs.Count == 0)
            return false;

        double sumSin = 0d;
        double sumCos = 0d;
        int count = 0;
        for (int index = 0; index < positionRefs.Count; index++)
        {
            float angleRadians = DecodeRawMprlPackedAngleRadians(positionRefs[index]);
            sumSin += Math.Sin(angleRadians);
            sumCos += Math.Cos(angleRadians);
            count++;
        }

        if (count == 0)
            return false;

        double length = Math.Sqrt(sumSin * sumSin + sumCos * sumCos);
        if (length < 1e-4)
            return false;

        yawRadians = (float)Math.Atan2(sumSin, sumCos);
        return true;
    }

    private static bool TryComputePlanarPrincipalYaw(
        IReadOnlyList<Vector3> objectVertices,
        int tileX,
        int tileY,
        Pm4CoordinateMode coordinateMode,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        out float yawRadians)
    {
        yawRadians = 0f;
        if (objectVertices.Count < 3)
            return false;

        int sampleCount = Math.Min(512, objectVertices.Count);
        int stride = Math.Max(1, objectVertices.Count / sampleCount);
        double meanX = 0d;
        double meanY = 0d;
        int used = 0;

        for (int index = 0; index < objectVertices.Count; index += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[index], tileX, tileY, coordinateMode, axisConvention, planarTransform);
            meanX += world.X;
            meanY += world.Y;
            used++;
        }

        if (used < 3)
            return false;

        meanX /= used;
        meanY /= used;

        double covXX = 0d;
        double covYY = 0d;
        double covXY = 0d;
        for (int index = 0; index < objectVertices.Count; index += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[index], tileX, tileY, coordinateMode, axisConvention, planarTransform);
            double dx = world.X - meanX;
            double dy = world.Y - meanY;
            covXX += dx * dx;
            covYY += dy * dy;
            covXY += dx * dy;
        }

        if (covXX + covYY < 1e-4)
            return false;

        yawRadians = 0.5f * (float)Math.Atan2(2.0 * covXY, covXX - covYY);
        return true;
    }

    private static float ComputeUndirectedAngleDelta(float a, float b)
    {
        float delta = MathF.Abs(a - b);
        while (delta > MathF.PI)
            delta -= 2f * MathF.PI;
        delta = MathF.Abs(delta);
        if (delta > MathF.PI * 0.5f)
            delta = MathF.PI - delta;

        return MathF.Abs(delta);
    }

    private static float ComputeMprlYawDeltaWithQuarterTurnFallback(float candidateYaw, float expectedYaw)
    {
        float bestDelta = ComputeUndirectedAngleDelta(candidateYaw, expectedYaw);
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw));

        const float quarterTurn = MathF.PI * 0.5f;
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, expectedYaw + quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, expectedYaw - quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw + quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw - quarterTurn));

        return bestDelta;
    }

    private static float NormalizeSignedRadians(float radians)
    {
        while (radians > MathF.PI)
            radians -= 2f * MathF.PI;
        while (radians < -MathF.PI)
            radians += 2f * MathF.PI;

        return radians;
    }

    private static float ComputeBestSignedYawDeltaWithBasisFallback(float candidateYaw, float expectedYaw)
    {
        const float quarterTurn = MathF.PI * 0.5f;
        float[] expectedCandidates =
        [
            expectedYaw,
            -expectedYaw,
            expectedYaw + quarterTurn,
            expectedYaw - quarterTurn,
            -expectedYaw + quarterTurn,
            -expectedYaw - quarterTurn
        ];

        float bestDelta = 0f;
        float bestAbsDelta = float.MaxValue;
        for (int index = 0; index < expectedCandidates.Length; index++)
        {
            float target = expectedCandidates[index];
            for (int parity = 0; parity < 2; parity++)
            {
                float orientedTarget = target + (parity == 0 ? 0f : MathF.PI);
                float delta = NormalizeSignedRadians(orientedTarget - candidateYaw);
                float absDelta = MathF.Abs(delta);
                if (absDelta < bestAbsDelta)
                {
                    bestAbsDelta = absDelta;
                    bestDelta = delta;
                }
            }
        }

        return bestDelta;
    }

    private static Pm4ConnectorKey QuantizeConnectorKey(Vector3 point)
    {
        return new Pm4ConnectorKey(
            (int)MathF.Round(point.X / ConnectorQuantizationUnits),
            (int)MathF.Round(point.Y / ConnectorQuantizationUnits),
            (int)MathF.Round(point.Z / ConnectorQuantizationUnits));
    }

    private static bool ShouldMergeConnectorGroups(Pm4ConnectorMergeCandidate a, Pm4ConnectorMergeCandidate b)
    {
        if (a.Key.TileX == b.Key.TileX && a.Key.TileY == b.Key.TileY)
            return false;

        if (Math.Abs(a.Key.TileX - b.Key.TileX) > 1 || Math.Abs(a.Key.TileY - b.Key.TileY) > 1)
            return false;

        if (a.ConnectorKeys.Count == 0 || b.ConnectorKeys.Count == 0)
            return false;

        float centerDistanceSquared = Vector3.DistanceSquared(a.Center, b.Center);
        bool boundsOverlap = BoundsOverlapExpanded(a.BoundsMin, a.BoundsMax, b.BoundsMin, b.BoundsMax, ConnectorMergeBoundsPadding);
        if (!boundsOverlap && centerDistanceSquared > ConnectorMergeMaxCenterDistance * ConnectorMergeMaxCenterDistance)
            return false;

        int sharedConnectorCount = CountSharedConnectorKeys(a.ConnectorKeys, b.ConnectorKeys);
        if (sharedConnectorCount == 0)
            return false;

        int minConnectorCount = Math.Min(a.ConnectorKeys.Count, b.ConnectorKeys.Count);
        float sharedRatio = sharedConnectorCount / (float)minConnectorCount;

        if (sharedConnectorCount >= 4)
            return true;

        if (sharedConnectorCount >= 2 && sharedRatio >= 0.5f)
            return true;

        return sharedConnectorCount >= 2
            && sharedRatio >= 0.35f
            && centerDistanceSquared <= ConnectorMergeCloseCenterDistance * ConnectorMergeCloseCenterDistance;
    }

    private static int CountSharedConnectorKeys(IReadOnlySet<Pm4ConnectorKey> a, IReadOnlySet<Pm4ConnectorKey> b)
    {
        IReadOnlySet<Pm4ConnectorKey> smaller = a.Count <= b.Count ? a : b;
        IReadOnlySet<Pm4ConnectorKey> larger = a.Count <= b.Count ? b : a;
        int shared = 0;

        foreach (Pm4ConnectorKey key in smaller)
        {
            if (larger.Contains(key))
                shared++;
        }

        return shared;
    }

    private static bool BoundsOverlapExpanded(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB, float padding)
    {
        return maxA.X + padding >= minB.X - padding
            && minA.X - padding <= maxB.X + padding
            && maxA.Y + padding >= minB.Y - padding
            && minA.Y - padding <= maxB.Y + padding
            && maxA.Z + padding >= minB.Z - padding
            && minA.Z - padding <= maxB.Z + padding;
    }

    private static float NearestPositionRefDistanceSquared(IReadOnlyList<Pm4MprlEntry> positionRefs, Vector3 world)
    {
        float best = float.MaxValue;
        for (int index = 0; index < positionRefs.Count; index++)
        {
            Vector3 refWorld = ConvertMprlPositionToWorld(positionRefs[index].Position);
            float dx = refWorld.X - world.X;
            float dy = refWorld.Y - world.Y;
            float distSq = dx * dx + dy * dy;
            if (distSq < best)
                best = distSq;
        }

        return best;
    }
}