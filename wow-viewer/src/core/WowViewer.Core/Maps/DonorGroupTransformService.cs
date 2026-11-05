namespace WowViewer.Core.Maps;

/// <summary>A non-destructive transform requested for all tile placements in one donor group.</summary>
/// <param name="RotationDegrees">Clockwise angle in tile-grid coordinates.</param>
/// <param name="TranslationX">Translation along tile X (rows).</param>
/// <param name="TranslationY">Translation along tile Y (columns).</param>
/// <param name="PivotTileX">Optional pivot X. Omit both pivot coordinates to use the group centroid.</param>
/// <param name="PivotTileY">Optional pivot Y. Omit both pivot coordinates to use the group centroid.</param>
public readonly record struct DonorGroupTransform(
    float RotationDegrees,
    float TranslationX = 0f,
    float TranslationY = 0f,
    float? PivotTileX = null,
    float? PivotTileY = null);

public enum DonorGroupTransformIssueCode
{
    EmptyGroup,
    InvalidRotation,
    InvalidTranslation,
    InvalidPlacement,
    PivotOutsideGroup,
    OutOfBounds,
    TargetCollision,
    GroupGap,
}

public enum DonorGroupTransformIssueSeverity
{
    Information,
    Warning,
    Error,
}

/// <summary>Structured validation result; gap and pivot-fallback issues are non-blocking warnings.</summary>
public sealed record DonorGroupTransformIssue(
    DonorGroupTransformIssueCode Code,
    DonorGroupTransformIssueSeverity Severity,
    string Message,
    int? TileX = null,
    int? TileY = null,
    IReadOnlyList<int>? PlacementIndices = null);

/// <summary>
/// An isolated candidate layer plus the exact normalized transform and validation issues. The layer
/// is a clone; the source layer and source donor data are never mutated by preview creation.
/// </summary>
public sealed class DonorGroupTransformPreview
{
    internal DonorGroupTransformPreview(
        PhaseLayerSettings previewLayer,
        DonorGroupTransform appliedTransform,
        IReadOnlyList<DonorGroupTransformIssue> issues)
    {
        PreviewLayer = previewLayer;
        AppliedTransform = appliedTransform;
        Issues = issues;
    }

    public PhaseLayerSettings PreviewLayer { get; }

    /// <summary>The normalized transform with the resolved pivot, including the exact angle.</summary>
    public DonorGroupTransform AppliedTransform { get; }

    public IReadOnlyList<DonorGroupTransformIssue> Issues { get; }

    public bool CanAccept => !Issues.Any(static issue => issue.Severity == DonorGroupTransformIssueSeverity.Error);
}

/// <summary>
/// Computes a rigid donor-tile-group preview without UI or I/O. Tile-grid destinations are snapped
/// to integer indices after the continuous transform; the preview also retains the exact normalized
/// transform for later persistence and object-placement handling. Positive angles are clockwise,
/// matching <see cref="PhaseCompositionPolicy.ForwardTransformTile"/>.
/// </summary>
public static class DonorGroupTransformService
{
    private const int MapTileCount = 64;

    public static DonorGroupTransformPreview CreatePreview(
        PhaseLayerSettings sourceLayer,
        DonorGroupTransform requestedTransform)
    {
        ArgumentNullException.ThrowIfNull(sourceLayer);

        PhaseLayerSettings previewLayer = sourceLayer.Clone();
        var issues = new List<DonorGroupTransformIssue>();

        if (!float.IsFinite(requestedTransform.RotationDegrees))
        {
            issues.Add(Error(DonorGroupTransformIssueCode.InvalidRotation,
                "Rotation angle must be finite."));
            return new DonorGroupTransformPreview(previewLayer, requestedTransform, issues);
        }

        if (!float.IsFinite(requestedTransform.TranslationX) || !float.IsFinite(requestedTransform.TranslationY))
        {
            issues.Add(Error(DonorGroupTransformIssueCode.InvalidTranslation,
                "Group translation must be finite."));
            return new DonorGroupTransformPreview(previewLayer, requestedTransform, issues);
        }

        if (previewLayer.TilePlacements.Count == 0)
        {
            issues.Add(Error(DonorGroupTransformIssueCode.EmptyGroup,
                "Select at least one donor tile before transforming the group."));
            return new DonorGroupTransformPreview(previewLayer, Normalize(requestedTransform, 0f, 0f), issues);
        }

        var validIndices = new List<int>(previewLayer.TilePlacements.Count);
        for (int i = 0; i < previewLayer.TilePlacements.Count; i++)
        {
            if (previewLayer.TilePlacements[i].IsValid)
            {
                validIndices.Add(i);
                continue;
            }

            issues.Add(new DonorGroupTransformIssue(
                DonorGroupTransformIssueCode.InvalidPlacement,
                DonorGroupTransformIssueSeverity.Error,
                $"Placement {i} has a donor or target coordinate outside the 64×64 map.",
                PlacementIndices: [i]));
        }

        if (validIndices.Count == 0)
            return new DonorGroupTransformPreview(previewLayer, Normalize(requestedTransform, 0f, 0f), issues);

        (double centroidX, double centroidY, int minX, int maxX, int minY, int maxY) =
            GetGroupBounds(previewLayer, validIndices);
        (double pivotX, double pivotY) = ResolvePivot(
            requestedTransform, centroidX, centroidY, minX, maxX, minY, maxY, issues);

        float normalizedAngle = NormalizeDegrees(requestedTransform.RotationDegrees);
        DonorGroupTransform appliedTransform = Normalize(requestedTransform, (float)pivotX, (float)pivotY)
            with { RotationDegrees = normalizedAngle };
        previewLayer.GroupTransform = appliedTransform;

        double radians = normalizedAngle * (Math.PI / 180d);
        double cos = Math.Cos(radians);
        double sin = Math.Sin(radians);
        var destinations = new List<(int Index, int X, int Y)>(validIndices.Count);

        foreach (int index in validIndices)
        {
            PhaseTilePlacement placement = previewLayer.TilePlacements[index];
            double dx = placement.TargetTileX - pivotX;
            double dy = placement.TargetTileY - pivotY;
            double transformedX = pivotX + (dx * cos) - (dy * sin) + requestedTransform.TranslationX;
            double transformedY = pivotY + (dx * sin) + (dy * cos) + requestedTransform.TranslationY;

            if (!double.IsFinite(transformedX) || !double.IsFinite(transformedY) ||
                transformedX < 0d || transformedX > MapTileCount - 1 ||
                transformedY < 0d || transformedY > MapTileCount - 1)
            {
                issues.Add(new DonorGroupTransformIssue(
                    DonorGroupTransformIssueCode.OutOfBounds,
                    DonorGroupTransformIssueSeverity.Error,
                    $"Placement {index} transforms outside the 64×64 target map.",
                    PlacementIndices: [index]));
                continue;
            }

            int targetX = (int)Math.Round(transformedX, MidpointRounding.AwayFromZero);
            int targetY = (int)Math.Round(transformedY, MidpointRounding.AwayFromZero);
            destinations.Add((index, targetX, targetY));
            previewLayer.TilePlacements[index] = placement with
            {
                TargetTileX = targetX,
                TargetTileY = targetY,
            };
        }

        AddCollisionIssues(destinations, issues);
        AddGapIssue(destinations, issues);

        return new DonorGroupTransformPreview(previewLayer, appliedTransform, issues);
    }

    /// <summary>Canonicalizes finite angles into [0, 360); non-finite input is returned unchanged.</summary>
    public static float NormalizeDegrees(float degrees)
    {
        if (!float.IsFinite(degrees))
            return degrees;

        float normalized = degrees % 360f;
        if (normalized < 0f)
            normalized += 360f;
        return normalized == 0f ? 0f : normalized;
    }

    private static (double CentroidX, double CentroidY, int MinX, int MaxX, int MinY, int MaxY)
        GetGroupBounds(PhaseLayerSettings layer, IReadOnlyList<int> indices)
    {
        double sumX = 0d;
        double sumY = 0d;
        int minX = int.MaxValue;
        int maxX = int.MinValue;
        int minY = int.MaxValue;
        int maxY = int.MinValue;

        foreach (int index in indices)
        {
            PhaseTilePlacement placement = layer.TilePlacements[index];
            sumX += placement.TargetTileX;
            sumY += placement.TargetTileY;
            minX = Math.Min(minX, placement.TargetTileX);
            maxX = Math.Max(maxX, placement.TargetTileX);
            minY = Math.Min(minY, placement.TargetTileY);
            maxY = Math.Max(maxY, placement.TargetTileY);
        }

        return (sumX / indices.Count, sumY / indices.Count, minX, maxX, minY, maxY);
    }

    private static (double X, double Y) ResolvePivot(
        DonorGroupTransform requested,
        double centroidX,
        double centroidY,
        int minX,
        int maxX,
        int minY,
        int maxY,
        List<DonorGroupTransformIssue> issues)
    {
        bool hasX = requested.PivotTileX.HasValue;
        bool hasY = requested.PivotTileY.HasValue;
        bool customPivotValid = hasX && hasY &&
            float.IsFinite(requested.PivotTileX!.Value) && float.IsFinite(requested.PivotTileY!.Value) &&
            requested.PivotTileX.Value >= minX && requested.PivotTileX.Value <= maxX &&
            requested.PivotTileY.Value >= minY && requested.PivotTileY.Value <= maxY;

        if (customPivotValid)
            return (requested.PivotTileX!.Value, requested.PivotTileY!.Value);

        if (hasX || hasY)
        {
            issues.Add(new DonorGroupTransformIssue(
                DonorGroupTransformIssueCode.PivotOutsideGroup,
                DonorGroupTransformIssueSeverity.Warning,
                "The requested pivot is incomplete or outside the group's tile bounds; the group centroid is used."));
        }

        return (centroidX, centroidY);
    }

    private static void AddCollisionIssues(
        IReadOnlyList<(int Index, int X, int Y)> destinations,
        List<DonorGroupTransformIssue> issues)
    {
        foreach (IGrouping<(int X, int Y), (int Index, int X, int Y)> collision in
                 destinations.GroupBy(static item => (item.X, item.Y)).Where(static group => group.Count() > 1))
        {
            int[] indices = collision.Select(static item => item.Index).ToArray();
            issues.Add(new DonorGroupTransformIssue(
                DonorGroupTransformIssueCode.TargetCollision,
                DonorGroupTransformIssueSeverity.Error,
                $"{indices.Length} donor tiles would claim target ({collision.Key.X}, {collision.Key.Y}).",
                collision.Key.X,
                collision.Key.Y,
                indices));
        }
    }

    private static void AddGapIssue(
        IReadOnlyList<(int Index, int X, int Y)> destinations,
        List<DonorGroupTransformIssue> issues)
    {
        if (destinations.Count < 2)
            return;

        int minX = destinations.Min(static item => item.X);
        int maxX = destinations.Max(static item => item.X);
        int minY = destinations.Min(static item => item.Y);
        int maxY = destinations.Max(static item => item.Y);
        var occupied = destinations.Select(static item => (item.X, item.Y)).ToHashSet();
        var gaps = new List<(int X, int Y)>();

        for (int x = minX; x <= maxX; x++)
        for (int y = minY; y <= maxY; y++)
        {
            if (!occupied.Contains((x, y)))
                gaps.Add((x, y));
        }

        if (gaps.Count == 0)
            return;

        issues.Add(new DonorGroupTransformIssue(
            DonorGroupTransformIssueCode.GroupGap,
            DonorGroupTransformIssueSeverity.Warning,
            $"The transformed group's target footprint contains {gaps.Count} unoccupied tile position(s)."));
    }

    private static DonorGroupTransform Normalize(DonorGroupTransform transform, float pivotX, float pivotY)
        => transform with
        {
            RotationDegrees = NormalizeDegrees(transform.RotationDegrees),
            PivotTileX = pivotX,
            PivotTileY = pivotY,
        };

    private static DonorGroupTransformIssue Error(DonorGroupTransformIssueCode code, string message)
        => new(code, DonorGroupTransformIssueSeverity.Error, message);
}
