using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;

internal sealed record Pm4AssetReferenceBuildResult(
    IReadOnlyList<Pm4AssetReferenceSignalRecord> Assets,
    IReadOnlyList<string> Warnings);

internal static class Pm4AssetReferenceSupport
{
    public static Pm4AssetReferenceBuildResult BuildFromPlacements(
        string placementsPath,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions archiveBootstrapOptions,
        int tileX,
        int tileY)
    {
        AdtPlacementCatalog placements = AdtPlacementReader.Read(placementsPath);
        string tileCoordinate = $"{tileX}_{tileY}";
        List<Pm4AssetReferenceSignalRecord> assets = new(placements.WorldModelPlacements.Count + placements.ModelPlacements.Count);
        List<string> warnings = [];

        foreach (AdtWorldModelPlacement placement in placements.WorldModelPlacements)
            assets.Add(BuildWmoAssetReference(placement, archiveRoot, archiveBootstrapOptions, tileCoordinate, warnings));

        foreach (AdtModelPlacement placement in placements.ModelPlacements)
            assets.Add(BuildM2AssetReference(placement, archiveRoot, archiveBootstrapOptions, tileCoordinate, warnings));

        return new Pm4AssetReferenceBuildResult(assets, warnings);
    }

    private static Pm4AssetReferenceSignalRecord BuildWmoAssetReference(
        AdtWorldModelPlacement placement,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions archiveBootstrapOptions,
        string tileCoordinate,
        List<string> warnings)
    {
        bool assetResolved = false;
        Vector3 worldBoundsMin = placement.BoundsMin;
        Vector3 worldBoundsMax = placement.BoundsMax;
        Vector2[] footprintHull = BuildAabbFootprintHull(worldBoundsMin, worldBoundsMax);
        float footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
        string validationTag = "fallback-placement-bounds";

        try
        {
            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(NormalizeVirtualPath(placement.ModelPath), [archiveRoot], archiveBootstrapOptions);
            using MemoryStream stream = new(bytes, writable: false);
            WmoSummary summary = WmoSummaryReader.Read(stream, placement.ModelPath);
            Matrix4x4 transform = BuildWmoTransform(placement.Position, placement.Rotation);
            TransformBounds(summary.BoundsMin, summary.BoundsMax, transform, out worldBoundsMin, out worldBoundsMax);
            footprintHull = BuildTransformedAabbFootprintHull(summary.BoundsMin, summary.BoundsMax, transform);
            footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
            assetResolved = true;
            validationTag = "resolved-wmo-summary";
        }
        catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
        {
            warnings.Add($"WMO placement {placement.UniqueId} '{placement.ModelPath}' fell back to placement bounds: {ex.Message}");
        }

        Vector3 center = (worldBoundsMin + worldBoundsMax) * 0.5f;
        return new Pm4AssetReferenceSignalRecord(
            $"wmo:{placement.UniqueId}",
            placement.ModelPath,
            "wmo",
            Path.GetFileName(Path.GetFullPath(archiveRoot)),
            [tileCoordinate],
            new Pm4Bounds3(worldBoundsMin, worldBoundsMax),
            center,
            footprintHull,
            footprintArea,
            placement.Position,
            placement.Rotation,
            1f,
            new Dictionary<string, int>(StringComparer.Ordinal)
            {
                ["assetKind:wmo"] = 1,
                ["geometry:resolved"] = assetResolved ? 1 : 0,
            },
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["boundsSpanX"] = worldBoundsMax.X - worldBoundsMin.X,
                ["boundsSpanY"] = worldBoundsMax.Y - worldBoundsMin.Y,
                ["boundsSpanZ"] = worldBoundsMax.Z - worldBoundsMin.Z,
            },
            Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
            null,
            [validationTag]);
    }

    private static Pm4AssetReferenceSignalRecord BuildM2AssetReference(
        AdtModelPlacement placement,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions archiveBootstrapOptions,
        string tileCoordinate,
        List<string> warnings)
    {
        bool assetResolved = false;
        Vector3 worldBoundsMin = placement.Position - new Vector3(2f);
        Vector3 worldBoundsMax = placement.Position + new Vector3(2f);
        Vector2[] footprintHull = BuildAabbFootprintHull(worldBoundsMin, worldBoundsMax);
        float footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
        string validationTag = "fallback-placement-bounds";

        try
        {
            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(NormalizeVirtualPath(placement.ModelPath), [archiveRoot], archiveBootstrapOptions);
            using MemoryStream stream = new(bytes, writable: false);
            MdxSummary summary = MdxSummaryReader.Read(stream, placement.ModelPath);
            Vector3? localBoundsMin = summary.Collision?.BoundsMin ?? summary.BoundsMin;
            Vector3? localBoundsMax = summary.Collision?.BoundsMax ?? summary.BoundsMax;
            if (localBoundsMin.HasValue && localBoundsMax.HasValue)
            {
                Matrix4x4 transform = BuildM2Transform(placement.Position, placement.Rotation, placement.Scale);
                TransformBounds(localBoundsMin.Value, localBoundsMax.Value, transform, out worldBoundsMin, out worldBoundsMax);
                footprintHull = BuildTransformedAabbFootprintHull(localBoundsMin.Value, localBoundsMax.Value, transform);
                footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
                assetResolved = true;
                validationTag = "resolved-m2-summary";
            }
        }
        catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
        {
            warnings.Add($"M2 placement {placement.UniqueId} '{placement.ModelPath}' fell back to placement bounds: {ex.Message}");
        }

        Vector3 center = (worldBoundsMin + worldBoundsMax) * 0.5f;
        return new Pm4AssetReferenceSignalRecord(
            $"m2:{placement.UniqueId}",
            placement.ModelPath,
            "m2",
            Path.GetFileName(Path.GetFullPath(archiveRoot)),
            [tileCoordinate],
            new Pm4Bounds3(worldBoundsMin, worldBoundsMax),
            center,
            footprintHull,
            footprintArea,
            placement.Position,
            placement.Rotation,
            placement.Scale,
            new Dictionary<string, int>(StringComparer.Ordinal)
            {
                ["assetKind:m2"] = 1,
                ["geometry:resolved"] = assetResolved ? 1 : 0,
            },
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["boundsSpanX"] = worldBoundsMax.X - worldBoundsMin.X,
                ["boundsSpanY"] = worldBoundsMax.Y - worldBoundsMin.Y,
                ["boundsSpanZ"] = worldBoundsMax.Z - worldBoundsMin.Z,
            },
            Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
            null,
            [validationTag]);
    }

    private static string NormalizeVirtualPath(string modelPath)
    {
        return modelPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }

    private static Matrix4x4 BuildM2Transform(Vector3 position, Vector3 rotationDegrees, float scale)
    {
        float rx = -rotationDegrees.Y * MathF.PI / 180f;
        float ry = -rotationDegrees.X * MathF.PI / 180f;
        float rz = rotationDegrees.Z * MathF.PI / 180f;

        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static Matrix4x4 BuildWmoTransform(Vector3 position, Vector3 rotationDegrees)
    {
        float rx = rotationDegrees.X * MathF.PI / 180f;
        float ry = rotationDegrees.Y * MathF.PI / 180f;
        float rz = rotationDegrees.Z * MathF.PI / 180f;

        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static void TransformBounds(Vector3 localMin, Vector3 localMax, Matrix4x4 transform, out Vector3 worldMin, out Vector3 worldMax)
    {
        Span<Vector3> corners = stackalloc Vector3[8]
        {
            new(localMin.X, localMin.Y, localMin.Z),
            new(localMin.X, localMin.Y, localMax.Z),
            new(localMin.X, localMax.Y, localMin.Z),
            new(localMin.X, localMax.Y, localMax.Z),
            new(localMax.X, localMin.Y, localMin.Z),
            new(localMax.X, localMin.Y, localMax.Z),
            new(localMax.X, localMax.Y, localMin.Z),
            new(localMax.X, localMax.Y, localMax.Z),
        };

        worldMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        worldMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        for (int index = 0; index < corners.Length; index++)
        {
            Vector3 world = Vector3.Transform(corners[index], transform);
            worldMin = Vector3.Min(worldMin, world);
            worldMax = Vector3.Max(worldMax, world);
        }
    }

    private static Vector2[] BuildTransformedAabbFootprintHull(Vector3 localMin, Vector3 localMax, Matrix4x4 transform)
    {
        return Pm4CorrelationMath.BuildTransformedFootprintHull(BuildAabbCorners(localMin, localMax), transform);
    }

    private static Vector2[] BuildAabbFootprintHull(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];
    }

    private static List<Vector3> BuildAabbCorners(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new(boundsMin.X, boundsMin.Y, boundsMin.Z),
            new(boundsMin.X, boundsMin.Y, boundsMax.Z),
            new(boundsMin.X, boundsMax.Y, boundsMin.Z),
            new(boundsMin.X, boundsMax.Y, boundsMax.Z),
            new(boundsMax.X, boundsMin.Y, boundsMin.Z),
            new(boundsMax.X, boundsMin.Y, boundsMax.Z),
            new(boundsMax.X, boundsMax.Y, boundsMin.Z),
            new(boundsMax.X, boundsMax.Y, boundsMax.Z),
        ];
    }
}
