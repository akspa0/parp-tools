using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;

namespace WoWViewer.Workbench.Services.Pm4Workbench;

public sealed record Pm4ShapeVisualPreview(
    IReadOnlyList<Vector2> NormalizedFootprintPolygon,
    IReadOnlyList<Vector2> NormalizedElevationProfile,
    Vector3 NormalizedBoxRatios,
    float PlanarAspectRatio,
    float HeightToLengthRatio,
    string ProportionSummary);

public sealed record Pm4ShapeLibraryVisualEntry(
    string EntryKey,
    string DisplayName,
    string Family,
    string AssetKind,
    string PrimaryModelPath,
    IReadOnlyList<string> AssociatedAssetPaths,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Dimensions,
    Pm4GeometricShapeDescriptor Descriptor,
    Pm4ShapeVisualPreview Preview,
    bool IsHealthy,
    string? HealthReason,
    bool IsCurated,
    int CorrelatedGroupsCount,
    float SolvedScale = 1.0f,
    float YawOffsetDegrees = 0f);

public sealed record Pm4ShapeLibraryFilter(
    string SearchText = "",
    string FamilyFilter = "",
    string AssetKindFilter = "",
    bool IncludeUnhealthy = true,
    bool CuratedOnly = false);

public sealed record Pm4ShapeLibraryComparison(
    Pm4ShapeLibraryVisualEntry Left,
    Pm4ShapeLibraryVisualEntry Right,
    Vector3 DimensionDelta,
    Vector3 DimensionRatio,
    float PlanarAspectDelta,
    float HeightRatioDelta,
    float DescriptorDistance,
    float SimilarityScore,
    string Summary);

/// <summary>
/// Spec 295 Phase 3: UI-independent visual shape library catalog, preview builder,
/// search/filter state (preserving selection across filter changes), health gating,
/// and side-by-side shape comparison.
/// </summary>
public sealed class Pm4ShapeLibraryVisualService
{
    private readonly List<Pm4ShapeLibraryVisualEntry> _allEntries = [];
    private readonly Dictionary<string, Pm4ShapeLibraryVisualEntry> _entriesByKey = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<Pm4ShapeLibraryVisualEntry> _filteredEntries = [];
    private readonly List<string> _availableFamilies = [];
    private readonly List<string> _availableAssetKinds = [];

    public IReadOnlyList<Pm4ShapeLibraryVisualEntry> AllEntries => _allEntries;
    public IReadOnlyList<Pm4ShapeLibraryVisualEntry> FilteredEntries => _filteredEntries;
    public IReadOnlyList<string> AvailableFamilies => _availableFamilies;
    public IReadOnlyList<string> AvailableAssetKinds => _availableAssetKinds;

    public Pm4ShapeLibraryFilter CurrentFilter { get; private set; } = new();
    public string? SelectedEntryKey { get; private set; }
    public string? ComparisonEntryKey { get; private set; }

    public Pm4ShapeLibraryVisualEntry? SelectedEntry =>
        SelectedEntryKey != null && _entriesByKey.TryGetValue(SelectedEntryKey, out var entry) ? entry : null;

    public Pm4ShapeLibraryVisualEntry? ComparisonEntry =>
        ComparisonEntryKey != null && _entriesByKey.TryGetValue(ComparisonEntryKey, out var entry) ? entry : null;

    public int HealthyCount => _allEntries.Count(static e => e.IsHealthy);
    public int UnhealthyCount => _allEntries.Count(static e => !e.IsHealthy);

    public void RebuildFromCorpus(
        Pm4PriorKnowledgeDatabase? database = null,
        IEnumerable<Pm4ModelShapeCandidate>? additionalCandidates = null,
        Func<string, (bool Exists, bool Readable, string? Reason)>? assetProbe = null)
    {
        database ??= Pm4PriorKnowledgeDatabase.Instance;
        Func<string, (bool Exists, bool Readable, string? Reason)> probe = assetProbe ?? DefaultAssetProbe;

        var rawItems = new List<(
            string ModelPath,
            string DisplayName,
            string AssetKind,
            string Family,
            Vector3 BoundsMin,
            Vector3 BoundsMax,
            Pm4GeometricShapeDescriptor? ExplicitDescriptor,
            int CorrelatedGroupsCount,
            bool IsCurated,
            float SolvedScale,
            float YawOffset)>();

        foreach (var asset in database.AllAssets)
        {
            rawItems.Add((
                asset.ModelPath,
                asset.ModelName,
                asset.AssetKind,
                asset.Family,
                asset.ApproximateBoundsMin,
                asset.ApproximateBoundsMax,
                null,
                asset.CorrelatedGroupsCount,
                asset.IsCurated,
                asset.SolvedScale,
                asset.YawOffset));
        }

        if (additionalCandidates != null)
        {
            foreach (var candidate in additionalCandidates)
            {
                if (string.IsNullOrWhiteSpace(candidate.ModelPath))
                    continue;

                Vector3 span = candidate.Descriptor?.BoundingBoxDimensions ?? new Vector3(10f, 10f, 10f);
                Vector3 min = new(-span.X * 0.5f, -span.Y * 0.5f, 0f);
                Vector3 max = new(span.X * 0.5f, span.Y * 0.5f, span.Z);
                string family = DetectFamily(candidate.ModelPath);

                rawItems.Add((
                    candidate.ModelPath,
                    Path.GetFileName(candidate.ModelPath),
                    candidate.AssetKind,
                    family,
                    min,
                    max,
                    candidate.Descriptor,
                    1,
                    false,
                    1.0f,
                    0f));
            }
        }

        // Group assets that share identical normalized paths, and also track sibling paths sharing identical quantized shape dimensions + family.
        var uniqueByPath = new Dictionary<string, (
            string ModelPath,
            string DisplayName,
            string AssetKind,
            string Family,
            Vector3 BoundsMin,
            Vector3 BoundsMax,
            Pm4GeometricShapeDescriptor? ExplicitDescriptor,
            int CorrelatedGroupsCount,
            bool IsCurated,
            float SolvedScale,
            float YawOffset)>(StringComparer.OrdinalIgnoreCase);

        foreach (var item in rawItems)
        {
            string normalizedPath = NormalizeAssetPath(item.ModelPath);
            if (string.IsNullOrWhiteSpace(normalizedPath))
                continue;

            if (!uniqueByPath.TryGetValue(normalizedPath, out var existing) || (item.IsCurated && !existing.IsCurated))
                uniqueByPath[normalizedPath] = item;
        }

        var shapeSignatureMap = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in uniqueByPath)
        {
            Vector3 dims = ComputeSortedDimensions(kvp.Value.BoundsMin, kvp.Value.BoundsMax, kvp.Value.ExplicitDescriptor);
            string sig = string.Create(
                CultureInfo.InvariantCulture,
                $"{kvp.Value.AssetKind}|{kvp.Value.Family}|{dims.X:F1}|{dims.Y:F1}|{dims.Z:F1}");
            if (!shapeSignatureMap.TryGetValue(sig, out var list))
            {
                list = [];
                shapeSignatureMap[sig] = list;
            }
            list.Add(kvp.Value.ModelPath);
        }

        _allEntries.Clear();
        _entriesByKey.Clear();

        foreach (var kvp in uniqueByPath.OrderByDescending(static x => x.Value.IsCurated).ThenBy(static x => x.Value.Family, StringComparer.OrdinalIgnoreCase).ThenBy(static x => x.Value.DisplayName, StringComparer.OrdinalIgnoreCase))
        {
            var item = kvp.Value;
            Vector3 dims = ComputeSortedDimensions(item.BoundsMin, item.BoundsMax, item.ExplicitDescriptor);
            Pm4GeometricShapeDescriptor descriptor = item.ExplicitDescriptor ?? BuildSyntheticDescriptor(dims);
            Pm4ShapeVisualPreview preview = BuildVisualPreview(dims, descriptor);

            string sig = string.Create(
                CultureInfo.InvariantCulture,
                $"{item.AssetKind}|{item.Family}|{dims.X:F1}|{dims.Y:F1}|{dims.Z:F1}");
            IReadOnlyList<string> associatedPaths = shapeSignatureMap.TryGetValue(sig, out var shared)
                ? shared
                : [item.ModelPath];

            Pm4ModelShapeCandidate candidateForHealth = new(item.ModelPath, item.AssetKind, descriptor);
            Pm4ShapeLibraryEntryHealth health = Pm4ShapeLibraryHealth.Check(candidateForHealth, probe);

            var entry = new Pm4ShapeLibraryVisualEntry(
                EntryKey: kvp.Key,
                DisplayName: string.IsNullOrWhiteSpace(item.DisplayName) ? Path.GetFileName(item.ModelPath) : item.DisplayName,
                Family: string.IsNullOrWhiteSpace(item.Family) ? "Generic" : item.Family,
                AssetKind: string.IsNullOrWhiteSpace(item.AssetKind) ? "M2" : item.AssetKind,
                PrimaryModelPath: item.ModelPath,
                AssociatedAssetPaths: associatedPaths,
                BoundsMin: item.BoundsMin,
                BoundsMax: item.BoundsMax,
                Dimensions: dims,
                Descriptor: descriptor,
                Preview: preview,
                IsHealthy: health.IsHealthy,
                HealthReason: health.Reason,
                IsCurated: item.IsCurated,
                CorrelatedGroupsCount: item.CorrelatedGroupsCount,
                SolvedScale: item.SolvedScale,
                YawOffsetDegrees: item.YawOffset);

            _allEntries.Add(entry);
            _entriesByKey[entry.EntryKey] = entry;
        }

        _availableFamilies.Clear();
        _availableFamilies.AddRange(_allEntries
            .Select(static e => e.Family)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(static f => f, StringComparer.OrdinalIgnoreCase));

        _availableAssetKinds.Clear();
        _availableAssetKinds.AddRange(_allEntries
            .Select(static e => e.AssetKind)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(static k => k, StringComparer.OrdinalIgnoreCase));

        if (SelectedEntryKey == null || !_entriesByKey.ContainsKey(SelectedEntryKey))
            SelectedEntryKey = _allEntries.FirstOrDefault()?.EntryKey;

        ApplyFilter(CurrentFilter);
    }

    public void ApplyFilter(Pm4ShapeLibraryFilter filter)
    {
        ArgumentNullException.ThrowIfNull(filter);
        CurrentFilter = filter;

        string search = filter.SearchText?.Trim() ?? string.Empty;
        string family = filter.FamilyFilter?.Trim() ?? string.Empty;
        string kind = filter.AssetKindFilter?.Trim() ?? string.Empty;

        _filteredEntries.Clear();
        foreach (var entry in _allEntries)
        {
            if (!filter.IncludeUnhealthy && !entry.IsHealthy)
                continue;

            if (filter.CuratedOnly && !entry.IsCurated)
                continue;

            if (!string.IsNullOrEmpty(family) &&
                !family.Equals("All", StringComparison.OrdinalIgnoreCase) &&
                !entry.Family.Equals(family, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!string.IsNullOrEmpty(kind) &&
                !kind.Equals("All", StringComparison.OrdinalIgnoreCase) &&
                !entry.AssetKind.Equals(kind, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!string.IsNullOrEmpty(search))
            {
                bool matchesName = entry.DisplayName.Contains(search, StringComparison.OrdinalIgnoreCase);
                bool matchesPath = entry.PrimaryModelPath.Contains(search, StringComparison.OrdinalIgnoreCase);
                bool matchesFamily = entry.Family.Contains(search, StringComparison.OrdinalIgnoreCase);
                bool matchesAssociated = entry.AssociatedAssetPaths.Any(p => p.Contains(search, StringComparison.OrdinalIgnoreCase));
                if (!matchesName && !matchesPath && !matchesFamily && !matchesAssociated)
                    continue;
            }

            _filteredEntries.Add(entry);
        }
        // Selection (SelectedEntryKey / ComparisonEntryKey) is intentionally preserved across filter changes (FR-003).
    }

    public bool SelectEntry(string? entryKey)
    {
        if (string.IsNullOrWhiteSpace(entryKey))
        {
            SelectedEntryKey = null;
            return false;
        }

        string normalized = NormalizeAssetPath(entryKey);
        if (_entriesByKey.ContainsKey(normalized))
        {
            SelectedEntryKey = normalized;
            return true;
        }

        if (_entriesByKey.ContainsKey(entryKey))
        {
            SelectedEntryKey = entryKey;
            return true;
        }

        return false;
    }

    public bool SetComparisonEntry(string? entryKey)
    {
        if (string.IsNullOrWhiteSpace(entryKey))
        {
            ComparisonEntryKey = null;
            return false;
        }

        string normalized = NormalizeAssetPath(entryKey);
        if (_entriesByKey.ContainsKey(normalized))
        {
            ComparisonEntryKey = normalized;
            return true;
        }

        if (_entriesByKey.ContainsKey(entryKey))
        {
            ComparisonEntryKey = entryKey;
            return true;
        }

        return false;
    }

    public Pm4ShapeLibraryComparison? GetActiveComparison()
    {
        if (SelectedEntry is not { } left || ComparisonEntry is not { } right)
            return null;

        return Compare(left, right);
    }

    public Pm4ShapeLibraryComparison Compare(Pm4ShapeLibraryVisualEntry left, Pm4ShapeLibraryVisualEntry right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        Vector3 delta = left.Dimensions - right.Dimensions;
        Vector3 ratio = new(
             left.Dimensions.X / MathF.Max(right.Dimensions.X, 0.01f),
             left.Dimensions.Y / MathF.Max(right.Dimensions.Y, 0.01f),
             left.Dimensions.Z / MathF.Max(right.Dimensions.Z, 0.01f));

        float planarDelta = left.Preview.PlanarAspectRatio - right.Preview.PlanarAspectRatio;
        float heightDelta = left.Preview.HeightToLengthRatio - right.Preview.HeightToLengthRatio;
        float distance = Pm4DirectModelMatcher.CompareShapes(left.Descriptor, right.Descriptor, new Pm4ShapeMatchingCriteria());
        float similarity = float.IsInfinity(distance) ? 0f : 1f / (1f + MathF.Max(0f, distance));

        string summary = string.Create(
            CultureInfo.InvariantCulture,
            $"{left.DisplayName} vs {right.DisplayName}: ΔL={delta.X:+0.0;-0.0;0.0} yd, ΔW={delta.Y:+0.0;-0.0;0.0} yd, ΔH={delta.Z:+0.0;-0.0;0.0} yd | Similarity={similarity:P0}");

        return new Pm4ShapeLibraryComparison(
            left,
            right,
            delta,
            ratio,
            planarDelta,
            heightDelta,
            distance,
            similarity,
            summary);
    }

    public IReadOnlyList<Pm4ModelShapeCandidate> GetHealthyModelCandidates()
    {
        var list = new List<Pm4ModelShapeCandidate>(_allEntries.Count);
        foreach (var entry in _allEntries)
        {
            if (!entry.IsHealthy || entry.AssetKind.Equals("PM4Hull", StringComparison.OrdinalIgnoreCase))
                continue;

            list.Add(new Pm4ModelShapeCandidate(entry.PrimaryModelPath, entry.AssetKind, entry.Descriptor));
        }
        return list;
    }

    public static Pm4ShapeVisualPreview BuildVisualPreview(Vector3 sortedDimensions, Pm4GeometricShapeDescriptor? descriptor = null)
    {
        float length = MathF.Max(sortedDimensions.X, 0.1f);
        float width = MathF.Max(sortedDimensions.Y, 0.1f);
        float height = MathF.Max(sortedDimensions.Z, 0.1f);
        float maxPlanar = MathF.Max(length, width);
        float max3D = MathF.Max(maxPlanar, height);

        float halfL = Math.Clamp(length / maxPlanar, 0.08f, 1.0f);
        float halfW = Math.Clamp(width / maxPlanar, 0.08f, 1.0f);

        IReadOnlyList<Vector2> footprint;
        if (descriptor?.FootprintHullVertices is { Count: >= 3 } hullVerts)
        {
            float minX = hullVerts.Min(static v => v.X);
            float maxX = hullVerts.Max(static v => v.X);
            float minY = hullVerts.Min(static v => v.Y);
            float maxY = hullVerts.Max(static v => v.Y);
            float cx = (minX + maxX) * 0.5f;
            float cy = (minY + maxY) * 0.5f;
            float span = MathF.Max(MathF.Max(maxX - minX, maxY - minY) * 0.5f, 0.05f);

            footprint = hullVerts
                .Select(v => new Vector2(
                    Math.Clamp((v.X - cx) / span, -1f, 1f),
                    Math.Clamp((v.Y - cy) / span, -1f, 1f)))
                .ToArray();
        }
        else
        {
            footprint =
            [
                new Vector2(-halfL, -halfW),
                new Vector2(halfL, -halfW),
                new Vector2(halfL, halfW),
                new Vector2(-halfL, halfW)
            ];
        }

        float normL = Math.Clamp(length / max3D, 0.08f, 1.0f);
        float normH = Math.Clamp(height / max3D, 0.08f, 1.0f);
        IReadOnlyList<Vector2> elevation =
        [
            new Vector2(-normL, -normH),
            new Vector2(normL, -normH),
            new Vector2(normL, normH),
            new Vector2(-normL, normH)
        ];

        Vector3 ratios = new(
            length / max3D,
            width / max3D,
            height / max3D);

        float planarAspect = length / width;
        float heightRatio = height / length;
        string summary = string.Create(
            CultureInfo.InvariantCulture,
            $"{length:F1} × {width:F1} × {height:F1} yd (L:W {planarAspect:F2}:1, H:L {heightRatio:F2}:1)");

        return new Pm4ShapeVisualPreview(
            footprint,
            elevation,
            ratios,
            planarAspect,
            heightRatio,
            summary);
    }

    public static Pm4GeometricShapeDescriptor BuildSyntheticDescriptor(Vector3 sortedDimensions)
    {
        float length = MathF.Max(sortedDimensions.X, 0.1f);
        float width = MathF.Max(sortedDimensions.Y, 0.1f);
        float height = MathF.Max(sortedDimensions.Z, 0.1f);

        Vector2[] hull =
        [
            new Vector2(-length * 0.5f, -width * 0.5f),
            new Vector2(length * 0.5f, -width * 0.5f),
            new Vector2(length * 0.5f, width * 0.5f),
            new Vector2(-length * 0.5f, width * 0.5f)
        ];

        Vector3 min = new(-length * 0.5f, -width * 0.5f, 0f);
        Vector3 max = new(length * 0.5f, width * 0.5f, height);

        return Pm4GeometricShapeDescriptor.FromBounds(min, max) with
        {
            PrismCentroid = new Vector3(0f, 0f, height * 0.5f),
            FootprintHullVertices = hull
        };
    }

    private static Vector3 ComputeSortedDimensions(Vector3 min, Vector3 max, Pm4GeometricShapeDescriptor? explicitDescriptor)
    {
        if (explicitDescriptor != null && explicitDescriptor.BoundingBoxDimensions.LengthSquared() > 1e-4f)
        {
            Vector3 d = explicitDescriptor.BoundingBoxDimensions;
            float l = MathF.Max(d.X, d.Y);
            float w = MathF.Min(d.X, d.Y);
            return new Vector3(MathF.Max(l, 0.1f), MathF.Max(w, 0.1f), MathF.Max(d.Z, 0.1f));
        }

        Vector3 span = max - min;
        float len = MathF.Max(MathF.Abs(span.X), MathF.Abs(span.Y));
        float wid = MathF.Min(MathF.Abs(span.X), MathF.Abs(span.Y));
        float hgt = MathF.Abs(span.Z);
        return new Vector3(MathF.Max(len, 0.1f), MathF.Max(wid, 0.1f), MathF.Max(hgt, 0.1f));
    }

    private static (bool Exists, bool Readable, string? Reason) DefaultAssetProbe(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return (false, false, "Empty model path.");

        if (modelPath.Contains("missing", StringComparison.OrdinalIgnoreCase) ||
            modelPath.Contains("corrupt", StringComparison.OrdinalIgnoreCase))
        {
            return (false, false, "Asset marked missing or unreadable.");
        }

        string ext = Path.GetExtension(modelPath);
        if (!ext.Equals(".wmo", StringComparison.OrdinalIgnoreCase) &&
            !ext.Equals(".m2", StringComparison.OrdinalIgnoreCase) &&
            !ext.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            return (true, false, $"Unsupported model extension '{ext}'.");
        }

        return (true, true, null);
    }

    private static string NormalizeAssetPath(string path) =>
        path.Trim().Replace('/', '\\');

    private static string DetectFamily(string path)
    {
        string p = path.ToLowerInvariant();
        if (p.Contains("wintergrasp") || p.Contains("wg_")) return "Wintergrasp";
        if (p.Contains("icecrown")) return "Icecrown";
        if (p.Contains("stormwind")) return "Stormwind";
        if (p.Contains("human") || p.Contains("farm") || p.Contains("barn") || p.Contains("inn")) return "Human";
        if (p.Contains("duskwood")) return "Duskwood";
        if (p.Contains("westfall")) return "Westfall";
        if (p.Contains("elwynn")) return "Elwynn";
        if (p.Contains("orc") || p.Contains("durotar") || p.Contains("barrens") || p.Contains("orgrimmar")) return "Orc";
        if (p.Contains("undead") || p.Contains("tirisfal") || p.Contains("silverpine")) return "Undead";
        if (p.Contains("dwarf") || p.Contains("khaz") || p.Contains("ironforge")) return "Dwarf";
        if (p.Contains("nightelf") || p.Contains("kalidar") || p.Contains("teldrassil") || p.Contains("ashenvale")) return "NightElf";
        return "Generic";
    }
}

