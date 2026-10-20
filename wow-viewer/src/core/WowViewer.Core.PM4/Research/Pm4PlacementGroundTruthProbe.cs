using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>One PM4 tile plus the authored placements from its companion <c>_obj0.adt</c>.</summary>
public sealed record Pm4PlacementAuditInput(
    string FileId,
    Pm4ResearchDocument Document,
    AdtPlacementCatalog? Placements);

public sealed record Pm4PlacementValueCount(string Value, long Count);

/// <summary>Feature distribution for one surface class (M2-only, WMO-only, both, neither).</summary>
public sealed record Pm4PlacementClassSummary(
    string Class,
    long Surfaces,
    long Ck24Zero,
    long Ck24NonZero,
    IReadOnlyList<Pm4PlacementValueCount> Ck24Types,
    IReadOnlyList<Pm4PlacementValueCount> GroupKeys,
    IReadOnlyList<Pm4PlacementValueCount> AttributeMasks,
    IReadOnlyList<Pm4PlacementValueCount> LinkTypes,
    double MeanVertexCount,
    double MeanArea,
    double MeanNormalZ);

/// <summary>How many authored placements are covered by PM4 surfaces, and how many surfaces cover each.</summary>
public sealed record Pm4PlacementCoverage(
    long M2Placements,
    long M2Covered,
    long M2CoveredBySingleSurface,
    long M2CoveredByMultipleSurfaces,
    long WmoPlacements,
    long WmoCovered,
    long WmoCoveredBySingleSurface,
    long WmoCoveredByMultipleSurfaces,
    long Surfaces,
    long SurfacesWithM2,
    long SurfacesWithWmo,
    long SurfacesWithBoth,
    long SurfacesWithNeither);

/// <summary>
/// For one feature value, how many surfaces carrying it fall in each placement class. A value that
/// is M2-exclusive is a candidate partition signal; a value spread across classes is not.
/// </summary>
public sealed record Pm4PlacementFeaturePurity(
    string Feature,
    string Value,
    long M2Only,
    long WmoOnly,
    long Both,
    long Neither);

/// <summary>A bounded, human-readable sample of one authored placement and the surfaces covering it.</summary>
public sealed record Pm4PlacementSample(
    string FileId,
    string ModelPath,
    Vector3 Position,
    float Scale,
    long CoveringSurfaces,
    string Ck24Types,
    string GroupKeys,
    string AttributeMasks,
    string LinkTypes);

public sealed record Pm4PlacementGroundTruthReport(
    int Files,
    int FilesWithPlacements,
    Pm4PlacementCoverage Coverage,
    IReadOnlyList<Pm4PlacementClassSummary> Classes,
    IReadOnlyList<Pm4PlacementFeaturePurity> Purity,
    IReadOnlyList<Pm4PlacementSample> M2Samples,
    IReadOnlyList<Pm4PlacementSample> WmoSamples);

/// <summary>
/// Works backwards from authored <c>_obj0.adt</c> placements: for every PM4 surface, decides whether
/// its 2D footprint covers an authored M2 or WMO placement, then compares the surface features that
/// the two classes carry. This is a correlation instrument, not an identity claim.
/// </summary>
/// <remarks>
/// The placement position is the model origin, not its full footprint, so "covered" means the
/// origin lies inside the surface's projected polygon (or within <see cref="BoundaryTolerance"/> of
/// its boundary). A surface that covers an M2 origin is evidence that the surface belongs to that
/// object's pathfinding mesh; it is not proof, because merged meshes can cover several objects.
/// </remarks>
public static class Pm4PlacementGroundTruthProbe
{
    public const float BoundaryTolerance = 0.05f;

    public static Pm4PlacementGroundTruthReport Analyze(IReadOnlyList<Pm4PlacementAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var classAccumulators = new Dictionary<string, ClassAccumulator>(StringComparer.Ordinal);
        var purity = new Dictionary<(string Feature, string Value), long[]>();
        var m2Samples = new List<Pm4PlacementSample>();
        var wmoSamples = new List<Pm4PlacementSample>();

        long m2Placements = 0, m2Covered = 0, m2Single = 0, m2Multiple = 0;
        long wmoPlacements = 0, wmoCovered = 0, wmoSingle = 0, wmoMultiple = 0;
        long surfaces = 0, withM2 = 0, withWmo = 0, withBoth = 0, withNeither = 0;
        int filesWithPlacements = 0;

        foreach (var file in input)
        {
            if (file.Placements is null)
                continue;
            filesWithPlacements++;

            var chunks = file.Document.KnownChunks;
            var surfacesInFile = BuildSurfaces(chunks);
            surfaces += surfacesInFile.Count;

            var m2 = file.Placements.ModelPlacements;
            var wmo = file.Placements.WorldModelPlacements;
            m2Placements += m2.Count;
            wmoPlacements += wmo.Count;

            var m2Hits = new int[surfacesInFile.Count];
            var wmoHits = new int[surfacesInFile.Count];

            foreach (var placement in m2)
            {
                var covering = CoveringSurfaces(surfacesInFile, placement.Position);
                if (covering.Count == 0)
                    continue;
                m2Covered++;
                if (covering.Count == 1) m2Single++; else m2Multiple++;
                foreach (int index in covering)
                    m2Hits[index]++;
                if (m2Samples.Count < 40)
                    m2Samples.Add(Sample(file.FileId, placement.ModelPath, placement.Position, placement.Scale, surfacesInFile, covering));
            }

            foreach (var placement in wmo)
            {
                var covering = CoveringSurfaces(surfacesInFile, placement.Position);
                if (covering.Count == 0)
                    continue;
                wmoCovered++;
                if (covering.Count == 1) wmoSingle++; else wmoMultiple++;
                foreach (int index in covering)
                    wmoHits[index]++;
                if (wmoSamples.Count < 40)
                    wmoSamples.Add(Sample(file.FileId, placement.ModelPath, placement.Position, 1f, surfacesInFile, covering));
            }

            for (int i = 0; i < surfacesInFile.Count; i++)
            {
                bool hasM2 = m2Hits[i] > 0;
                bool hasWmo = wmoHits[i] > 0;
                string cls = (hasM2, hasWmo) switch
                {
                    (true, true) => "both",
                    (true, false) => "m2-only",
                    (false, true) => "wmo-only",
                    _ => "neither",
                };
                if (hasM2) withM2++;
                if (hasWmo) withWmo++;
                if (hasM2 && hasWmo) withBoth++;
                if (!hasM2 && !hasWmo) withNeither++;

                var surface = surfacesInFile[i];
                if (!classAccumulators.TryGetValue(cls, out var acc))
                    classAccumulators[cls] = acc = new ClassAccumulator();
                acc.Add(surface);

                Observe(purity, "Ck24Type", surface.Ck24Type.ToString(), cls);
                Observe(purity, "GroupKey", surface.GroupKey.ToString(), cls);
                Observe(purity, "AttributeMask", surface.AttributeMask.ToString(), cls);
                foreach (string type in surface.LinkTypes)
                    Observe(purity, "LinkType", type, cls);
            }
        }

        var classes = classAccumulators
            .OrderBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => p.Value.ToResult(p.Key))
            .ToArray();

        var purityRows = purity
            .OrderBy(p => p.Key.Feature, StringComparer.Ordinal)
            .ThenByDescending(p => p.Value[0] + p.Value[1] + p.Value[2] + p.Value[3])
            .ThenBy(p => p.Key.Value, StringComparer.Ordinal)
            .Select(p => new Pm4PlacementFeaturePurity(
                p.Key.Feature, p.Key.Value, p.Value[0], p.Value[1], p.Value[2], p.Value[3]))
            .ToArray();

        return new Pm4PlacementGroundTruthReport(
            input.Count,
            filesWithPlacements,
            new Pm4PlacementCoverage(
                m2Placements, m2Covered, m2Single, m2Multiple,
                wmoPlacements, wmoCovered, wmoSingle, wmoMultiple,
                surfaces, withM2, withWmo, withBoth, withNeither),
            classes,
            purityRows,
            m2Samples,
            wmoSamples);
    }

    private static void Observe(Dictionary<(string Feature, string Value), long[]> purity, string feature, string value, string cls)
    {
        var key = (feature, value);
        if (!purity.TryGetValue(key, out var counts))
            purity[key] = counts = new long[4];
        counts[cls switch
        {
            "m2-only" => 0,
            "wmo-only" => 1,
            "both" => 2,
            _ => 3,
        }]++;
    }

    private static Pm4PlacementSample Sample(
        string fileId, string modelPath, Vector3 position, float scale,
        IReadOnlyList<SurfaceFeature> surfaces, IReadOnlyList<int> covering)
    {
        var types = new SortedSet<string>(StringComparer.Ordinal);
        var groups = new SortedSet<string>(StringComparer.Ordinal);
        var masks = new SortedSet<string>(StringComparer.Ordinal);
        var links = new SortedSet<string>(StringComparer.Ordinal);
        foreach (int index in covering)
        {
            var s = surfaces[index];
            types.Add(s.Ck24Type.ToString());
            groups.Add(s.GroupKey.ToString());
            masks.Add(s.AttributeMask.ToString());
            foreach (string t in s.LinkTypes) links.Add(t);
        }
        return new Pm4PlacementSample(
            fileId, modelPath, position, scale, covering.Count,
            string.Join(',', types), string.Join(',', groups), string.Join(',', masks), string.Join(',', links));
    }

    private static List<SurfaceFeature> BuildSurfaces(Pm4KnownChunkSet chunks)
    {
        var result = new List<SurfaceFeature>(chunks.Msur.Count);
        for (int i = 0; i < chunks.Msur.Count; i++)
        {
            var surface = chunks.Msur[i];
            if ((long)surface.MsviFirstIndex + surface.IndexCount > chunks.Msvi.Count)
                throw new InvalidDataException("Invalid surface geometry window.");

            var polygon = new List<Vector3>(surface.IndexCount);
            for (int j = 0; j < surface.IndexCount; j++)
            {
                uint vi = chunks.Msvi[(int)surface.MsviFirstIndex + j];
                if (vi >= chunks.Msvt.Count)
                    throw new InvalidDataException("Invalid surface vertex.");
                polygon.Add(Pm4CoordinateService.Pm4LocalToAdtPlacement(chunks.Msvt[(int)vi]));
            }

            long end = (long)surface._0x18 + surface.AttributeMask;
            if (end > chunks.Mslk.Count)
                throw new InvalidDataException("Invalid proposed MSUR/MSLK window.");
            var linkTypes = new SortedSet<string>(StringComparer.Ordinal);
            for (long k = surface._0x18; k < end; k++)
                linkTypes.Add(chunks.Mslk[(int)k].TypeFlags.ToString());

            result.Add(new SurfaceFeature(
                polygon.ToArray(),
                surface.Ck24Type,
                surface.GroupKey,
                surface.AttributeMask,
                linkTypes.ToArray(),
                surface.Normal.Z));
        }
        return result;
    }

    private static List<int> CoveringSurfaces(IReadOnlyList<SurfaceFeature> surfaces, Vector3 point)
    {
        var covering = new List<int>();
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (surfaces[i].Contains(point))
                covering.Add(i);
        }
        return covering;
    }

    private sealed class SurfaceFeature(
        Vector3[] polygon,
        byte ck24Type,
        byte groupKey,
        byte attributeMask,
        string[] linkTypes,
        float normalZ)
    {
        public byte Ck24Type { get; } = ck24Type;
        public byte GroupKey { get; } = groupKey;
        public byte AttributeMask { get; } = attributeMask;
        public string[] LinkTypes { get; } = linkTypes;
        public float NormalZ { get; } = normalZ;
        public int VertexCount => polygon.Length;
        public double Area { get; } = Area2D(polygon);

        public bool Contains(Vector3 p)
        {
            if (polygon.Length < 3)
                return false;

            bool inside = false;
            for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
            {
                Vector3 a = polygon[i], b = polygon[j];
                if (DistanceToSegmentSquared(p, a, b) <= BoundaryTolerance * BoundaryTolerance)
                    return true;
                bool crosses = (a.Y > p.Y) != (b.Y > p.Y)
                    && p.X < (b.X - a.X) * (p.Y - a.Y) / (b.Y - a.Y) + a.X;
                if (crosses)
                    inside = !inside;
            }
            return inside;
        }

        private static double Area2D(Vector3[] polygon)
        {
            if (polygon.Length < 3)
                return 0;
            double sum = 0;
            for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
                sum += (double)polygon[j].X * polygon[i].Y - (double)polygon[i].X * polygon[j].Y;
            return Math.Abs(sum) * 0.5;
        }

        private static float DistanceToSegmentSquared(Vector3 p, Vector3 a, Vector3 b)
        {
            float dx = b.X - a.X, dy = b.Y - a.Y;
            float lengthSquared = dx * dx + dy * dy;
            if (lengthSquared <= 1e-12f)
                return (p.X - a.X) * (p.X - a.X) + (p.Y - a.Y) * (p.Y - a.Y);
            float t = Math.Clamp(((p.X - a.X) * dx + (p.Y - a.Y) * dy) / lengthSquared, 0, 1);
            float cx = a.X + t * dx, cy = a.Y + t * dy;
            return (p.X - cx) * (p.X - cx) + (p.Y - cy) * (p.Y - cy);
        }
    }

    private sealed class ClassAccumulator
    {
        private readonly Dictionary<string, long> _types = new(StringComparer.Ordinal);
        private readonly Dictionary<string, long> _groups = new(StringComparer.Ordinal);
        private readonly Dictionary<string, long> _masks = new(StringComparer.Ordinal);
        private readonly Dictionary<string, long> _links = new(StringComparer.Ordinal);
        private long _surfaces, _zero, _nonZero, _vertices;
        private double _area, _normalZ;

        public void Add(SurfaceFeature s)
        {
            _surfaces++;
            if (s.Ck24Type == 0) _zero++; else _nonZero++;
            Increment(_types, s.Ck24Type.ToString());
            Increment(_groups, s.GroupKey.ToString());
            Increment(_masks, s.AttributeMask.ToString());
            foreach (string t in s.LinkTypes)
                Increment(_links, t);
            _vertices += s.VertexCount;
            _area += s.Area;
            _normalZ += s.NormalZ;
        }

        public Pm4PlacementClassSummary ToResult(string cls) => new(
            cls, _surfaces, _zero, _nonZero,
            Rows(_types), Rows(_groups), Rows(_masks), Rows(_links),
            _surfaces == 0 ? 0 : (double)_vertices / _surfaces,
            _surfaces == 0 ? 0 : _area / _surfaces,
            _surfaces == 0 ? 0 : _normalZ / _surfaces);

        private static void Increment(Dictionary<string, long> map, string key)
            => map[key] = map.TryGetValue(key, out long n) ? n + 1 : 1;

        private static IReadOnlyList<Pm4PlacementValueCount> Rows(Dictionary<string, long> map)
            => map.OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal)
                .Select(p => new Pm4PlacementValueCount(p.Key, p.Value)).ToArray();
    }
}
