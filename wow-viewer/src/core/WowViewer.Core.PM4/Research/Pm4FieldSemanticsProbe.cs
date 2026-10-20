using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4VersionRow(uint Version, int Files);

public sealed record Pm4MshdSpanRow(
    string FileId,
    uint Field00,
    uint Field04,
    uint Field08,
    float MsvtSpanX,
    float MsvtSpanY,
    float MspvSpanX,
    float MspvSpanY,
    float MscnSpanX,
    float MscnSpanY,
    float PlacementSpanX,
    float PlacementSpanY);

public sealed record Pm4SurfaceClassRow(
    byte Class,
    long Surfaces,
    long ZeroPlacementZ,
    double MeanPlacementZ,
    double MeanNormalZ,
    double MeanVertexCount,
    double MeanArea);

public sealed record Pm4MslkBitRow(
    byte Flags,
    long Records,
    long WithGeometry,
    long CrossTile,
    double MeanSubtype,
    double MeanGroupSize);

public sealed record Pm4PathKindRow(
    byte IndexCount,
    long Records,
    long Coplanar,
    long ZDominant,
    double MeanSpan);

public sealed record Pm4ChunkPresenceRow(string Signature, int Files);

public sealed record Pm4MprrRunRow(int Length, int Runs);

public sealed record Pm4FieldSemanticsReport(
    int Files,
    IReadOnlyList<Pm4VersionRow> Versions,
    IReadOnlyList<Pm4MshdSpanRow> MshdSpans,
    double MshdField00VsMsvtX,
    double MshdField00VsMsvtY,
    double MshdField08VsMsvtX,
    double MshdField08VsMsvtY,
    double MshdField00VsPlacementX,
    double MshdField00VsPlacementY,
    double MshdField08VsPlacementX,
    double MshdField08VsPlacementY,
    IReadOnlyList<Pm4SurfaceClassRow> SurfaceClasses,
    long SurfacesWithPolygon,
    long NormalMatchesPolygon,
    long NormalOppositePolygon,
    double NormalMedianAngleDegrees,
    long PlaneDistanceMatches,
    double PlaneDistanceMedianError,
    long SurfacesWithNonzeroPlacementZ,
    long SurfacesWithNonzeroPlacementZMatched,
    long GroupsWithOneModel,
    long GroupsWithMultipleModels,
    long GroupsWithNoModel,
    IReadOnlyList<Pm4MslkBitRow> MslkBits,
    long LinkIdHighHalfFFFF,
    long LinkIdHighHalfOther,
    IReadOnlyList<Pm4PathKindRow> PathKinds,
    long MprlCounterMatchesIndex,
    long MprlCounterMatchesReverseIndex,
    long MprlCounterOther,
    long MprrRuns,
    long MprrFirstTwoAreBounds,
    long MprrMiddleAllValidIndices,
    IReadOnlyList<Pm4MprrRunRow> MprrRunLengths,
    long MdbhCount,
    long MdbiCount,
    long MdbfCount,
    long MdosCount,
    long MdsfCount,
    long MdsfMsurInRange,
    long MdsfMdosInRange,
    IReadOnlyList<Pm4ChunkPresenceRow> ChunkPresence,
    long TallSurfaces,
    long TallNonZDominant,
    long ShortSurfaces,
    long ShortNonZDominant,
    long MslkNoGeometry,
    long MslkNoGeometryTargetResolved,
    long MslkNoGeometryTargetAbsent);

/// <summary>
/// Field-level tests for the open questions in the PM4 reference. Every test is corpus-wide and
/// reports counts, not identity claims.
/// </summary>
public static class Pm4FieldSemanticsProbe
{
    public static Pm4FieldSemanticsReport Analyze(IReadOnlyList<Pm4PlacementAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var versions = new Dictionary<uint, int>();
        var mshdRows = new List<Pm4MshdSpanRow>();
        var classStats = new Dictionary<byte, ClassAccumulator>();
        var bitStats = new Dictionary<byte, BitAccumulator>();
        var pathStats = new Dictionary<byte, PathAccumulator>();
        var chunkPresence = new Dictionary<string, int>(StringComparer.Ordinal);
        var mprrLengths = new Dictionary<int, int>();

        long surfacesWithPolygon = 0, normalMatches = 0, normalOpposite = 0;
        var normalAngles = new List<double>();
        long planeMatches = 0;
        var planeErrors = new List<double>();
        long nonzeroPlacementZ = 0, nonzeroPlacementZMatched = 0;
        long groupsOneModel = 0, groupsMultipleModels = 0, groupsNoModel = 0;
        long linkIdHighFFFF = 0, linkIdHighOther = 0;
        long counterIndex = 0, counterReverse = 0, counterOther = 0;
        long mprrRuns = 0, mprrFirstTwoBounds = 0, mprrMiddleValid = 0;
        long mdbh = 0, mdbi = 0, mdbf = 0, mdos = 0, mdsf = 0, mdsfMsur = 0, mdsfMdos = 0;
        long tall = 0, tallNonZ = 0, shortCount = 0, shortNonZ = 0;
        long noGeometry = 0, noGeometryResolved = 0, noGeometryAbsent = 0;

        var byTile = new Dictionary<(int X, int Y), Pm4ResearchDocument>();
        foreach (var file in input)
        {
            if (Pm4CoordinateService.TryParseTileCoordinates(file.FileId, out int tx, out int ty))
                byTile[(tx, ty)] = file.Document;
        }

        foreach (var file in input)
        {
            var doc = file.Document;
            var chunks = doc.KnownChunks;

            versions[doc.Version] = versions.GetValueOrDefault(doc.Version) + 1;

            foreach (var signature in doc.Chunks.Select(c => c.Signature).Distinct(StringComparer.Ordinal))
                chunkPresence[signature] = chunkPresence.GetValueOrDefault(signature) + 1;

            if (chunks.Mshd is not null)
            {
                var (msvtX, msvtY) = Span(chunks.Msvt, Pm4CoordinateService.Pm4LocalToAdtPlacement);
                var (mspvX, mspvY) = Span(chunks.Mspv, Pm4CoordinateService.Pm4LocalToAdtPlacement);
                var (mscnX, mscnY) = Span(chunks.Mscn, Pm4CoordinateService.Pm4LocalToAdtPlacement);
                var (placeX, placeY) = PlacementSpan(file.Placements);
                mshdRows.Add(new Pm4MshdSpanRow(
                    file.FileId, chunks.Mshd.Field00, chunks.Mshd.Field04, chunks.Mshd.Field08,
                    msvtX, msvtY, mspvX, mspvY, mscnX, mscnY, placeX, placeY));
            }

            var polygons = BuildPolygons(chunks);
            var placementZ = BuildPlacementZSet(file.Placements);

            for (int s = 0; s < chunks.Msur.Count; s++)
            {
                var surface = chunks.Msur[s];
                float placement = BitConverter.UInt32BitsToSingle(surface.PackedParams);
                var acc = classStats.TryGetValue(surface.GroupKey, out var existing)
                    ? existing
                    : classStats[surface.GroupKey] = new ClassAccumulator();
                acc.Add(surface, placement, polygons[s]);

                if (placement != 0f)
                {
                    nonzeroPlacementZ++;
                    if (placementZ.Contains(surface.PackedParams))
                        nonzeroPlacementZMatched++;
                }

                if (polygons[s].Length >= 3)
                {
                    surfacesWithPolygon++;
                    Vector3 normal = PolygonNormal(polygons[s]);
                    double angle = AngleDegrees(normal, surface.Normal);
                    normalAngles.Add(angle);
                    if (angle <= 5d) normalMatches++;
                    if (angle >= 175d) normalOpposite++;

                    double plane = Vector3.Dot(surface.Normal, polygons[s][0]);
                    double error = Math.Abs(plane - surface.Height);
                    planeErrors.Add(error);
                    if (error <= 0.01d) planeMatches++;

                    float minZ = polygons[s].Min(v => v.Z);
                    float maxZ = polygons[s].Max(v => v.Z);
                    bool isTall = maxZ - minZ > 5f;
                    bool nonZ = Math.Abs(surface.Normal.Z) <= 0.7f;
                    if (isTall) { tall++; if (nonZ) tallNonZ++; }
                    else { shortCount++; if (nonZ) shortNonZ++; }
                }
            }

            foreach (var link in chunks.Mslk)
            {
                var acc = bitStats.TryGetValue(link.TypeFlags, out var existing)
                    ? existing
                    : bitStats[link.TypeFlags] = new BitAccumulator();
                bool crossTile = (link.LinkId & 0xFF) != 0 || ((link.LinkId >> 8) & 0xFF) != 0;
                acc.Add(link, crossTile);

                if ((link.LinkId >> 16) == 0xFFFF) linkIdHighFFFF++;
                else linkIdHighOther++;

                if (link.MspiIndexCount == 0)
                {
                    noGeometry++;
                    int targetX = (int)(link.LinkId & 0xFF);
                    int targetY = (int)((link.LinkId >> 8) & 0xFF);
                    if (byTile.TryGetValue((targetX, targetY), out var target) && link.RefIndex < target.KnownChunks.Msur.Count)
                        noGeometryResolved++;
                    else
                        noGeometryAbsent++;
                }
                else
                {
                    var acc2 = pathStats.TryGetValue(link.MspiIndexCount, out var existing2)
                        ? existing2
                        : pathStats[link.MspiIndexCount] = new PathAccumulator();
                    acc2.Add(link, chunks);
                }
            }

            var groupOrder = new Dictionary<ushort, List<int>>();
            for (int i = 0; i < chunks.Mprl.Count; i++)
            {
                ushort id = chunks.Mprl[i].Unk04;
                if (!groupOrder.TryGetValue(id, out var list))
                    groupOrder[id] = list = new List<int>();
                list.Add(i);
            }
            foreach (var list in groupOrder.Values)
            {
                for (int k = 0; k < list.Count; k++)
                {
                    short value = chunks.Mprl[list[k]].Unk14;
                    if (value == k) counterIndex++;
                    else if (value == list.Count - 1 - k) counterReverse++;
                    else counterOther++;
                }
            }

            if (file.Placements is not null)
            {
                var models = file.Placements.ModelPlacements.Select(p => (p.Position, p.ModelPath))
                    .Concat(file.Placements.WorldModelPlacements.Select(p => (p.Position, p.ModelPath)))
                    .ToArray();
                foreach (var group in chunks.Mprl.GroupBy(p => p.Unk04))
                {
                    var modelSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    foreach (var point in group)
                    {
                        Vector3 placement = Pm4CoordinateService.MprlToAdtPlacement(point.Position);
                        foreach (var model in models)
                        {
                            if (Vector3.Distance(placement, model.Position) <= 25f)
                                modelSet.Add(model.ModelPath);
                        }
                    }
                    if (modelSet.Count == 0) groupsNoModel++;
                    else if (modelSet.Count == 1) groupsOneModel++;
                    else groupsMultipleModels++;
                }
            }

            var mprr = AnalyzeMprr(chunks, mprrLengths);
            mprrRuns += mprr.Runs;
            mprrFirstTwoBounds += mprr.FirstTwoBounds;
            mprrMiddleValid += mprr.MiddleValid;

            mdbh += chunks.Mdbh is null ? 0 : 1;
            mdbi += chunks.Mdbi.Count;
            mdbf += chunks.Mdbf.Count;
            mdos += chunks.Mdos.Count;
            mdsf += chunks.Mdsf.Count;
            foreach (var entry in chunks.Mdsf)
            {
                if (entry.MsurIndex < chunks.Msur.Count) mdsfMsur++;
                if (entry.MdosIndex < chunks.Mdos.Count) mdsfMdos++;
            }
        }

        return new Pm4FieldSemanticsReport(
            input.Count,
            versions.OrderByDescending(kv => kv.Value).Select(kv => new Pm4VersionRow(kv.Key, kv.Value)).ToList(),
            mshdRows,
            Pearson(mshdRows.Select(r => (double)r.Field00), mshdRows.Select(r => (double)r.MsvtSpanX)),
            Pearson(mshdRows.Select(r => (double)r.Field00), mshdRows.Select(r => (double)r.MsvtSpanY)),
            Pearson(mshdRows.Select(r => (double)r.Field08), mshdRows.Select(r => (double)r.MsvtSpanX)),
            Pearson(mshdRows.Select(r => (double)r.Field08), mshdRows.Select(r => (double)r.MsvtSpanY)),
            Pearson(mshdRows.Select(r => (double)r.Field00), mshdRows.Select(r => (double)r.PlacementSpanX)),
            Pearson(mshdRows.Select(r => (double)r.Field00), mshdRows.Select(r => (double)r.PlacementSpanY)),
            Pearson(mshdRows.Select(r => (double)r.Field08), mshdRows.Select(r => (double)r.PlacementSpanX)),
            Pearson(mshdRows.Select(r => (double)r.Field08), mshdRows.Select(r => (double)r.PlacementSpanY)),
            classStats.OrderBy(kv => kv.Key).Select(kv => kv.Value.ToRow(kv.Key)).ToList(),
            surfacesWithPolygon,
            normalMatches,
            normalOpposite,
            Median(normalAngles),
            planeMatches,
            Median(planeErrors),
            nonzeroPlacementZ,
            nonzeroPlacementZMatched,
            groupsOneModel,
            groupsMultipleModels,
            groupsNoModel,
            bitStats.OrderBy(kv => kv.Key).Select(kv => kv.Value.ToRow(kv.Key)).ToList(),
            linkIdHighFFFF,
            linkIdHighOther,
            pathStats.OrderBy(kv => kv.Key).Select(kv => kv.Value.ToRow(kv.Key)).ToList(),
            counterIndex,
            counterReverse,
            counterOther,
            mprrRuns,
            mprrFirstTwoBounds,
            mprrMiddleValid,
            mprrLengths.OrderBy(kv => kv.Key).Select(kv => new Pm4MprrRunRow(kv.Key, kv.Value)).ToList(),
            mdbh,
            mdbi,
            mdbf,
            mdos,
            mdsf,
            mdsfMsur,
            mdsfMdos,
            chunkPresence.OrderByDescending(kv => kv.Value).Select(kv => new Pm4ChunkPresenceRow(kv.Key, kv.Value)).ToList(),
            tall,
            tallNonZ,
            shortCount,
            shortNonZ,
            noGeometry,
            noGeometryResolved,
            noGeometryAbsent);
    }

    private static (long Runs, long FirstTwoBounds, long MiddleValid) AnalyzeMprr(
        Pm4KnownChunkSet chunks,
        Dictionary<int, int> lengths)
    {
        long runs = 0, firstTwoBounds = 0, middleValid = 0;
        var current = new List<Pm4MprrEntry>();
        void Flush()
        {
            if (current.Count == 0) return;
            runs++;
            lengths[current.Count] = lengths.GetValueOrDefault(current.Count) + 1;
            if (current.Count >= 3)
            {
                var middle = current.Skip(2).Select(r => (int)r.Value1).ToArray();
                int min = middle.Min();
                int max = middle.Max();
                if ((current[0].Value1 == min && current[1].Value1 == max) ||
                    (current[0].Value1 == max && current[1].Value1 == min))
                    firstTwoBounds++;
                if (middle.All(v => v < chunks.Mprl.Count))
                    middleValid++;
            }
            current.Clear();
        }
        foreach (var record in chunks.Mprr)
        {
            if (record.IsSentinel) Flush();
            else current.Add(record);
        }
        Flush();
        return (runs, firstTwoBounds, middleValid);
    }

    private static HashSet<uint> BuildPlacementZSet(AdtPlacementCatalog? catalog)
    {
        var set = new HashSet<uint>();
        if (catalog is null) return set;
        foreach (var placement in catalog.ModelPlacements)
            set.Add(BitConverter.SingleToUInt32Bits(placement.Position.Z));
        foreach (var placement in catalog.WorldModelPlacements)
            set.Add(BitConverter.SingleToUInt32Bits(placement.Position.Z));
        return set;
    }

    private static Vector3[][] BuildPolygons(Pm4KnownChunkSet chunks)
    {
        var polygons = new Vector3[chunks.Msur.Count][];
        for (int s = 0; s < chunks.Msur.Count; s++)
        {
            var surface = chunks.Msur[s];
            int start = (int)surface.MsviFirstIndex;
            int count = surface.IndexCount;
            if (start < 0 || count < 3 || start + count > chunks.Msvi.Count)
            {
                polygons[s] = [];
                continue;
            }
            var vertices = new Vector3[count];
            bool valid = true;
            for (int i = 0; i < count; i++)
            {
                uint index = chunks.Msvi[start + i];
                if (index >= chunks.Msvt.Count) { valid = false; break; }
                vertices[i] = chunks.Msvt[(int)index];
            }
            polygons[s] = valid ? vertices : [];
        }
        return polygons;
    }

    private static Vector3 PolygonNormal(Vector3[] polygon)
    {
        Vector3 a = polygon[1] - polygon[0];
        Vector3 b = polygon[2] - polygon[0];
        Vector3 normal = Vector3.Cross(a, b);
        float length = normal.Length();
        return length > 1e-6f ? normal / length : Vector3.Zero;
    }

    private static double AngleDegrees(Vector3 left, Vector3 right)
    {
        float dot = Vector3.Dot(Vector3.Normalize(left), Vector3.Normalize(right));
        dot = Math.Clamp(dot, -1f, 1f);
        return Math.Acos(dot) * 180d / Math.PI;
    }

    private static (float X, float Y) Span(IReadOnlyList<Vector3> points, Func<Vector3, Vector3> transform)
    {
        if (points.Count == 0) return (0f, 0f);
        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        foreach (var point in points)
        {
            var p = transform(point);
            minX = MathF.Min(minX, p.X); maxX = MathF.Max(maxX, p.X);
            minY = MathF.Min(minY, p.Y); maxY = MathF.Max(maxY, p.Y);
        }
        return (maxX - minX, maxY - minY);
    }

    private static (float X, float Y) PlacementSpan(AdtPlacementCatalog? catalog)
    {
        if (catalog is null) return (0f, 0f);
        var positions = catalog.ModelPlacements.Select(p => p.Position)
            .Concat(catalog.WorldModelPlacements.Select(p => p.Position))
            .ToArray();
        if (positions.Length == 0) return (0f, 0f);
        return (positions.Max(p => p.X) - positions.Min(p => p.X),
                positions.Max(p => p.Y) - positions.Min(p => p.Y));
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0d;
        values.Sort();
        return values[values.Count / 2];
    }

    private static double Pearson(IEnumerable<double> x, IEnumerable<double> y)
    {
        var xs = x.ToArray();
        var ys = y.ToArray();
        if (xs.Length < 2 || xs.Length != ys.Length) return 0d;
        double mx = xs.Average(), my = ys.Average();
        double numerator = 0d, dx = 0d, dy = 0d;
        for (int i = 0; i < xs.Length; i++)
        {
            double a = xs[i] - mx, b = ys[i] - my;
            numerator += a * b; dx += a * a; dy += b * b;
        }
        return dx > 0 && dy > 0 ? numerator / Math.Sqrt(dx * dy) : 0d;
    }

    private sealed class ClassAccumulator
    {
        private long _surfaces, _zero;
        private double _placementZ, _normalZ, _vertices, _area;

        public void Add(Pm4MsurEntry surface, float placementZ, Vector3[] polygon)
        {
            _surfaces++;
            if (placementZ == 0f) _zero++;
            _placementZ += placementZ;
            _normalZ += surface.Normal.Z;
            _vertices += surface.IndexCount;
            if (polygon.Length >= 3)
            {
                double area = 0d;
                for (int i = 1; i + 1 < polygon.Length; i++)
                    area += Vector3.Cross(polygon[i] - polygon[0], polygon[i + 1] - polygon[0]).Length() * 0.5d;
                _area += area;
            }
        }

        public Pm4SurfaceClassRow ToRow(byte key) => new(
            key, _surfaces, _zero,
            _surfaces == 0 ? 0 : _placementZ / _surfaces,
            _surfaces == 0 ? 0 : _normalZ / _surfaces,
            _surfaces == 0 ? 0 : _vertices / _surfaces,
            _surfaces == 0 ? 0 : _area / _surfaces);
    }

    private sealed class BitAccumulator
    {
        private long _records, _geometry, _crossTile, _subtype, _groupSize;

        public void Add(Pm4MslkEntry link, bool crossTile)
        {
            _records++;
            if (link.MspiIndexCount > 0) _geometry++;
            if (crossTile) _crossTile++;
            _subtype += link.Subtype;
            _groupSize += link.GroupObjectId;
        }

        public Pm4MslkBitRow ToRow(byte key) => new(
            key, _records, _geometry, _crossTile,
            _records == 0 ? 0 : (double)_subtype / _records,
            _records == 0 ? 0 : (double)_groupSize / _records);
    }

    private sealed class PathAccumulator
    {
        private long _records, _coplanar, _zDominant;
        private double _span;

        public void Add(Pm4MslkEntry link, Pm4KnownChunkSet chunks)
        {
            _records++;
            int start = link.MspiFirstIndex;
            int count = link.MspiIndexCount;
            if (start < 0 || count < 3 || start + count > chunks.Mspi.Count) return;
            var vertices = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                uint index = chunks.Mspi[start + i];
                if (index >= chunks.Mspv.Count) return;
                vertices[i] = chunks.Mspv[(int)index];
            }
            Vector3 normal = PolygonNormal(vertices);
            if (Math.Abs(normal.Z) > 0.7f) _zDominant++;
            double maxDistance = 0d;
            for (int i = 3; i < vertices.Length; i++)
                maxDistance = Math.Max(maxDistance, Math.Abs(Vector3.Dot(normal, vertices[i] - vertices[0])));
            if (maxDistance <= 0.01d) _coplanar++;
            _span += (vertices.Max(v => v.Z) - vertices.Min(v => v.Z));
        }

        public Pm4PathKindRow ToRow(byte key) => new(
            key, _records, _coplanar, _zDominant,
            _records == 0 ? 0 : _span / _records);
    }
}
