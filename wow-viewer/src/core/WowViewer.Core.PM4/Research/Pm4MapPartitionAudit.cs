using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Read-only whole-map partition hypotheses. Raw field names are aliases, not semantic assertions.
/// No hypothesis is selected as an object partition and no asset match is produced.
/// </summary>
public static class Pm4MapPartitionAudit
{
    public static Pm4MapPartitionAuditReport Analyze(
        IReadOnlyList<Pm4MapAuditInput> files,
        IReadOnlyList<Pm4MapPlacementInput>? placementFiles = null,
        float geometryCellSize = 0.25f)
    {
        ArgumentNullException.ThrowIfNull(files);
        if (!float.IsFinite(geometryCellSize) || geometryCellSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(geometryCellSize));
        var ordered = files.OrderBy(f => f.FileId, StringComparer.Ordinal).ToArray();
        if (ordered.Select(f => f.FileId).Distinct(StringComparer.OrdinalIgnoreCase).Count() != ordered.Length)
            throw new ArgumentException("File IDs must be unique; local surface/link indices are not global IDs.", nameof(files));

        var nodes = new List<Pm4MapSurfaceNode>();
        var links = new List<Pm4MapLinkNode>();
        var triangles = new List<(Vector3 A, Vector3 B, Vector3 C)>[ordered.Sum(f => f.Document.KnownChunks.Msur.Count)];
        var cells = new Dictionary<(int X, int Y, int Z), HashSet<int>>();
        var nodeFiles = new List<int>();
        var surfaceOffsets = new int[ordered.Length];
        int invalidWindows = 0;
        for (int fileIndex = 0; fileIndex < ordered.Length; fileIndex++)
        {
            var file = ordered[fileIndex];
            var c = file.Document.KnownChunks;
            int surfaceBase = nodes.Count, linkBase = links.Count;
            surfaceOffsets[fileIndex] = surfaceBase;
            var owners = Enumerable.Range(0, c.Mslk.Count).Select(_ => new List<int>()).ToArray();
            for (int local = 0; local < c.Msur.Count; local++)
            {
                var s = c.Msur[local];
                int id = surfaceBase + local;
                long start = s._0x18, end = start + s.AttributeMask;
                bool validWindow = start <= c.Mslk.Count && end <= c.Mslk.Count;
                var owned = new List<int>();
                if (validWindow)
                {
                    for (long j = start; j < end; j++)
                    {
                        owners[(int)j].Add(id);
                        owned.Add(linkBase + (int)j);
                    }
                }
                else
                    invalidWindows++;

                var points = new List<Vector3>();
                int invalidIndices = 0;
                long vertexEnd = (long)s.MsviFirstIndex + s.IndexCount;
                if (vertexEnd > c.Msvi.Count)
                    invalidIndices = s.IndexCount;
                else
                    for (long j = s.MsviFirstIndex; j < vertexEnd; j++)
                    {
                        uint vi = c.Msvi[(int)j];
                        if (vi >= c.Msvt.Count) { invalidIndices++; continue; }
                        Vector3 p = c.Msvt[(int)vi];
                        points.Add(p);
                        if (!Finite(p)) { invalidIndices++; continue; }
                        var cell = (Cell(p.X, geometryCellSize), Cell(p.Y, geometryCellSize), Cell(p.Z, geometryCellSize));
                        if (!cells.TryGetValue(cell, out var incidence))
                            cells[cell] = incidence = [];
                        incidence.Add(id);
                    }
                // Never invent a polygon by closing across an invalid/missing corner.
                triangles[id] = [];
                if (invalidIndices == 0)
                    for (int j = 1; j + 1 < points.Count; j++)
                        triangles[id].Add((points[0], points[j], points[j + 1]));
                nodes.Add(new Pm4MapSurfaceNode(id, file.FileId, local, c.Mshd?.Field04, s, validWindow, owned, invalidIndices));
                nodeFiles.Add(fileIndex);
            }
            for (int local = 0; local < c.Mslk.Count; local++)
            {
                var link = c.Mslk[local];
                int? target = link.RefIndex < c.Msur.Count ? surfaceBase + link.RefIndex : null;
                bool pathValid = link.MspiFirstIndex >= 0
                    && (long)link.MspiFirstIndex + link.MspiIndexCount <= c.Mspi.Count;
                links.Add(new Pm4MapLinkNode(linkBase + local, file.FileId, local, link, owners[local],
                    target, link.RefIndex < c.Mprl.Count, pathValid));
            }
        }

        // This is a measured corpus address interpretation, not a renderer/reader rewrite.
        // Keep the old local-index reading as an explicit competing control.
        var tileFiles = new Dictionary<(int, int), int>();
        for (int f = 0; f < ordered.Length; f++)
            if (Pm4CoordinateService.TryParseTileCoordinates(ordered[f].FileId, out int x, out int y)
                && !tileFiles.TryAdd((x, y), f))
                throw new ArgumentException("Duplicate tile address within one map.");
        for (int i = 0; i < links.Count; i++)
        {
            var l = links[i];
            var tile = ((int)(l.Raw.LinkId & 255), (int)((l.Raw.LinkId >> 8) & 255));
            if (!tileFiles.TryGetValue(tile, out int f))
                links[i] = l with { AddressStatus = "TargetTileAbsentOrUnidentified" };
            else if (l.Raw.RefIndex >= ordered[f].Document.KnownChunks.Msur.Count)
                links[i] = l with { AddressedFileId = ordered[f].FileId, AddressStatus = "ReferenceOutsideAddressedTile" };
            else
                links[i] = l with { AddressedFileId = ordered[f].FileId,
                    AddressedSurfaceId = surfaceOffsets[f] + l.Raw.RefIndex, AddressStatus = "ResolvedByPackedTileHypothesis" };
        }
        var addressedEdges = links.Where(l => l.AddressedSurfaceId.HasValue)
            .SelectMany(l => l.WindowOwnerIds.Select(owner => new Pm4MapAuditEdge(owner, l.Id, l.AddressedSurfaceId!.Value))).ToArray();
        var edges = links.Where(l => l.SurfaceReferenceId.HasValue)
            .SelectMany(l => l.WindowOwnerIds.Select(owner => new Pm4MapAuditEdge(owner, l.Id, l.SurfaceReferenceId!.Value))).ToArray();
        var edgeSet = edges.Select(e => (e.OwnerSurfaceId, e.ReferencedSurfaceId)).ToHashSet();
        var cross = cells.Where(p => p.Value.Select(id => nodeFiles[id]).Distinct().Skip(1).Any())
            .OrderBy(p => p.Key.X).ThenBy(p => p.Key.Y).ThenBy(p => p.Key.Z)
            .Select(p => new Pm4MapGeometryIncidence(p.Key.X, p.Key.Y, p.Key.Z, p.Value.Order().ToArray())).ToArray();
        var hypotheses = new List<Pm4MapPartitionHypothesis>();

        Add("geometry-map", "Shared rounded XYZ cells across every file; may merge touching objects and terrain.",
            d => { foreach (var cell in cells.Values) Join(d, cell); });
        Add("window-ref-all", "MSUR window entries treated as edges to file-local RefIndex; semantic hypothesis.",
            d => { foreach (var e in edges) d.Union(e.OwnerSurfaceId, e.ReferencedSurfaceId); });
        Add("tile-addressed-window-ref-all", "RefIndex interpreted in tile (LinkId low byte, next byte), supported by whole-map boundary/back-link evidence; all raw kinds retained.",
            d => { foreach (var e in addressedEdges) d.Union(e.OwnerSurfaceId, e.ReferencedSurfaceId); });
        Add("tile-addressed-window-ref-low-bit-one", "Same addressed graph, only raw type low-bit set; observed direct-neighbour subset, not an object partition.",
            d => { foreach (var e in addressedEdges.Where(e => (links[e.LinkId].Raw.TypeFlags & 1) != 0)) d.Union(e.OwnerSurfaceId, e.ReferencedSurfaceId); });
        Add("window-ref-anchor", "Only negative path-start records; geometry-less links remain candidates.",
            d => { foreach (var e in edges.Where(e => links[e.LinkId].Raw.MspiFirstIndex < 0)) d.Union(e.OwnerSurfaceId, e.ReferencedSurfaceId); });
        Add("window-ref-path", "Only nonnegative path-start records; invalid path ranges retained in raw evidence.",
            d => { foreach (var e in edges.Where(e => links[e.LinkId].Raw.MspiFirstIndex >= 0)) d.Union(e.OwnerSurfaceId, e.ReferencedSurfaceId); });
        Add("window-ref-reciprocal", "Only reciprocal owner/reference pairs; not proof of object membership.",
            d => { foreach (var e in edges.Where(e => edgeSet.Contains((e.ReferencedSurfaceId, e.OwnerSurfaceId)))) d.Union(e.OwnerSurfaceId, e.ReferencedSurfaceId); });
        Add("window-ref-plus-cross-file-geometry", "All window/reference edges plus measured cross-file rounded-cell coincidences.",
            d => { foreach (var e in edges) d.Union(e.OwnerSurfaceId, e.ReferencedSurfaceId); foreach (var cell in cross) Join(d, cell.SurfaceIds); });
        Add("group-value-owner-file-scope", "Same raw GroupObjectId among window owners in one file, including zero; not an asserted ID.",
            d => { foreach (var g in links.GroupBy(l => (l.FileId, l.Raw.GroupObjectId))) Join(d, g.SelectMany(l => l.WindowOwnerIds)); });
        Add("group-value-owner-map-scope", "Same raw GroupObjectId among window owners across the map, including zero; tests accidental reuse.",
            d => { foreach (var g in links.GroupBy(l => l.Raw.GroupObjectId)) Join(d, g.SelectMany(l => l.WindowOwnerIds)); });
        Add("surface-attr-equality-file-scope", "Raw attrMask equality within each file; deliberately does not assume it is a count, flag or object ID.",
            d => { foreach (var g in nodes.GroupBy(n => (n.FileId, n.Raw.AttributeMask))) Join(d, g.Select(n => n.Id)); });
        Add("surface-group-attr-file-scope", "Current seed-style equality of raw GroupKey and attrMask; comparison baseline, not a semantic claim.",
            d => { foreach (var g in nodes.GroupBy(n => (n.FileId, n.Raw.GroupKey, n.Raw.AttributeMask))) Join(d, g.Select(n => n.Id)); });

        var fieldValues = new List<Pm4MapFieldValue>();
        foreach (var field in new (string Name, Func<Pm4MslkEntry, uint> Get)[]
        {
            ("_0x00-TypeFlags", l => l.TypeFlags), ("_0x01-Subtype", l => l.Subtype),
            ("_0x02-Padding", l => l.Padding), ("_0x04-GroupObjectId", l => l.GroupObjectId),
            ("LinkId", l => l.LinkId), ("RefIndex", l => l.RefIndex), ("SystemFlag", l => l.SystemFlag),
        })
            foreach (var g in links.GroupBy(l => field.Get(l.Raw)).OrderBy(g => g.Key))
                fieldValues.Add(new Pm4MapFieldValue(field.Name, g.Key, g.Select(l => l.Id).ToArray(),
                    g.SelectMany(l => l.WindowOwnerIds).Distinct().Order().ToArray(),
                    g.Where(l => l.SurfaceReferenceId.HasValue).Select(l => l.SurfaceReferenceId!.Value).Distinct().Order().ToArray(),
                    g.Select(l => l.FileId).Distinct(StringComparer.Ordinal).Count()));

        var surfaceFields = new List<Pm4MapSurfaceFieldValue>();
        foreach (var field in new (string Name, Func<Pm4MsurEntry, uint> Get)[]
        {
            ("_0x00-GroupKey", s => s.GroupKey), ("_0x01-IndexCount", s => s.IndexCount),
            ("_0x02-attrMask", s => s.AttributeMask), ("_0x03-Padding", s => s.Padding),
            ("_0x18", s => s._0x18), ("_0x1C", s => s.PackedParams),
        })
            foreach (var g in nodes.GroupBy(n => field.Get(n.Raw)).OrderBy(g => g.Key))
                surfaceFields.Add(new Pm4MapSurfaceFieldValue(field.Name, g.Key, g.Select(n => n.Id).ToArray()));
        for (int bit = 0; bit < 8; bit++)
        {
            uint mask = 1u << bit;
            surfaceFields.Add(new Pm4MapSurfaceFieldValue("_0x02-attrMask-bit-set", mask,
                nodes.Where(n => (n.Raw.AttributeMask & mask) != 0).Select(n => n.Id).ToArray()));
        }
        var windowControls = new List<Pm4MapWindowControl>();
        foreach (var length in new (string Name, Func<Pm4MsurEntry, int> Get)[]
        {
            ("attrMask-as-length", s => s.AttributeMask),
            ("IndexCount-as-length-control", s => s.IndexCount),
            ("attrMask-plus-one-control", s => s.AttributeMask + 1),
        })
        {
            int bad = 0, unowned = 0, multiple = 0, self = 0, fits = 0;
            foreach (var file in ordered)
            {
                var c = file.Document.KnownChunks;
                var coverage = new int[c.Mslk.Count];
                for (int i = 0; i < c.Msur.Count; i++)
                {
                    long start = c.Msur[i]._0x18, end = start + length.Get(c.Msur[i]);
                    if (start > c.Mslk.Count || end > c.Mslk.Count) { bad++; continue; }
                    for (long j = start; j < end; j++)
                    {
                        coverage[(int)j]++;
                        if (c.Mslk[(int)j].RefIndex == i) self++;
                        if (c.Mslk[(int)j].RefIndex < c.Msur.Count) fits++;
                    }
                }
                unowned += coverage.Count(n => n == 0);
                multiple += coverage.Count(n => n > 1);
            }
            windowControls.Add(new Pm4MapWindowControl(length.Name, bad, unowned, multiple, self, fits));
        }

        var authored = new List<Pm4MapPlacementReference>();
        foreach (var file in (placementFiles ?? []).OrderBy(f => f.FileId, StringComparer.Ordinal))
        {
            authored.AddRange(file.Catalog.ModelPlacements.Select(p => new Pm4MapPlacementReference(
                file.FileId, "M2", p.ModelPath, p.UniqueId, p.Position, p.Rotation, p.Scale)));
            authored.AddRange(file.Catalog.WorldModelPlacements.Select(p => new Pm4MapPlacementReference(
                file.FileId, "WMO", p.ModelPath, p.UniqueId, p.Position, p.Rotation, p.Scale / 1024f)));
        }
        var authoredFamilies = authored.Select((p, i) => (p, i)).Where(x => x.p.Kind == "M2")
            .GroupBy(x => x.p.ModelPath.Replace('\\', '/'), StringComparer.OrdinalIgnoreCase)
            .OrderBy(g => g.Key, StringComparer.OrdinalIgnoreCase)
            .Select(g => new Pm4MapAuthoredFamily(g.Key, g.Select(x => x.i).ToArray(),
                g.Select(x => x.p.Scale).Distinct().Order().ToArray(), g.Select(x => x.p.FileId).Distinct().Order().ToArray())).ToArray();

        return new Pm4MapPartitionAuditReport(
            "Hypotheses only. Parsing/domain fits/family similarity do not establish field semantics, M2 identity or partition accuracy.",
            ordered.Length, nodes.Count, links.Count, nodes.Count(n => n.Raw.Ck24 == 0), nodes.Count(n => n.Raw.PackedParams == 0),
            invalidWindows, links.Count(l => l.WindowOwnerIds.Count == 0), links.Count(l => !l.SurfaceReferenceId.HasValue),
            links.Count(l => l.Raw.MspiFirstIndex < 0), links.Count(l => l.Raw.MspiFirstIndex >= 0 && !l.PathWindowValid),
            nodes, links, edges, addressedEdges, cross, fieldValues, surfaceFields, windowControls, hypotheses, authored, authoredFamilies);

        void Add(string name, string assumption, Action<DisjointSet> build)
        {
            var d = new DisjointSet(nodes.Count);
            build(d);
            var components = Enumerable.Range(0, nodes.Count).GroupBy(d.Find).OrderBy(g => g.Min())
                .Select((g, i) =>
                {
                    int[] members = g.Order().ToArray();
                    return new Pm4MapPartitionComponent(i, members, members.Count(id => nodes[id].Raw.Ck24 == 0),
                        members.Count(id => nodes[id].Raw.PackedParams == 0), members.Select(id => nodes[id].FileId).Distinct().Order().ToArray(),
                        Pm4MapPartitionShape.Measure(members.SelectMany(id => triangles[id])));
                }).ToArray();
            var families = components.Where(c => c.ZeroCk24Surfaces > 0 && c.Shape.NormalizedSignature != null)
                .GroupBy(c => c.Shape.NormalizedSignature!, StringComparer.Ordinal).Where(g => g.Count() > 1)
                .OrderBy(g => g.Key, StringComparer.Ordinal)
                .Select(g => new Pm4MapScaleFamily(g.Key, g.Select(c => c.Id).ToArray(),
                    g.Min(c => c.Shape.RmsRadius), g.Max(c => c.Shape.RmsRadius),
                    g.Max(c => c.Shape.RmsRadius) / g.Min(c => c.Shape.RmsRadius),
                    g.SelectMany(c => c.Files).Distinct().Count())).ToArray();
            hypotheses.Add(new Pm4MapPartitionHypothesis(name, assumption, components, families));
        }
    }

    private static bool Finite(Vector3 p) => float.IsFinite(p.X) && float.IsFinite(p.Y) && float.IsFinite(p.Z);
    private static int Cell(float value, float size) => checked((int)MathF.Round(value / size));
    private static void Join(DisjointSet d, IEnumerable<int> members)
    {
        using var it = members.GetEnumerator();
        if (!it.MoveNext()) return;
        int first = it.Current;
        while (it.MoveNext()) d.Union(first, it.Current);
    }

    private sealed class DisjointSet
    {
        private readonly int[] parent;
        public DisjointSet(int count) => parent = Enumerable.Range(0, count).ToArray();
        public int Find(int id)
        {
            int root = id;
            while (parent[root] != root) root = parent[root];
            while (parent[id] != id) { int next = parent[id]; parent[id] = root; id = next; }
            return root;
        }
        public void Union(int a, int b) { int x = Find(a), y = Find(b); if (x != y) parent[Math.Max(x, y)] = Math.Min(x, y); }
    }
}
