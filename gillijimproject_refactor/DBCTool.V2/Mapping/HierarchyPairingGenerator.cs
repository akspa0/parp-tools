using System;
using System.Collections.Generic;
using System.Linq;
using DBCTool.V2.IO;

namespace DBCTool.V2.Mapping;

public static class HierarchyPairingGenerator
{
    public sealed record HierarchySourceRecord(int MapId, string? MapName, IReadOnlyList<HierarchyNodeRecord> Nodes);

    public sealed record HierarchyNodeRecord(int AreaId, int ParentId, string Name);

    public static AreaHierarchyGraph BuildGraphFromRows(IEnumerable<HierarchySourceRecord> maps)
    {
        if (maps is null) throw new ArgumentNullException(nameof(maps));

        var mapList = new List<AreaHierarchyMap>();
        foreach (var map in maps)
        {
            var zoneById = new Dictionary<int, AreaHierarchyNode>();
            var nodesById = new Dictionary<int, AreaHierarchyNode>();

            foreach (var zone in map.Nodes.Where(n => n.ParentId == n.AreaId))
            {
                var node = new AreaHierarchyNode(map.MapId, zone.AreaId, zone.Name, zone.ParentId, isZone: true, parent: null);
                zoneById[zone.AreaId] = node;
                nodesById[zone.AreaId] = node;
            }

            foreach (var nodeRec in map.Nodes.Where(n => n.ParentId != n.AreaId))
            {
                if (!nodesById.TryGetValue(nodeRec.ParentId, out var parent))
                {
                    if (!zoneById.TryGetValue(nodeRec.ParentId, out parent))
                    {
                        parent = new AreaHierarchyNode(map.MapId, nodeRec.ParentId, string.Empty, nodeRec.ParentId, isZone: true, parent: null);
                        zoneById[nodeRec.ParentId] = parent;
                        nodesById[nodeRec.ParentId] = parent;
                    }
                }

                var child = new AreaHierarchyNode(map.MapId, nodeRec.AreaId, nodeRec.Name, nodeRec.ParentId, isZone: false, parent);
                parent.AddChild(child);
                nodesById[nodeRec.AreaId] = child;
            }

            var zones = zoneById.Values.ToList();
            mapList.Add(new AreaHierarchyMap(map.MapId, map.MapName ?? string.Empty, zones));
        }

        return new AreaHierarchyGraph(mapList);
    }

    public static IReadOnlyList<HierarchyPairingCandidate> GenerateCandidates(AreaHierarchyGraph srcGraph, AreaHierarchyGraph tgtGraph)
    {
        if (srcGraph is null) throw new ArgumentNullException(nameof(srcGraph));
        if (tgtGraph is null) throw new ArgumentNullException(nameof(tgtGraph));

        var indices = BuildIndex(tgtGraph);

        var results = new List<HierarchyPairingCandidate>();
        foreach (var srcNode in srcGraph.EnumerateNodes())
        {
            var variants = BuildVariants(srcNode);
            var dedupe = new HashSet<(int areaId, int mapId)>();
            var matches = new List<HierarchyPairingMatch>();

            if (variants.Count > 0 && indices.ByMap.TryGetValue(srcNode.MapId, out var mapIndex))
            {
                CollectMatches(mapIndex, variants, matches, dedupe, scope: "map");
            }

            if (variants.Count > 0)
            {
                CollectMatches(indices.Global, variants, matches, dedupe, scope: "global");
            }

            results.Add(new HierarchyPairingCandidate(
                srcNode.MapId,
                srcNode.AreaId,
                srcNode.Name,
                srcNode.ParentId,
                srcNode.IsZone,
                matches));
        }

        return results;
    }

    private static (Dictionary<int, Dictionary<string, List<AreaHierarchyNode>>> ByMap, Dictionary<string, List<AreaHierarchyNode>> Global) BuildIndex(AreaHierarchyGraph graph)
    {
        var index = new Dictionary<int, Dictionary<string, List<AreaHierarchyNode>>>();
        var global = new Dictionary<string, List<AreaHierarchyNode>>(StringComparer.OrdinalIgnoreCase);
        foreach (var map in graph.Maps)
        {
            var dict = new Dictionary<string, List<AreaHierarchyNode>>(StringComparer.OrdinalIgnoreCase);
            foreach (var node in map.Zones.SelectMany(z => z.SelfAndDescendants()))
            {
                foreach (var variant in BuildVariants(node))
                {
                    if (!dict.TryGetValue(variant.NormalizedKey, out var list))
                    {
                        list = new List<AreaHierarchyNode>();
                        dict[variant.NormalizedKey] = list;
                    }
                    list.Add(node);

                    if (!global.TryGetValue(variant.NormalizedKey, out var globalList))
                    {
                        globalList = new List<AreaHierarchyNode>();
                        global[variant.NormalizedKey] = globalList;
                    }
                    globalList.Add(node);
                }
            }
            index[map.MapId] = dict;
        }
        return (index, global);
    }

    private static void CollectMatches(Dictionary<string, List<AreaHierarchyNode>> index, IReadOnlyList<NameVariant> variants, List<HierarchyPairingMatch> matches, HashSet<(int areaId, int mapId)> dedupe, string scope)
    {
        foreach (var variant in variants)
        {
            if (!index.TryGetValue(variant.NormalizedKey, out var nodes)) continue;
            foreach (var node in nodes)
            {
                var key = (node.AreaId, node.MapId);
                if (dedupe.Contains(key)) continue;
                dedupe.Add(key);
                matches.Add(new HierarchyPairingMatch(
                    node.MapId,
                    node.AreaId,
                    node.Name,
                    node.Parent?.AreaId ?? node.AreaId,
                    $"{scope}_{variant.Reason}",
                    node.IsUnused));
            }
        }
    }

    internal static List<NameVariant> BuildVariants(AreaHierarchyNode node)
    {
        var dict = new Dictionary<string, NameVariant>(StringComparer.OrdinalIgnoreCase);
        void Add(string? value, string reason)
        {
            if (string.IsNullOrWhiteSpace(value)) return;
            var normalized = DbdcHelper.NormKey(value);
            if (string.IsNullOrEmpty(normalized)) return;
            if (!dict.ContainsKey(normalized))
            {
                dict[normalized] = new NameVariant(normalized, reason);
            }
        }

        Add(node.Name, "name");
        if (node.Parent is not null)
        {
            Add($"{node.Parent.Name}:{node.Name}", "parent_colon");
            Add($"{node.Parent.Name} {node.Name}", "parent_concat");
        }
        return dict.Values.ToList();
    }

    internal readonly record struct NameVariant(string NormalizedKey, string Reason);
}

public sealed record HierarchyPairingCandidate(
    int SrcMapId,
    int SrcAreaId,
    string SrcName,
    int SrcParentId,
    bool SrcIsZone,
    IReadOnlyList<HierarchyPairingMatch> Matches);

public sealed record HierarchyPairingMatch(
    int MapId,
    int AreaId,
    string Name,
    int ParentAreaId,
    string Reason,
    bool IsUnused);
