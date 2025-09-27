using System;
using System.Collections.Generic;
using System.Linq;

namespace DBCTool.V2.Mapping;

public sealed class AreaHierarchyGraph
{
    private readonly IReadOnlyList<AreaHierarchyMap> _maps;
    private readonly IReadOnlyDictionary<int, AreaHierarchyMap> _mapsById;
    private readonly IReadOnlyDictionary<int, AreaHierarchyNode> _nodesByAreaId;

    public AreaHierarchyGraph(IEnumerable<AreaHierarchyMap> maps)
    {
        if (maps is null) throw new ArgumentNullException(nameof(maps));
        _maps = maps.ToList();
        _mapsById = _maps.ToDictionary(m => m.MapId);

        var globalNodes = new Dictionary<int, AreaHierarchyNode>();
        foreach (var map in _maps)
        {
            foreach (var node in map.EnumerateNodes())
            {
                if (!globalNodes.ContainsKey(node.AreaId))
                {
                    globalNodes[node.AreaId] = node;
                }
            }
        }
        _nodesByAreaId = globalNodes;
    }

    public IReadOnlyList<AreaHierarchyMap> Maps => _maps;

    public IReadOnlyDictionary<int, AreaHierarchyMap> MapsById => _mapsById;

    public IReadOnlyDictionary<int, AreaHierarchyNode> NodesByAreaId => _nodesByAreaId;

    public AreaHierarchyNode? FindArea(int areaId) => _nodesByAreaId.TryGetValue(areaId, out var node) ? node : null;

    public IEnumerable<AreaHierarchyNode> EnumerateNodes()
    {
        foreach (var map in _maps)
        {
            foreach (var zone in map.Zones)
            {
                foreach (var node in zone.SelfAndDescendants())
                {
                    yield return node;
                }
            }
        }
    }
}

public sealed class AreaHierarchyMap
{
    private readonly IReadOnlyList<AreaHierarchyNode> _zones;
    private readonly IReadOnlyDictionary<int, AreaHierarchyNode> _nodesByAreaId;

    public AreaHierarchyMap(int mapId, string mapName, IEnumerable<AreaHierarchyNode> zones)
    {
        MapId = mapId;
        MapName = mapName ?? string.Empty;
        if (zones is null) throw new ArgumentNullException(nameof(zones));
        var zoneList = zones.ToList();
        _zones = zoneList;
        var dict = new Dictionary<int, AreaHierarchyNode>();
        foreach (var zone in zoneList)
        {
            foreach (var node in zone.SelfAndDescendants())
            {
                if (!dict.ContainsKey(node.AreaId))
                {
                    dict[node.AreaId] = node;
                }
            }
        }
        _nodesByAreaId = dict;
    }

    public int MapId { get; }

    public string MapName { get; }

    public IReadOnlyList<AreaHierarchyNode> Zones => _zones;

    public IReadOnlyDictionary<int, AreaHierarchyNode> NodesByAreaId => _nodesByAreaId;

    public AreaHierarchyNode? FindArea(int areaId) => _nodesByAreaId.TryGetValue(areaId, out var node) ? node : null;

    internal IEnumerable<AreaHierarchyNode> EnumerateNodes() => _zones.SelectMany(z => z.SelfAndDescendants());
}

public sealed class AreaHierarchyNode
{
    private readonly List<AreaHierarchyNode> _children = new();

    internal AreaHierarchyNode(int mapId, int areaId, string name, int parentId, bool isZone, AreaHierarchyNode? parent)
    {
        MapId = mapId;
        AreaId = areaId;
        Name = name ?? string.Empty;
        ParentId = parentId;
        IsZone = isZone;
        Parent = parent;
    }

    public int MapId { get; }

    public int AreaId { get; }

    public string Name { get; }

    public int ParentId { get; }

    public bool IsZone { get; }

    public AreaHierarchyNode? Parent { get; private set; }

    public IReadOnlyList<AreaHierarchyNode> Children => _children;

    public bool IsUnused => Name.IndexOf("UNUSED", StringComparison.OrdinalIgnoreCase) >= 0;

    internal void AddChild(AreaHierarchyNode child)
    {
        if (child is null) throw new ArgumentNullException(nameof(child));
        child.Parent = this;
        _children.Add(child);
    }

    public IEnumerable<AreaHierarchyNode> SelfAndDescendants()
    {
        yield return this;
        foreach (var child in _children)
        {
            foreach (var desc in child.SelfAndDescendants())
            {
                yield return desc;
            }
        }
    }

    public AreaHierarchyNode? FindDescendantByAreaId(int areaId)
    {
        if (AreaId == areaId) return this;
        foreach (var child in _children)
        {
            var found = child.FindDescendantByAreaId(areaId);
            if (found is not null) return found;
        }
        return null;
    }
}
