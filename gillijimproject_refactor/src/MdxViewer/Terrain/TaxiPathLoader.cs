using System.Numerics;
using DBCD;
using DBCD.Providers;
using MdxViewer.Logging;
using MdxViewer.Rendering;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads TaxiNodes, TaxiPath, and TaxiPathNode from DBC to visualize flight paths.
/// Renders taxi nodes as named markers and flight paths as 3D polylines.
/// </summary>
public class TaxiPathLoader
{
    public record TaxiNode(
        int Id,
        string Name,
        Vector3 Position,
        int ContinentId,
        int[] MountCreatureIds,
        int MountCreatureId,
        int MountDisplayId,
        float MountScale,
        string? MountModelPath);
    public record TaxiRoute(int PathId, int FromNodeId, int ToNodeId, int Cost, List<Vector3> Waypoints);

    public List<TaxiNode> Nodes { get; } = new();
    public List<TaxiRoute> Routes { get; } = new();

    /// <summary>
    /// Load taxi data for a specific map from DBC.
    /// </summary>
    public void Load(DBCD.DBCD dbcd, string build, int mapId)
    {
        Nodes.Clear();
        Routes.Clear();

        // 1. Load TaxiNodes
        IDBCDStorage nodeStorage;
        try
        {
            try { nodeStorage = dbcd.Load("TaxiNodes", build, Locale.EnUS); }
            catch { nodeStorage = dbcd.Load("TaxiNodes", build, Locale.None); }
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TaxiPath] Failed to load TaxiNodes.dbc: {ex.Message}");
            return;
        }

        var nodePositions = new Dictionary<int, Vector3>();
        var creatureDisplayIds = LoadCreatureDisplayIds(dbcd, build);
        var displayInfo = LoadCreatureDisplayInfo(dbcd, build);
        var creatureModelPaths = LoadCreatureModelPaths(dbcd, build);
        foreach (var key in nodeStorage.Keys)
        {
            var row = nodeStorage[key];
            int continentId = TryGetInt(row, "ContinentID") ?? 0;
            if (continentId != mapId) continue;

            string name = Sanitize(TryGetString(row, "Name_lang") ?? $"Node #{key}");
            Vector3 pos = ReadPos(row);

            int[] mountCreatureIds = ReadIntArray(row, "MountCreatureID", 2);
            int mountCreatureId = mountCreatureIds.FirstOrDefault(id => id > 0);
            int mountDisplayId = 0;
            float mountScale = 1.0f;
            string? mountModelPath = null;
            if (mountCreatureId > 0 && creatureDisplayIds.TryGetValue(mountCreatureId, out int[]? displayIds))
            {
                mountDisplayId = displayIds.FirstOrDefault(id => id > 0);
                if (mountDisplayId > 0 && displayInfo.TryGetValue(mountDisplayId, out var display))
                {
                    mountScale = display.scale > 0.01f ? display.scale : 1.0f;
                    if (display.modelId > 0)
                        creatureModelPaths.TryGetValue(display.modelId, out mountModelPath);
                }
            }

            var node = new TaxiNode(key, name, pos, continentId, mountCreatureIds, mountCreatureId, mountDisplayId, mountScale, mountModelPath);
            Nodes.Add(node);
            nodePositions[key] = pos;
        }

        ViewerLog.Trace($"[TaxiPath] Loaded {Nodes.Count} taxi nodes for mapId={mapId}");

        if (Nodes.Count == 0) return;

        // 2. Load TaxiPath (routes between nodes)
        IDBCDStorage pathStorage;
        try
        {
            try { pathStorage = dbcd.Load("TaxiPath", build, Locale.EnUS); }
            catch { pathStorage = dbcd.Load("TaxiPath", build, Locale.None); }
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TaxiPath] Failed to load TaxiPath.dbc: {ex.Message}");
            return;
        }

        var pathIds = new Dictionary<int, (int from, int to, int cost)>();
        foreach (var key in pathStorage.Keys)
        {
            var row = pathStorage[key];
            int fromNode = TryGetInt(row, "FromTaxiNode") ?? 0;
            int toNode = TryGetInt(row, "ToTaxiNode") ?? 0;
            int cost = TryGetInt(row, "Cost") ?? 0;

            // Only include paths where at least one endpoint is on our map
            if (nodePositions.ContainsKey(fromNode) || nodePositions.ContainsKey(toNode))
                pathIds[key] = (fromNode, toNode, cost);
        }

        ViewerLog.Trace($"[TaxiPath] Found {pathIds.Count} paths touching mapId={mapId}");

        // 3. Load TaxiPathNode (waypoints along each path)
        IDBCDStorage waypointStorage;
        try
        {
            try { waypointStorage = dbcd.Load("TaxiPathNode", build, Locale.EnUS); }
            catch { waypointStorage = dbcd.Load("TaxiPathNode", build, Locale.None); }
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TaxiPath] Failed to load TaxiPathNode.dbc: {ex.Message}");
            return;
        }

        // Group waypoints by PathID, sorted by NodeIndex
        var waypointsByPath = new Dictionary<int, SortedList<int, Vector3>>();
        foreach (var key in waypointStorage.Keys)
        {
            var row = waypointStorage[key];
            int pathId = TryGetInt(row, "PathID") ?? 0;
            if (!pathIds.ContainsKey(pathId)) continue;

            int nodeIndex = TryGetInt(row, "NodeIndex") ?? 0;
            int continentId = TryGetInt(row, "ContinentID") ?? 0;
            // Only include waypoints on our map
            if (continentId != mapId) continue;

            Vector3 loc = ReadLoc(row);

            if (!waypointsByPath.ContainsKey(pathId))
                waypointsByPath[pathId] = new SortedList<int, Vector3>();
            waypointsByPath[pathId][nodeIndex] = loc;
        }

        // Build routes
        foreach (var (pathId, (from, to, cost)) in pathIds)
        {
            var waypoints = new List<Vector3>();
            if (waypointsByPath.TryGetValue(pathId, out var sorted))
            {
                waypoints.AddRange(sorted.Values);
            }
            else
            {
                // No waypoints — use node positions as fallback
                if (nodePositions.TryGetValue(from, out var fromPos)) waypoints.Add(fromPos);
                if (nodePositions.TryGetValue(to, out var toPos)) waypoints.Add(toPos);
            }

            if (waypoints.Count >= 2)
                Routes.Add(new TaxiRoute(pathId, from, to, cost, waypoints));
        }

        ViewerLog.Trace($"[TaxiPath] Built {Routes.Count} routes with waypoints");

        // Diagnostic: print first few
        foreach (var n in Nodes.Take(5))
            ViewerLog.Trace($"[TaxiPath]   Node [{n.Id}] \"{n.Name}\" pos=({n.Position.X:F0},{n.Position.Y:F0},{n.Position.Z:F0}) mountCreature={n.MountCreatureId} model=\"{n.MountModelPath ?? "?"}\"");
        foreach (var r in Routes.Take(3))
            ViewerLog.Trace($"[TaxiPath]   Route [{r.PathId}] {r.FromNodeId}->{r.ToNodeId} ({r.Waypoints.Count} waypoints)");
    }

    private static Dictionary<int, int[]> LoadCreatureDisplayIds(DBCD.DBCD dbcd, string build)
    {
        var results = new Dictionary<int, int[]>();

        IDBCDStorage storage;
        try
        {
            storage = LoadDbc(dbcd, "Creature", build);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TaxiPath] Failed to load Creature.dbc: {ex.Message}");
            return results;
        }

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int[] displayIds = ReadIntArray(row, "DisplayID", 4);
            if (displayIds.Any(id => id > 0))
                results[key] = displayIds;
        }

        return results;
    }

    private static Dictionary<int, (int modelId, float scale)> LoadCreatureDisplayInfo(DBCD.DBCD dbcd, string build)
    {
        var results = new Dictionary<int, (int modelId, float scale)>();

        IDBCDStorage storage;
        try
        {
            storage = LoadDbc(dbcd, "CreatureDisplayInfo", build);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TaxiPath] Failed to load CreatureDisplayInfo.dbc: {ex.Message}");
            return results;
        }

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int modelId = TryGetInt(row, "ModelID") ?? TryGetInt(row, "ModelId") ?? 0;
            float scale = TryGetFloat(row, "CreatureModelScale") ?? 1.0f;
            if (modelId > 0)
                results[key] = (modelId, scale);
        }

        return results;
    }

    private static Dictionary<int, string> LoadCreatureModelPaths(DBCD.DBCD dbcd, string build)
    {
        var results = new Dictionary<int, string>();

        IDBCDStorage storage;
        try
        {
            storage = LoadDbc(dbcd, "CreatureModelData", build);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TaxiPath] Failed to load CreatureModelData.dbc: {ex.Message}");
            return results;
        }

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            string? modelName = TryGetString(row, "ModelName") ?? TryGetString(row, "ModelPath");
            if (!string.IsNullOrWhiteSpace(modelName))
                results[key] = modelName.Replace('/', '\\');
        }

        return results;
    }

    private static IDBCDStorage LoadDbc(DBCD.DBCD dbcd, string tableName, string build)
    {
        try { return dbcd.Load(tableName, build, Locale.EnUS); }
        catch { return dbcd.Load(tableName, build, Locale.None); }
    }

    private static Vector3 ReadPos(dynamic row)
    {
        try
        {
            var posArr = row["Pos"];
            if (posArr is float[] fa && fa.Length >= 3)
                return new Vector3(fa[0], fa[1], fa[2]);
            if (posArr is object[] oa && oa.Length >= 3)
                return new Vector3(Convert.ToSingle(oa[0]), Convert.ToSingle(oa[1]), Convert.ToSingle(oa[2]));
        }
        catch { }
        return Vector3.Zero;
    }

    private static Vector3 ReadLoc(dynamic row)
    {
        try
        {
            var locArr = row["Loc"];
            if (locArr is float[] fa && fa.Length >= 3)
                return new Vector3(fa[0], fa[1], fa[2]);
            if (locArr is object[] oa && oa.Length >= 3)
                return new Vector3(Convert.ToSingle(oa[0]), Convert.ToSingle(oa[1]), Convert.ToSingle(oa[2]));
        }
        catch { }
        return Vector3.Zero;
    }

    private static string Sanitize(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        int nullIdx = s.IndexOf('\0');
        if (nullIdx >= 0) s = s[..nullIdx];
        return new string(s.Where(c => !char.IsControl(c) || c == '\n').ToArray());
    }

    private static string? TryGetString(dynamic row, string fieldName)
    {
        try
        {
            var val = row[fieldName];
            if (val is string s) return s;
            return val?.ToString();
        }
        catch { return null; }
    }

    private static int? TryGetInt(dynamic row, string fieldName)
    {
        try
        {
            var val = row[fieldName];
            if (val is int i) return i;
            if (val is uint u) return (int)u;
            if (val is short s) return s;
            if (val is ushort us) return us;
            if (int.TryParse(val?.ToString(), out int parsed)) return parsed;
            return null;
        }
        catch { return null; }
    }

    private static float? TryGetFloat(dynamic row, string fieldName)
    {
        try
        {
            var val = row[fieldName];
            if (val is float f) return f;
            if (val is double d) return (float)d;
            if (float.TryParse(val?.ToString(), out float parsed)) return parsed;
            return null;
        }
        catch { return null; }
    }

    private static int[] ReadIntArray(dynamic row, string fieldName, int expectedLength)
    {
        try
        {
            var value = row[fieldName];
            if (value is int[] intArray)
                return intArray;
            if (value is uint[] uintArray)
                return uintArray.Select(static entry => (int)entry).ToArray();
            if (value is object[] objectArray)
                return objectArray.Select(ConvertToInt).ToArray();
        }
        catch { }

        var fallback = new int[expectedLength];
        for (int i = 0; i < expectedLength; i++)
            fallback[i] = TryGetInt(row, $"{fieldName}[{i}]") ?? TryGetInt(row, $"{fieldName}_{i}") ?? 0;
        return fallback;
    }

    private static int ConvertToInt(object? value)
    {
        if (value is null)
            return 0;
        if (value is int intValue)
            return intValue;
        if (value is uint uintValue)
            return (int)uintValue;
        if (value is short shortValue)
            return shortValue;
        if (value is ushort ushortValue)
            return ushortValue;
        return int.TryParse(value.ToString(), out int parsed) ? parsed : 0;
    }
}
