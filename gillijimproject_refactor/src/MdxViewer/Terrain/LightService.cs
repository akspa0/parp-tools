using System.Numerics;
using DBCD;
using DBCD.Providers;
using MdxViewer.Logging;
using MdxViewer.Rendering;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads Light.dbc and LightData.dbc to provide zone-based ambient and directional lighting.
/// Light.dbc defines light zones with position + falloff radius on each map.
/// LightData.dbc provides time-of-day keyframed colors (DirectColor, AmbientColor, FogColor, etc.).
/// For Alpha 0.5.3, we use a fixed time-of-day (noon) since there's no day/night cycle.
/// </summary>
public class LightService
{
    private readonly List<LightZone> _zones = new();
    private readonly Dictionary<int, List<LightDataEntry>> _lightData = new(); // LightParamID → entries sorted by Time

    // Current interpolated light state
    public Vector3 AmbientColor { get; private set; } = new(0.5f, 0.5f, 0.5f);
    public Vector3 DirectColor { get; private set; } = new(1.0f, 0.95f, 0.85f);
    public Vector3 SkyTopColor { get; private set; } = new(0.4f, 0.6f, 0.9f);
    public Vector3 FogColor { get; private set; } = new(0.6f, 0.7f, 0.85f);
    public float FogEnd { get; private set; } = 1500f;
    public float FogScaler { get; private set; } = 1.0f;
    public int ActiveLightId { get; private set; } = -1;

    // Fixed time of day (0-2880, where 1440 = noon in WoW's 24-minute cycle)
    // 1440 = noon, 0 = midnight
    public int TimeOfDay { get; set; } = 1440;

    public int ZoneCount => _zones.Count;
    public int DataEntryCount => _lightData.Values.Sum(v => v.Count);

    /// <summary>
    /// Load Light.dbc and LightData.dbc from the given DBC provider.
    /// </summary>
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        LoadLightZones(dbcd, build, mapId);
        LoadLightData(dbcd, build);

        ViewerLog.Trace($"[LightService] Loaded {_zones.Count} light zones for map {mapId}, {DataEntryCount} data entries");
    }

    private void LoadLightZones(DBCD.DBCD dbcd, string build, int mapId)
    {
        IDBCDStorage storage;
        try
        {
            try { storage = dbcd.Load("Light", build, Locale.EnUS); }
            catch { storage = dbcd.Load("Light", build, Locale.None); }
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[LightService] Failed to load Light.dbc: {ex.Message}");
            return;
        }

        // Detect columns
        string continentCol = DetectColumn(storage, "ContinentID", "MapID");
        string coordsCol = DetectColumn(storage, "GameCoords");
        string falloffStartCol = DetectColumn(storage, "GameFalloffStart");
        string falloffEndCol = DetectColumn(storage, "GameFalloffEnd");
        string paramsCol = DetectColumn(storage, "LightParamsID");

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int continent = SafeField<int>(row, continentCol, -1);
            if (continent != mapId) continue;

            // GameCoords is float[3] — X, Y, Z in WoW world coords
            float[] coords;
            try
            {
                var rawCoords = row[coordsCol];
                if (rawCoords is float[] fa) coords = fa;
                else if (rawCoords is object[] oa) coords = oa.Select(o => Convert.ToSingle(o)).ToArray();
                else continue;
            }
            catch { continue; }

            if (coords.Length < 3) continue;

            float falloffStart = SafeField<float>(row, falloffStartCol, 0f);
            float falloffEnd = SafeField<float>(row, falloffEndCol, 0f);

            // LightParamsID is int[5] or int[8] depending on version
            int[] paramIds;
            try
            {
                var rawParams = row[paramsCol];
                if (rawParams is int[] ia) paramIds = ia;
                else if (rawParams is uint[] ua) paramIds = ua.Select(u => (int)u).ToArray();
                else if (rawParams is ushort[] usa) paramIds = usa.Select(u => (int)u).ToArray();
                else if (rawParams is object[] oa) paramIds = oa.Select(o => Convert.ToInt32(o)).ToArray();
                else continue;
            }
            catch { continue; }

            // Convert WoW coords to renderer coords
            // rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX
            float rendererX = WoWConstants.MapOrigin - coords[1];
            float rendererY = WoWConstants.MapOrigin - coords[0];
            float rendererZ = coords[2];

            _zones.Add(new LightZone
            {
                Id = (int)key,
                Position = new Vector3(rendererX, rendererY, rendererZ),
                FalloffStart = falloffStart,
                FalloffEnd = falloffEnd,
                ParamIds = paramIds
            });
        }

        // Sort by falloff end (smallest first) so specific zones override global ones
        _zones.Sort((a, b) => a.FalloffEnd.CompareTo(b.FalloffEnd));
    }

    private void LoadLightData(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage storage;
        try
        {
            try { storage = dbcd.Load("LightData", build, Locale.EnUS); }
            catch { storage = dbcd.Load("LightData", build, Locale.None); }
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[LightService] Failed to load LightData.dbc: {ex.Message}");
            return;
        }

        string paramCol = DetectColumn(storage, "LightParamID", "LightParamsID");
        string timeCol = DetectColumn(storage, "Time");
        string directCol = DetectColumn(storage, "DirectColor");
        string ambientCol = DetectColumn(storage, "AmbientColor");
        string skyTopCol = DetectColumn(storage, "SkyTopColor");
        string fogCol = DetectColumn(storage, "SkyFogColor");
        string fogEndCol = DetectColumn(storage, "FogEnd");
        string fogScalerCol = DetectColumn(storage, "FogScaler");

        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int paramId = SafeField<int>(row, paramCol, 0);
            if (paramId == 0) continue;

            int time = SafeField<int>(row, timeCol, 0);
            int directColor = SafeField<int>(row, directCol, 0);
            int ambientColor = SafeField<int>(row, ambientCol, 0);
            int skyTopColor = SafeField<int>(row, skyTopCol, 0);
            int fogColor = SafeField<int>(row, fogCol, 0);
            float fogEnd = SafeField<float>(row, fogEndCol, 0f);
            float fogScaler = SafeField<float>(row, fogScalerCol, 0f);

            if (!_lightData.ContainsKey(paramId))
                _lightData[paramId] = new List<LightDataEntry>();

            _lightData[paramId].Add(new LightDataEntry
            {
                Time = time,
                DirectColor = UnpackColor(directColor),
                AmbientColor = UnpackColor(ambientColor),
                SkyTopColor = UnpackColor(skyTopColor),
                FogColor = UnpackColor(fogColor),
                FogEnd = fogEnd,
                FogScaler = fogScaler
            });
        }

        // Sort each param's entries by time
        foreach (var list in _lightData.Values)
            list.Sort((a, b) => a.Time.CompareTo(b.Time));
    }

    /// <summary>
    /// Update lighting based on camera position. Finds the nearest light zone
    /// and interpolates colors from LightData at the current time of day.
    /// </summary>
    public void Update(Vector3 cameraPos)
    {
        if (_zones.Count == 0) return;

        // Find the best matching light zone:
        // 1. Zones with FalloffEnd=0 are global defaults (apply everywhere)
        // 2. Zones with position+falloff define local overrides
        // 3. Camera inside falloff range → blend based on distance
        LightZone? bestZone = null;
        float bestWeight = 0f;

        // First, find global default (FalloffEnd == 0 or very large)
        LightZone? globalZone = null;
        foreach (var zone in _zones)
        {
            if (zone.FalloffEnd <= 0.01f || zone.FalloffEnd > 50000f)
            {
                globalZone = zone;
                break;
            }
        }

        // Then find the nearest local zone the camera is inside
        foreach (var zone in _zones)
        {
            if (zone.FalloffEnd <= 0.01f || zone.FalloffEnd > 50000f) continue;
            float dist = Vector3.Distance(cameraPos, zone.Position);
            if (dist > zone.FalloffEnd) continue;

            float weight;
            if (dist <= zone.FalloffStart)
                weight = 1.0f; // Fully inside
            else
                weight = 1.0f - (dist - zone.FalloffStart) / (zone.FalloffEnd - zone.FalloffStart);

            if (weight > bestWeight)
            {
                bestWeight = weight;
                bestZone = zone;
            }
        }

        // Use best local zone, or fall back to global
        var activeZone = bestZone ?? globalZone;
        if (activeZone == null) return;

        ActiveLightId = activeZone.Id;

        // Get the normal-day param set (index 0)
        int paramId = activeZone.ParamIds.Length > 0 ? activeZone.ParamIds[0] : 0;
        if (paramId == 0) return;

        // Look up LightData for this param at current time
        if (!_lightData.TryGetValue(paramId, out var entries) || entries.Count == 0)
            return;

        // Interpolate between time keyframes
        var data = InterpolateTime(entries, TimeOfDay);

        // If we have a local zone with partial weight, blend with global
        if (bestZone != null && globalZone != null && bestWeight < 1.0f)
        {
            int globalParamId = globalZone.ParamIds.Length > 0 ? globalZone.ParamIds[0] : 0;
            if (globalParamId != 0 && _lightData.TryGetValue(globalParamId, out var globalEntries) && globalEntries.Count > 0)
            {
                var globalData = InterpolateTime(globalEntries, TimeOfDay);
                data = BlendData(globalData, data, bestWeight);
            }
        }

        AmbientColor = data.AmbientColor;
        DirectColor = data.DirectColor;
        SkyTopColor = data.SkyTopColor;
        FogColor = data.FogColor;
        if (data.FogEnd > 10f) FogEnd = data.FogEnd;
        FogScaler = data.FogScaler;
    }

    /// <summary>
    /// Interpolate between LightData keyframes at the given time.
    /// </summary>
    private static LightDataEntry InterpolateTime(List<LightDataEntry> entries, int time)
    {
        if (entries.Count == 1) return entries[0];

        // Find the two keyframes surrounding the current time
        int idx = 0;
        for (int i = 0; i < entries.Count; i++)
        {
            if (entries[i].Time > time) break;
            idx = i;
        }

        int nextIdx = (idx + 1) % entries.Count;
        var a = entries[idx];
        var b = entries[nextIdx];

        if (a.Time == b.Time) return a;

        // Handle wrap-around (midnight crossing)
        int range = b.Time > a.Time ? b.Time - a.Time : (2880 - a.Time) + b.Time;
        int elapsed = time >= a.Time ? time - a.Time : (2880 - a.Time) + time;
        float t = range > 0 ? (float)elapsed / range : 0f;
        t = Math.Clamp(t, 0f, 1f);

        return new LightDataEntry
        {
            Time = time,
            DirectColor = Vector3.Lerp(a.DirectColor, b.DirectColor, t),
            AmbientColor = Vector3.Lerp(a.AmbientColor, b.AmbientColor, t),
            SkyTopColor = Vector3.Lerp(a.SkyTopColor, b.SkyTopColor, t),
            FogColor = Vector3.Lerp(a.FogColor, b.FogColor, t),
            FogEnd = a.FogEnd + (b.FogEnd - a.FogEnd) * t,
            FogScaler = a.FogScaler + (b.FogScaler - a.FogScaler) * t
        };
    }

    private static LightDataEntry BlendData(LightDataEntry global, LightDataEntry local, float localWeight)
    {
        float gw = 1.0f - localWeight;
        return new LightDataEntry
        {
            Time = local.Time,
            DirectColor = global.DirectColor * gw + local.DirectColor * localWeight,
            AmbientColor = global.AmbientColor * gw + local.AmbientColor * localWeight,
            SkyTopColor = global.SkyTopColor * gw + local.SkyTopColor * localWeight,
            FogColor = global.FogColor * gw + local.FogColor * localWeight,
            FogEnd = global.FogEnd * gw + local.FogEnd * localWeight,
            FogScaler = global.FogScaler * gw + local.FogScaler * localWeight
        };
    }

    /// <summary>
    /// Unpack a DBC color int (BGRA or RGBA packed) to normalized RGB Vector3.
    /// WoW DBC colors are stored as 0xAARRGGBB.
    /// </summary>
    private static Vector3 UnpackColor(int packed)
    {
        float r = ((packed >> 16) & 0xFF) / 255f;
        float g = ((packed >> 8) & 0xFF) / 255f;
        float b = (packed & 0xFF) / 255f;
        return new Vector3(r, g, b);
    }

    private static string DetectColumn(IDBCDStorage storage, params string[] candidates)
    {
        if (storage.Values.Count == 0) return candidates[0];
        var row = storage.Values.First();
        foreach (var col in candidates)
        {
            try { _ = row[col]; return col; }
            catch { }
        }
        return candidates[0];
    }

    private static T SafeField<T>(dynamic row, string col, T fallback)
    {
        try { return (T)row[col]; }
        catch { return fallback; }
    }

    private class LightZone
    {
        public int Id;
        public Vector3 Position;
        public float FalloffStart;
        public float FalloffEnd;
        public int[] ParamIds = Array.Empty<int>();
    }

    private class LightDataEntry
    {
        public int Time;
        public Vector3 DirectColor;
        public Vector3 AmbientColor;
        public Vector3 SkyTopColor;
        public Vector3 FogColor;
        public float FogEnd;
        public float FogScaler;
    }
}
