using System.Numerics;
using DBCD;
using DBCD.Providers;
using WowViewer.Core.IO.Lighting;
using WoWViewer.Logging;
using WoWViewer.Rendering;

namespace WoWViewer.Terrain;

/// <summary>
/// Experimental loader for zone-based ambient and directional lighting.
/// Light.dbc defines light zones with position + falloff radius on each map.
/// The current flattened LightData path is only valid when that exact build/table schema exists;
/// classic LightParams/LightIntBand/LightFloatBand tables are a separate DBCD-backed contract.
/// DBC rows are database records, not animation keyframes.
/// </summary>
public class LightService
{
    private readonly List<LightZone> _zones = new();
    private readonly Dictionary<int, List<LightDataEntry>> _lightData = new(); // LightParamID → entries sorted by Time
    private LightDbcCatalog? _classicCatalog;
    private int _mapId = -1;

    // Current interpolated light state
    public Vector3 AmbientColor { get; private set; } = new(0.5f, 0.5f, 0.5f);
    public Vector3 DirectColor { get; private set; } = new(1.0f, 0.95f, 0.85f);
    public Vector3 SkyTopColor { get; private set; } = new(0.4f, 0.6f, 0.9f);
    public Vector3 FogColor { get; private set; } = new(0.6f, 0.7f, 0.85f);
    public float FogEnd { get; private set; } = 1500f;
    public float FogScaler { get; private set; } = 1.0f;
    /// <summary>The selected in-range local record. Global DBC rows never replace the viewer sun.</summary>
    public int ActiveLightId { get; private set; } = -1;
    public bool HasActiveLocalOverlay { get; private set; }
    public float ActiveLocalWeight { get; private set; }

    // Fixed time of day (0-2880, where 1440 = noon in WoW's 24-minute cycle)
    // 1440 = noon, 0 = midnight
    public int TimeOfDay { get; set; } = 1440;

    public int ZoneCount => _classicCatalog is null
        ? _zones.Count
        : _classicCatalog.Zones.Count(zone => zone.ContinentId == _mapId);
    public int DataEntryCount => _classicCatalog?.TimedSampleCount ?? _lightData.Values.Sum(v => v.Count);
    public LightDbcEvaluationEvidence? LastDbcEvidence { get; private set; }
    public string Source { get; private set; } = "not loaded";
    public string Status { get; private set; } = "Lighting database not loaded.";
    public int BandCountRecoveryCount => _classicCatalog?.BandCountRecoveries.Length ?? 0;
    public int MissingOptionalSkyboxCount => _classicCatalog?.MissingSkyboxReferences.Length ?? 0;

    /// <summary>
    /// Load the exact-build lighting database chain. Classic builds use the native
    /// Light/LightParams/LightIntBand/LightFloatBand/LightSkybox contract first. The
    /// flattened LightData path is retained only as a later-build compatibility fallback.
    /// </summary>
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _zones.Clear();
        _lightData.Clear();
        _classicCatalog = null;
        LastDbcEvidence = null;
        ResetActiveLocalOverlayState();
        _mapId = mapId;
        Source = "loading";
        Status = $"Loading exact-build outdoor lighting for map {mapId}...";

        try
        {
            _classicCatalog = new BuildScopedLightDbcProfileResolver().Load(
                dbcProvider,
                dbdDir,
                build);
            ViewerLog.Trace(
                $"[LightService] Loaded exact-build Light* chain for map {mapId}: " +
                $"{ZoneCount} zones, {DataEntryCount} timed samples");
            Source = "Light/LightParams/LightIntBand/LightFloatBand";
            Status = $"Loaded exact-build Light* DBC chain ({ZoneCount} map zones, " +
                $"{BandCountRecoveryCount} explicit band-count recoveries, " +
                $"{MissingOptionalSkyboxCount} missing optional skybox references).";
            return;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace(
                $"[LightService] Classic Light* chain unavailable for exact build {build}; " +
                $"trying flattened LightData compatibility path: {ex.Message}");
        }

        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        LoadLightZones(dbcd, build, mapId);
        LoadLightData(dbcd, build);

        ViewerLog.Trace($"[LightService] Loaded {_zones.Count} light zones for map {mapId}, {DataEntryCount} data entries");
        Source = _zones.Count > 0 && DataEntryCount > 0 ? "LightData compatibility" : "unavailable";
        Status = _zones.Count > 0 && DataEntryCount > 0
            ? $"Loaded flattened LightData compatibility path ({_zones.Count} zones, {DataEntryCount} samples)."
            : $"No usable exact-build Light* or LightData profile was loaded for map {mapId}.";
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

            // GameCoords is float[3] in fixed-scale X,Z,Y order.
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

            // Convert X,Z,Y fixed-scale values to renderer coordinates.
            float wowX = coords[0] / LightDbcZoneRecord.GameUnitsPerWorldUnit;
            float wowZ = coords[1] / LightDbcZoneRecord.GameUnitsPerWorldUnit;
            float wowY = coords[2] / LightDbcZoneRecord.GameUnitsPerWorldUnit;
            float rendererX = WoWConstants.MapOrigin - wowY;
            float rendererY = WoWConstants.MapOrigin - wowX;
            float rendererZ = wowZ;

            _zones.Add(new LightZone
            {
                Id = (int)key,
                Position = new Vector3(rendererX, rendererY, rendererZ),
                FalloffStart = falloffStart / LightDbcZoneRecord.GameUnitsPerWorldUnit,
                FalloffEnd = falloffEnd / LightDbcZoneRecord.GameUnitsPerWorldUnit,
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
        ResetActiveLocalOverlayState();
        LastDbcEvidence = null;

        if (_classicCatalog is not null)
        {
            UpdateClassicCatalog(cameraPos);
            return;
        }

        if (_zones.Count == 0)
        {
            Status = $"Global viewer light active; no local LightData compatibility overlay exists for map {_mapId}.";
            return;
        }

        // Find the best matching light zone:
        // zones with position+falloff define local overlays. Global DBC rows are metadata
        // for the native table chain; they do not replace the viewer's always-present sun.
        LightZone? bestZone = null;
        float bestWeight = 0f;

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

        if (bestZone == null)
        {
            Status = $"Global viewer light active; no in-range local LightData compatibility overlay at the camera.";
            return;
        }

        // Get the normal-day param set (index 0)
        int paramId = bestZone.ParamIds.Length > 0 ? bestZone.ParamIds[0] : 0;
        if (paramId == 0)
        {
            Status = $"Global viewer light active; local LightData record {bestZone.Id} has no clear-weather profile.";
            return;
        }

        // Look up LightData for this param at current time
        if (!_lightData.TryGetValue(paramId, out var entries) || entries.Count == 0)
        {
            Status = $"Global viewer light active; local LightData record {bestZone.Id} has no timed samples.";
            return;
        }

        // Interpolate between time-sampled LightData records.
        var data = InterpolateTime(entries, TimeOfDay);

        AmbientColor = data.AmbientColor;
        DirectColor = data.DirectColor;
        SkyTopColor = data.SkyTopColor;
        FogColor = data.FogColor;
        if (data.FogEnd > 10f) FogEnd = data.FogEnd;
        FogScaler = data.FogScaler;
        ActiveLightId = bestZone.Id;
        HasActiveLocalOverlay = true;
        ActiveLocalWeight = Math.Clamp(bestWeight, 0f, 1f);
        Status = $"Local LightData compatibility overlay {ActiveLightId} active at weight {ActiveLocalWeight:F3}.";
    }

    private void UpdateClassicCatalog(Vector3 rendererCameraPosition)
    {
        // Inverse of LightDbcZoneRecord.ToRendererPosition(MapOrigin).
        var worldPosition = new Vector3(
            WoWConstants.MapOrigin - rendererCameraPosition.Y,
            WoWConstants.MapOrigin - rendererCameraPosition.X,
            rendererCameraPosition.Z);

        try
        {
            LightDbcEvaluation value = _classicCatalog!.EvaluateClearWeather(
                _mapId,
                worldPosition,
                TimeOfDay);
            LastDbcEvidence = value.Evidence;
            if (!value.HasLocalProfile || value.Evidence.LocalWeight <= 0f)
            {
                Status = $"Global viewer light active; no in-range local DBC overlay at time " +
                    $"{value.Evidence.NormalizedTime}/2880.";
                return;
            }

            DirectColor = value.GetLocalColor(LightDbcColorBand.Direct);
            AmbientColor = value.GetLocalColor(LightDbcColorBand.Ambient);
            SkyTopColor = value.GetLocalColor(LightDbcColorBand.SkyTop);
            FogColor = value.GetLocalColor(LightDbcColorBand.Fog);

            float fogEnd = value.GetLocalFloat(LightDbcFloatBand.FogEnd);
            if (float.IsFinite(fogEnd) && fogEnd > 10f)
                FogEnd = fogEnd;
            float fogScaler = value.GetLocalFloat(LightDbcFloatBand.FogStartScalar);
            if (float.IsFinite(fogScaler))
                FogScaler = fogScaler;

            ActiveLightId = value.Evidence.LocalProfile!.LightRecordId;
            HasActiveLocalOverlay = true;
            ActiveLocalWeight = Math.Clamp(value.Evidence.LocalWeight, 0f, 1f);
            Status = $"Local exact-build DBC overlay {ActiveLightId} active at weight {ActiveLocalWeight:F3}, " +
                $"time {value.Evidence.NormalizedTime}/2880 " +
                $"({BandCountRecoveryCount} recorded band-count recoveries).";
        }
        catch (Exception ex)
        {
            ResetActiveLocalOverlayState();
            LastDbcEvidence = null;
            Status = $"Global viewer light active; exact-build local Light* evaluation failed: {ex.Message}";
            ViewerLog.Trace(
                $"[LightService] Exact-build Light* evaluation failed for map {_mapId}, " +
                $"time {TimeOfDay}, world position {worldPosition}: {ex.Message}");
        }
    }

    private void ResetActiveLocalOverlayState()
    {
        ActiveLightId = -1;
        HasActiveLocalOverlay = false;
        ActiveLocalWeight = 0f;

        // Do not let a departed local profile leak its colors into diagnostics
        // or a future opt-in overlay evaluation. The renderer's global viewer
        // light remains the authoritative identity case when no local profile
        // is active.
        DirectColor = new Vector3(1.0f, 0.95f, 0.85f);
        AmbientColor = new Vector3(0.5f, 0.5f, 0.5f);
        SkyTopColor = new Vector3(0.4f, 0.6f, 0.9f);
        FogColor = new Vector3(0.6f, 0.7f, 0.85f);
        FogEnd = 1500f;
        FogScaler = 1.0f;
    }

    /// <summary>
    /// Interpolate between time-sampled LightData records at the given time.
    /// </summary>
    private static LightDataEntry InterpolateTime(List<LightDataEntry> entries, int time)
    {
        if (entries.Count == 1) return entries[0];

        // Find the two records surrounding the current time.
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
