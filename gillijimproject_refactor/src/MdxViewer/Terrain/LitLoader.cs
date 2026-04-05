using System.Numerics;
using System.Text;
using MdxViewer.DataSources;
using MdxViewer.Logging;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads Alpha-era World\<map>\lights.lit files and exposes enough structure for
/// inspection, overlay visualization, and experimental runtime fog/light sampling.
/// </summary>
public sealed class LitLoader
{
    public const uint Version02Test = 0x00000002;
    public const uint Version83 = 0x80000003;
    public const uint Version84 = 0x80000004;
    public const uint Version85 = 0x80000005;

    public const int TrackDirectColor = 0;
    public const int TrackAmbientColor = 1;
    public const int TrackSkyTop = 2;
    public const int TrackSkyMiddle = 3;
    public const int TrackSkyMiddleToHorizon = 4;
    public const int TrackSkyAboveHorizon = 5;
    public const int TrackSkyHorizon = 6;
    public const int TrackFogColor = 7;
    public const int TrackShadowOpacity = 8;
    public const int TrackSunColor = 9;
    public const int TrackSunHaloColor = 10;
    public const int TrackCloudColor = 12;
    public const int TrackGroundShadowColor = 15;
    public const int TrackWaterColorLight = 16;
    public const int TrackWaterColorDark = 17;

    private const float RadiusScale = 36f;
    private readonly IDataSource _dataSource;
    private readonly string _mapName;
    private readonly string? _preferredSourcePath;
    private readonly List<string> _availableSourcePaths = new();

    public sealed record LitColorKeyframe(int TimeOfDay, uint PackedColor, Vector3 Color);

    public sealed class LitTrack
    {
        public LitTrack(int trackIndex, IReadOnlyList<LitColorKeyframe> keyframes)
        {
            TrackIndex = trackIndex;
            Keyframes = keyframes;
        }

        public int TrackIndex { get; }

        public IReadOnlyList<LitColorKeyframe> Keyframes { get; }

        public Vector3 Evaluate(float timeOfDay)
        {
            if (Keyframes.Count == 0)
                return Vector3.Zero;

            float wrappedTime = WrapTime(timeOfDay);
            if (Keyframes.Count == 1)
                return Keyframes[0].Color;

            LitColorKeyframe previous = Keyframes[^1];
            float previousTime = previous.TimeOfDay;
            if (previousTime > wrappedTime)
                previousTime -= 2880f;

            for (int i = 0; i < Keyframes.Count; i++)
            {
                LitColorKeyframe current = Keyframes[i];
                float currentTime = current.TimeOfDay;
                if (i == 0 && currentTime < previousTime)
                    currentTime += 2880f;

                if (wrappedTime <= currentTime)
                {
                    float span = currentTime - previousTime;
                    if (span <= 0.0001f)
                        return current.Color;

                    float t = (wrappedTime - previousTime) / span;
                    return Vector3.Lerp(previous.Color, current.Color, Math.Clamp(t, 0f, 1f));
                }

                previous = current;
                previousTime = currentTime;
            }

            LitColorKeyframe first = Keyframes[0];
            float wrapSpan = (first.TimeOfDay + 2880f) - previousTime;
            if (wrapSpan <= 0.0001f)
                return first.Color;

            float wrapT = (wrappedTime + 2880f - previousTime) / wrapSpan;
            return Vector3.Lerp(previous.Color, first.Color, Math.Clamp(wrapT, 0f, 1f));
        }
    }

    public sealed class LitGroup
    {
        public LitGroup(
            int groupIndex,
            IReadOnlyList<LitTrack> tracks,
            IReadOnlyList<float> fogEndSamples,
            IReadOnlyList<float> fogStartScalers,
            int highlightSky,
            IReadOnlyList<IReadOnlyList<float>> skyFloatBands,
            int cloudMask,
            IReadOnlyList<IReadOnlyList<float>> parameterBands)
        {
            GroupIndex = groupIndex;
            Tracks = tracks;
            FogEndSamples = fogEndSamples;
            FogStartScalers = fogStartScalers;
            HighlightSky = highlightSky;
            SkyFloatBands = skyFloatBands;
            CloudMask = cloudMask;
            ParameterBands = parameterBands;
        }

        public int GroupIndex { get; }

        public IReadOnlyList<LitTrack> Tracks { get; }

        public IReadOnlyList<float> FogEndSamples { get; }

        public IReadOnlyList<float> FogStartScalers { get; }

        public int HighlightSky { get; }

        public IReadOnlyList<IReadOnlyList<float>> SkyFloatBands { get; }

        public int CloudMask { get; }

        public IReadOnlyList<IReadOnlyList<float>> ParameterBands { get; }

        public int SkyValue => HighlightSky;

        public int CloudValue => CloudMask;

        public bool TryEvaluateTrack(int trackIndex, float timeOfDay, out Vector3 color)
        {
            for (int i = 0; i < Tracks.Count; i++)
            {
                if (Tracks[i].TrackIndex != trackIndex)
                    continue;

                color = Tracks[i].Evaluate(timeOfDay);
                return true;
            }

            color = Vector3.Zero;
            return false;
        }

        public float EvaluateFogEnd(float timeOfDay)
        {
            return EvaluateFloatCurve(FogEndSamples, timeOfDay, fallback: 1500f);
        }

        public float EvaluateFogStartScaler(float timeOfDay)
        {
            return EvaluateFloatCurve(FogStartScalers, timeOfDay, fallback: 0.25f);
        }
    }

    public sealed class LitLight
    {
        public LitLight(
            int index,
            int chunkX,
            int chunkY,
            int chunkRadius,
            Vector3 position,
            float radiusRaw,
            float dropoffRaw,
            string name,
            IReadOnlyList<LitGroup> groups)
        {
            Index = index;
            ChunkX = chunkX;
            ChunkY = chunkY;
            ChunkRadius = chunkRadius;
            Position = position;
            RadiusRaw = radiusRaw;
            DropoffRaw = dropoffRaw;
            Radius = NormalizeDistance(radiusRaw);
            Dropoff = NormalizeDistance(dropoffRaw);
            Name = string.IsNullOrWhiteSpace(name) ? $"Light {index}" : name;
            Groups = groups;
        }

        public int Index { get; }

        public int ChunkX { get; }

        public int ChunkY { get; }

        public int ChunkRadius { get; }

        public Vector3 Position { get; }

        public float RadiusRaw { get; }

        public float DropoffRaw { get; }

        public float Radius { get; }

        public float Dropoff { get; }

        public string Name { get; }

        public IReadOnlyList<LitGroup> Groups { get; }

        public bool IsDefaultLight => ChunkX == -1 && ChunkY == -1 && ChunkRadius == -1;

        public bool HasMeaningfulPosition => !float.IsNaN(Position.X) && !float.IsNaN(Position.Y) && !float.IsNaN(Position.Z);

        public string DisplayName => IsDefaultLight ? $"{Name} (default)" : Name;
    }

    public sealed record LitLightingSample(
        int DominantLightIndex,
        string DominantLightName,
        float DominantWeight,
        float TimeOfDay,
        Vector3 DirectColor,
        Vector3 AmbientColor,
        Vector3 FogColor,
        Vector3 SkyTopColor,
        Vector3 SkyHorizonColor,
        float FogEnd,
        float FogStart,
        float FogStartScalar);

    public List<LitLight> Lights { get; } = new();

    public uint Version { get; private set; }

    public int RawLightCount { get; private set; }

    public string? SourcePath { get; private set; }

    public IReadOnlyList<string> AvailableSourcePaths => _availableSourcePaths;

    public string Status { get; private set; } = "LIT not loaded.";

    public bool HasData => Lights.Count > 0;

    public LitLoader(IDataSource dataSource, string mapName, string? preferredSourcePath = null)
    {
        _dataSource = dataSource;
        _mapName = mapName;
        _preferredSourcePath = string.IsNullOrWhiteSpace(preferredSourcePath) ? null : preferredSourcePath;
    }

    public bool Load()
    {
        Lights.Clear();
        _availableSourcePaths.Clear();
        _availableSourcePaths.AddRange(ResolveAvailableSourcePaths());
        SourcePath = ResolveSourcePath();
        if (SourcePath == null)
        {
            Status = $"LIT: no lit variants found for map '{_mapName}' (lights.lit, areatest.lit, light.lit).";
            return false;
        }

        byte[]? data = _dataSource.ReadFile(SourcePath);
        if (data == null || data.Length < 8)
        {
            Status = $"LIT: failed to read {SourcePath}.";
            return false;
        }

        try
        {
            using var stream = new MemoryStream(data, writable: false);
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);

            Version = reader.ReadUInt32();
            RawLightCount = reader.ReadInt32();
            int dataLightCount = RawLightCount < 0 ? Math.Abs(RawLightCount) : RawLightCount;
            int trackCount = GetTrackCountForVersion(Version);
            bool partialDataOnly = RawLightCount < 0;

            if (partialDataOnly)
            {
                var groups = new List<LitGroup>
                {
                    Version == Version02Test
                        ? ReadLegacyVersion02PartialGroup(reader, 0)
                        : ReadGroup(reader, 0, trackCount)
                };
                Lights.Add(new LitLight(0, -1, -1, -1, Vector3.Zero, 0f, 0f, "Default", groups));
            }
            else
            {
                for (int lightIndex = 0; lightIndex < dataLightCount; lightIndex++)
                {
                    var meta = ReadLightHeader(reader, lightIndex);

                    var groups = new List<LitGroup>(4);
                    for (int groupIndex = 0; groupIndex < 4; groupIndex++)
                        groups.Add(ReadGroup(reader, groupIndex, trackCount));

                    Lights.Add(new LitLight(
                        lightIndex,
                        meta.Item1,
                        meta.Item2,
                        meta.Item3,
                        meta.Item4,
                        meta.Item5,
                        meta.Item6,
                        meta.Item7,
                        groups));
                }
            }

            Status = $"LIT: loaded {Lights.Count} light entries from {SourcePath}.";
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[LIT] Loaded {Lights.Count} light entries from {SourcePath} (version=0x{Version:X8}, rawCount={RawLightCount}).");
            return Lights.Count > 0;
        }
        catch (Exception ex)
        {
            Lights.Clear();
            Status = $"LIT: failed to parse {SourcePath}: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[LIT] Failed to parse {SourcePath}: {ex}");
            return false;
        }
    }

    public LitLightingSample? EvaluateLighting(Vector3 cameraPosition, float gameTime)
    {
        if (Lights.Count == 0)
            return null;

        float timeOfDay = WrapTime(gameTime * 2880f);
        int clearGroupIndex = 0;
        int baseLightIndex = Lights.FindIndex(light => light.IsDefaultLight);

        LitLight? baseLight = baseLightIndex >= 0 && baseLightIndex < Lights.Count ? Lights[baseLightIndex] : null;
        LitLightingSample? baseSample = baseLight != null
            ? EvaluateSingleLight(baseLight, timeOfDay, clearGroupIndex, dominantWeight: 1f)
            : null;

        Vector3 direct = baseSample?.DirectColor ?? new Vector3(0.8f, 0.78f, 0.7f);
        Vector3 ambient = baseSample?.AmbientColor ?? new Vector3(0.55f, 0.55f, 0.6f);
        Vector3 fog = baseSample?.FogColor ?? new Vector3(0.6f, 0.7f, 0.85f);
        Vector3 skyTop = baseSample?.SkyTopColor ?? fog;
        Vector3 skyHorizon = baseSample?.SkyHorizonColor ?? fog;
        float fogEnd = baseSample?.FogEnd ?? 1500f;
        float fogStartScalar = baseSample?.FogStartScalar ?? 0.25f;

        float totalWeight = 0f;
        int dominantLightIndex = baseSample?.DominantLightIndex ?? -1;
        string dominantLightName = baseSample?.DominantLightName ?? "None";
        float dominantWeight = 0f;

        for (int i = 0; i < Lights.Count; i++)
        {
            LitLight light = Lights[i];
            if (light.IsDefaultLight)
                continue;

            float spatialWeight = ComputeSpatialWeight(light, cameraPosition);
            if (spatialWeight <= 0f)
                continue;

            LitLightingSample? localSample = EvaluateSingleLight(light, timeOfDay, clearGroupIndex, spatialWeight);
            if (localSample == null)
                continue;

            if (dominantLightIndex < 0 || spatialWeight > dominantWeight)
            {
                dominantLightIndex = localSample.DominantLightIndex;
                dominantLightName = localSample.DominantLightName;
                dominantWeight = spatialWeight;
            }

            if (totalWeight <= 0f)
            {
                direct = Vector3.Lerp(direct, localSample.DirectColor, spatialWeight);
                ambient = Vector3.Lerp(ambient, localSample.AmbientColor, spatialWeight);
                fog = Vector3.Lerp(fog, localSample.FogColor, spatialWeight);
                skyTop = Vector3.Lerp(skyTop, localSample.SkyTopColor, spatialWeight);
                skyHorizon = Vector3.Lerp(skyHorizon, localSample.SkyHorizonColor, spatialWeight);
                fogEnd = LerpScalar(fogEnd, localSample.FogEnd, spatialWeight);
                fogStartScalar = LerpScalar(fogStartScalar, localSample.FogStartScalar, spatialWeight);
                totalWeight = spatialWeight;
            }
            else
            {
                float newWeight = Math.Clamp(totalWeight + spatialWeight, 0f, 1f);
                float blend = newWeight <= 0.0001f ? 0f : spatialWeight / newWeight;
                direct = Vector3.Lerp(direct, localSample.DirectColor, blend);
                ambient = Vector3.Lerp(ambient, localSample.AmbientColor, blend);
                fog = Vector3.Lerp(fog, localSample.FogColor, blend);
                skyTop = Vector3.Lerp(skyTop, localSample.SkyTopColor, blend);
                skyHorizon = Vector3.Lerp(skyHorizon, localSample.SkyHorizonColor, blend);
                fogEnd = LerpScalar(fogEnd, localSample.FogEnd, blend);
                fogStartScalar = LerpScalar(fogStartScalar, localSample.FogStartScalar, blend);
                totalWeight = newWeight;
            }
        }

        if (dominantLightIndex < 0 && baseSample != null)
        {
            dominantLightIndex = baseSample.DominantLightIndex;
            dominantLightName = baseSample.DominantLightName;
            dominantWeight = 1f;
        }

        fogEnd = Math.Clamp(fogEnd, 50f, 100000f);
        fogStartScalar = Math.Clamp(fogStartScalar, 0.02f, 1.0f);
        float fogStart = Math.Clamp(fogEnd * fogStartScalar, 0f, fogEnd);

        return new LitLightingSample(
            dominantLightIndex,
            dominantLightName,
            dominantWeight,
            timeOfDay,
            direct,
            ambient,
            fog,
            skyTop,
            skyHorizon,
            fogEnd,
            fogStart,
            fogStartScalar);
    }

    public Vector3 EvaluateOverlayColor(LitLight light, float gameTime)
    {
        if (light.Groups.Count == 0)
            return new Vector3(1f, 0.8f, 0.2f);

        float timeOfDay = WrapTime(gameTime * 2880f);
        LitGroup group = light.Groups[0];
        if (group.TryEvaluateTrack(TrackFogColor, timeOfDay, out Vector3 fogColor))
            return fogColor;
        if (group.TryEvaluateTrack(TrackDirectColor, timeOfDay, out Vector3 directColor))
            return directColor;
        return new Vector3(1f, 0.8f, 0.2f);
    }

    private LitLightingSample? EvaluateSingleLight(LitLight light, float timeOfDay, int groupIndex, float dominantWeight)
    {
        if (light.Groups.Count <= groupIndex)
            return null;

        LitGroup group = light.Groups[groupIndex];
        Vector3 direct = group.TryEvaluateTrack(TrackDirectColor, timeOfDay, out Vector3 directColor)
            ? directColor
            : new Vector3(0.8f, 0.78f, 0.7f);
        Vector3 ambient = group.TryEvaluateTrack(TrackAmbientColor, timeOfDay, out Vector3 ambientColor)
            ? ambientColor
            : new Vector3(0.55f, 0.55f, 0.6f);
        Vector3 fog = group.TryEvaluateTrack(TrackFogColor, timeOfDay, out Vector3 fogColor)
            ? fogColor
            : new Vector3(0.6f, 0.7f, 0.85f);
        Vector3 skyTop = group.TryEvaluateTrack(TrackSkyTop, timeOfDay, out Vector3 topColor)
            ? topColor
            : fog;
        Vector3 skyHorizon = group.TryEvaluateTrack(TrackSkyHorizon, timeOfDay, out Vector3 horizonColor)
            ? horizonColor
            : fog;
        float fogEnd = group.EvaluateFogEnd(timeOfDay);
        float fogStartScalar = group.EvaluateFogStartScaler(timeOfDay);
        float fogStart = Math.Clamp(fogEnd * Math.Clamp(fogStartScalar, 0.02f, 1f), 0f, fogEnd);

        return new LitLightingSample(
            light.Index,
            light.DisplayName,
            dominantWeight,
            timeOfDay,
            direct,
            ambient,
            fog,
            skyTop,
            skyHorizon,
            fogEnd,
            fogStart,
            fogStartScalar);
    }

    private LitGroup ReadGroup(BinaryReader reader, int groupIndex, int trackCount)
    {
        var lengths = new int[trackCount];
        for (int i = 0; i < trackCount; i++)
            lengths[i] = Math.Clamp(reader.ReadInt32(), 0, 32);

        var tracks = new List<LitTrack>(trackCount);
        for (int trackIndex = 0; trackIndex < trackCount; trackIndex++)
        {
            var keyframes = new List<LitColorKeyframe>(lengths[trackIndex]);
            for (int sampleIndex = 0; sampleIndex < 32; sampleIndex++)
            {
                int timeOfDay = reader.ReadInt32();
                uint packedColor = unchecked((uint)reader.ReadInt32());
                if (sampleIndex < lengths[trackIndex])
                    keyframes.Add(new LitColorKeyframe(timeOfDay, packedColor, DecodeBgrx(packedColor)));
            }

            tracks.Add(new LitTrack(trackIndex, keyframes));
        }

        float[] fogEndSamples = ReadFloatArray(reader, 32);
        float[] fogStartScalers = ReadFloatArray(reader, 32);
        int highlightSky = reader.ReadInt32();
        var skyFloatBands = new IReadOnlyList<float>[]
        {
            ReadFloatArray(reader, 32),
            ReadFloatArray(reader, 32),
            ReadFloatArray(reader, 32),
            ReadFloatArray(reader, 32),
        };
        int cloudMask = reader.ReadInt32();
        IReadOnlyList<float>[] parameterBands = Array.Empty<IReadOnlyList<float>>();

        if (Version >= Version85)
        {
            parameterBands = new IReadOnlyList<float>[]
            {
                ReadFloatArray(reader, 10),
                ReadFloatArray(reader, 10),
                ReadFloatArray(reader, 10),
                ReadFloatArray(reader, 10),
            };
        }

        return new LitGroup(groupIndex, tracks, fogEndSamples, fogStartScalers, highlightSky, skyFloatBands, cloudMask, parameterBands);
    }

    private LitGroup ReadLegacyVersion02PartialGroup(BinaryReader reader, int groupIndex)
    {
        const int legacyTrackCount = 17;

        var lengths = new int[legacyTrackCount];
        for (int i = 0; i < legacyTrackCount; i++)
            lengths[i] = Math.Clamp(reader.ReadInt32(), 0, 32);

        var tracks = new List<LitTrack>(legacyTrackCount);
        for (int trackIndex = 0; trackIndex < legacyTrackCount; trackIndex++)
        {
            var keyframes = new List<LitColorKeyframe>(lengths[trackIndex]);
            for (int sampleIndex = 0; sampleIndex < 32; sampleIndex++)
            {
                int timeOfDay = reader.ReadInt32();
                uint packedColor = unchecked((uint)reader.ReadInt32());
                if (sampleIndex < lengths[trackIndex])
                    keyframes.Add(new LitColorKeyframe(timeOfDay, packedColor, DecodeBgrx(packedColor)));
            }

            tracks.Add(new LitTrack(trackIndex, keyframes));
        }

        var legacyFloatBands = new List<IReadOnlyList<float>>(7);
        int availableLegacyBands = (int)((reader.BaseStream.Length - reader.BaseStream.Position) / (32 * sizeof(float)));
        int bandCount = Math.Clamp(availableLegacyBands, 0, 7);
        for (int i = 0; i < bandCount; i++)
            legacyFloatBands.Add(ReadFloatArray(reader, 32));

        while (legacyFloatBands.Count < 7)
            legacyFloatBands.Add(Array.Empty<float>());

        var skyFloatBands = new IReadOnlyList<float>[]
        {
            legacyFloatBands[2],
            legacyFloatBands[3],
            legacyFloatBands[4],
            legacyFloatBands[5],
        };

        IReadOnlyList<float>[] parameterBands = legacyFloatBands[6].Count > 0
            ? new[] { legacyFloatBands[6] }
            : Array.Empty<IReadOnlyList<float>>();

        return new LitGroup(
            groupIndex,
            tracks,
            legacyFloatBands[0],
            legacyFloatBands[1],
            highlightSky: 0,
            skyFloatBands,
            cloudMask: 0,
            parameterBands);
    }

    private static (int ChunkX, int ChunkY, int ChunkRadius, Vector3 Position, float RadiusRaw, float DropoffRaw, string Name) ReadLightHeader(BinaryReader reader, int lightIndex)
    {
        int chunkX = reader.ReadInt32();
        int chunkY = reader.ReadInt32();
        int chunkRadius = reader.ReadInt32();
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        float z = reader.ReadSingle();
        float radiusRaw = reader.ReadSingle();
        float dropoffRaw = reader.ReadSingle();
        string name = ReadFixedString(reader, 32);
        return (chunkX, chunkY, chunkRadius, new Vector3(x, y, z), radiusRaw, dropoffRaw, string.IsNullOrWhiteSpace(name) ? $"Light {lightIndex}" : name);
    }

    private string? ResolveSourcePath()
    {
        if (!string.IsNullOrWhiteSpace(_preferredSourcePath))
        {
            for (int i = 0; i < _availableSourcePaths.Count; i++)
            {
                if (string.Equals(_availableSourcePaths[i], _preferredSourcePath, StringComparison.OrdinalIgnoreCase))
                    return _availableSourcePaths[i];
            }
        }

        return _availableSourcePaths.Count > 0 ? _availableSourcePaths[0] : null;
    }

    private IReadOnlyList<string> ResolveAvailableSourcePaths()
    {
        string[] candidates =
        {
            $"World\\{_mapName}\\lights.lit",
            $"World\\Maps\\{_mapName}\\lights.lit",
            $"World\\{_mapName}\\areatest.lit",
            $"World\\Maps\\{_mapName}\\areatest.lit",
            $"World\\{_mapName}\\light.lit",
            $"World\\Maps\\{_mapName}\\light.lit",
        };

        var available = new List<string>(candidates.Length);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < candidates.Length; i++)
        {
            string candidate = candidates[i];
            if (!seen.Add(candidate))
                continue;

            if (_dataSource.FileExists(candidate))
                available.Add(candidate);
        }

        return available;
    }

    private static float[] ReadFloatArray(BinaryReader reader, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = reader.ReadSingle();
        return result;
    }

    private static int GetTrackCountForVersion(uint version)
    {
        return version switch
        {
            Version02Test => 17,
            Version83 => 14,
            _ => 18,
        };
    }

    private static string ReadFixedString(BinaryReader reader, int length)
    {
        byte[] bytes = reader.ReadBytes(length);
        int end = Array.IndexOf(bytes, (byte)0);
        if (end < 0)
            end = bytes.Length;
        string raw = Encoding.ASCII.GetString(bytes, 0, end).Trim();
        return raw.Replace("\uFFFD", string.Empty).Trim();
    }

    private static Vector3 DecodeBgrx(uint packedColor)
    {
        float b = (packedColor & 0xFF) / 255f;
        float g = ((packedColor >> 8) & 0xFF) / 255f;
        float r = ((packedColor >> 16) & 0xFF) / 255f;
        return new Vector3(r, g, b);
    }

    private static float EvaluateFloatCurve(IReadOnlyList<float> samples, float timeOfDay, float fallback)
    {
        if (samples.Count == 0)
            return fallback;

        int populatedCount = samples.Count;
        while (populatedCount > 1 && Math.Abs(samples[populatedCount - 1]) < 0.0001f)
            populatedCount--;

        if (populatedCount <= 0)
            return fallback;
        if (populatedCount == 1)
            return Math.Abs(samples[0]) < 0.0001f ? fallback : samples[0];

        float wrappedTime = WrapTime(timeOfDay);
        float step = 2880f / (populatedCount - 1);
        float indexFloat = wrappedTime / step;
        int lowerIndex = (int)MathF.Floor(indexFloat);
        int upperIndex = Math.Min(lowerIndex + 1, populatedCount - 1);
        float t = Math.Clamp(indexFloat - lowerIndex, 0f, 1f);

        float a = samples[lowerIndex];
        float b = samples[upperIndex];
        if (Math.Abs(a) < 0.0001f && Math.Abs(b) < 0.0001f)
            return fallback;
        if (Math.Abs(a) < 0.0001f)
            a = b;
        if (Math.Abs(b) < 0.0001f)
            b = a;

        return a + (b - a) * t;
    }

    private static float ComputeSpatialWeight(LitLight light, Vector3 cameraPosition)
    {
        if (!light.HasMeaningfulPosition)
            return 0f;

        float radius = light.Radius > 0.01f ? light.Radius : 0f;
        float dropoff = light.Dropoff > 0.01f ? light.Dropoff : 0f;
        float distance = Vector3.Distance(cameraPosition, light.Position);

        if (radius <= 0f && dropoff <= 0f)
            return 0f;

        if (radius > 0f && distance <= radius)
            return 1f;

        if (dropoff <= 0f)
            return 0f;

        float fadeDistance = Math.Max(1f, dropoff);
        float distancePastRadius = Math.Max(0f, distance - radius);
        if (distancePastRadius >= fadeDistance)
            return 0f;

        return 1f - (distancePastRadius / fadeDistance);
    }

    private static float NormalizeDistance(float rawDistance)
    {
        if (rawDistance <= 0f)
            return 0f;

        float normalized = rawDistance / RadiusScale;
        return normalized > 0.01f ? normalized : rawDistance;
    }

    private static float WrapTime(float timeOfDay)
    {
        float wrapped = timeOfDay % 2880f;
        return wrapped < 0f ? wrapped + 2880f : wrapped;
    }

    private static float LerpScalar(float a, float b, float t)
    {
        return a + (b - a) * Math.Clamp(t, 0f, 1f);
    }
}