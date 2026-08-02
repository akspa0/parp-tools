using System.Collections.ObjectModel;
using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Lit;

/// <summary>
/// A fully decoded Alpha-era LIT file.
/// </summary>
public sealed record LitFileProfile
{
    public LitFileProfile(
        string sourcePath,
        uint versionNumber,
        int rawLightCount,
        int trackCount,
        int groupStride,
        IEnumerable<LitLightProfile> lights)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(lights);

        SourcePath = sourcePath;
        VersionNumber = versionNumber;
        RawLightCount = rawLightCount;
        TrackCount = trackCount;
        GroupStride = groupStride;
        Lights = LitReadOnly.Copy(lights);
    }

    public string SourcePath { get; }

    public uint VersionNumber { get; }

    public int RawLightCount { get; }

    public int TrackCount { get; }

    public int GroupStride { get; }

    public IReadOnlyList<LitLightProfile> Lights { get; }

    public bool IsSinglePartialProfile => RawLightCount == -1;
}

/// <summary>
/// The fixed 64-byte spatial header associated with a list-based LIT light.
/// A negative-count partial profile has no spatial header.
/// </summary>
/// <param name="Position">Raw fixed-point position exactly as stored on disk.</param>
/// <param name="Radius">Raw fixed-point core radius exactly as stored on disk.</param>
/// <param name="Dropoff">Raw fixed-point falloff distance exactly as stored on disk.</param>
/// <remarks>
/// The constructor parameters are the untouched disk values. Anything working in world space must
/// use <see cref="WorldPosition"/> / <see cref="WorldRadius"/> / <see cref="WorldDropoff"/>: LIT
/// spatial records are client fixed-point at
/// <see cref="TerrainLightingMath.ClientFixedUnitsPerWorldUnit"/> (1/36), which is why unscaled
/// values plot roughly 36x outside the map.
/// </remarks>
public sealed record LitLightHeaderProfile(
    int Index,
    int ChunkX,
    int ChunkY,
    int ChunkRadius,
    Vector3 Position,
    float Radius,
    float Dropoff,
    string Name)
{
    public bool IsDefault => ChunkX == -1 && ChunkY == -1 && ChunkRadius == -1;

    /// <summary>Light centre in renderer world units.</summary>
    public Vector3 WorldPosition => Position / TerrainLightingMath.ClientFixedUnitsPerWorldUnit;

    /// <summary>Core radius in renderer world units.</summary>
    public float WorldRadius => Radius / TerrainLightingMath.ClientFixedUnitsPerWorldUnit;

    /// <summary>Falloff distance in renderer world units.</summary>
    public float WorldDropoff => Dropoff / TerrainLightingMath.ClientFixedUnitsPerWorldUnit;

    /// <summary>Outer influence radius in renderer world units.</summary>
    public float WorldOuterRadius => MathF.Max(WorldRadius, WorldRadius + MathF.Max(WorldDropoff, 0f));
}

/// <summary>
/// One spatial light and its weather/water groups, or the single group in a
/// negative-count partial profile.
/// </summary>
public sealed record LitLightProfile
{
    public LitLightProfile(
        int index,
        LitLightHeaderProfile? header,
        IEnumerable<LitLightGroupProfile> groups)
    {
        ArgumentNullException.ThrowIfNull(groups);

        Index = index;
        Header = header;
        Groups = LitReadOnly.Copy(groups);
    }

    public int Index { get; }

    public LitLightHeaderProfile? Header { get; }

    public IReadOnlyList<LitLightGroupProfile> Groups { get; }

    public bool IsPartial => Header is null;
}

public enum LitLightGroupKind
{
    Clear = 0,
    Storm = 1,
    ClearWater = 2,
    StormWater = 3,
    Partial = 4,
}

/// <summary>
/// One fixed-stride lighting group. Float bands retain their disk order:
/// fog end, fog-start scalar, and four sky bands for v8.3-v8.5; v2 exposes
/// all seven legacy bands without assigning semantics that the format does
/// not establish.
/// </summary>
public sealed record LitLightGroupProfile
{
    public LitLightGroupProfile(
        int index,
        LitLightGroupKind kind,
        IEnumerable<LitColorTrack> tracks,
        IEnumerable<LitFloatBand> floatBands,
        int? highlightSky,
        int? cloudMask,
        IEnumerable<LitFloatBand> parameterBands,
        int encodedSize)
    {
        ArgumentNullException.ThrowIfNull(tracks);
        ArgumentNullException.ThrowIfNull(floatBands);
        ArgumentNullException.ThrowIfNull(parameterBands);

        Index = index;
        Kind = kind;
        Tracks = LitReadOnly.Copy(tracks);
        FloatBands = LitReadOnly.Copy(floatBands);
        HighlightSky = highlightSky;
        CloudMask = cloudMask;
        ParameterBands = LitReadOnly.Copy(parameterBands);
        EncodedSize = encodedSize;
    }

    public int Index { get; }

    public LitLightGroupKind Kind { get; }

    public IReadOnlyList<LitColorTrack> Tracks { get; }

    public IReadOnlyList<LitFloatBand> FloatBands { get; }

    public int? HighlightSky { get; }

    public int? CloudMask { get; }

    public IReadOnlyList<LitFloatBand> ParameterBands { get; }

    public int EncodedSize { get; }

    public bool TryGetTrack(int trackIndex, out LitColorTrack track)
    {
        if ((uint)trackIndex < (uint)Tracks.Count && Tracks[trackIndex].Index == trackIndex)
        {
            track = Tracks[trackIndex];
            return true;
        }

        for (int index = 0; index < Tracks.Count; index++)
        {
            if (Tracks[index].Index == trackIndex)
            {
                track = Tracks[index];
                return true;
            }
        }

        track = null!;
        return false;
    }
}

/// <summary>
/// One float-array field retained in disk order.
/// </summary>
public sealed record LitFloatBand
{
    public LitFloatBand(int index, IEnumerable<float> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);

        Index = index;
        Samples = LitReadOnly.Copy(samples);
    }

    public int Index { get; }

    public IReadOnlyList<float> Samples { get; }
}

/// <summary>
/// One used time/BGRX pair from a LIT color track.
/// </summary>
public sealed record LitColorKeyframe(
    int TimeOfDay,
    uint PackedBgrx,
    Vector3 Color);

/// <summary>
/// A decoded LIT color track with cyclic linear interpolation over the
/// inclusive 0..2880 day domain.
/// </summary>
public sealed record LitColorTrack
{
    private readonly LitColorKeyframe[] _evaluationKeyframes;

    public LitColorTrack(int index, int declaredLength, IEnumerable<LitColorKeyframe> keyframes)
    {
        ArgumentNullException.ThrowIfNull(keyframes);
        if (declaredLength is < 0 or > LitProfileReader.MaximumTrackLength)
            throw new ArgumentOutOfRangeException(nameof(declaredLength));

        LitColorKeyframe[] copied = keyframes.ToArray();
        if (copied.Length != declaredLength)
        {
            throw new ArgumentException(
                $"Track declares {declaredLength} keyframes but received {copied.Length}.",
                nameof(keyframes));
        }

        for (int keyIndex = 0; keyIndex < copied.Length; keyIndex++)
        {
            if (copied[keyIndex].TimeOfDay is < 0 or > LitProfileReader.TimeUnitsPerDay)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(keyframes),
                    $"Keyframe {keyIndex} has time {copied[keyIndex].TimeOfDay}; expected 0..{LitProfileReader.TimeUnitsPerDay}.");
            }
        }

        Index = index;
        DeclaredLength = declaredLength;
        Keyframes = Array.AsReadOnly(copied);
        _evaluationKeyframes = copied
            .OrderBy(keyframe => keyframe.TimeOfDay)
            .ToArray();
    }

    public int Index { get; }

    public int DeclaredLength { get; }

    public IReadOnlyList<LitColorKeyframe> Keyframes { get; }

    public bool TryEvaluate(float timeOfDay, out Vector3 color)
    {
        if (!float.IsFinite(timeOfDay) || timeOfDay < 0f || timeOfDay > LitProfileReader.TimeUnitsPerDay)
        {
            throw new ArgumentOutOfRangeException(
                nameof(timeOfDay),
                timeOfDay,
                $"LIT time must be within 0..{LitProfileReader.TimeUnitsPerDay}.");
        }

        if (_evaluationKeyframes.Length == 0)
        {
            color = Vector3.Zero;
            return false;
        }

        if (_evaluationKeyframes.Length == 1)
        {
            color = _evaluationKeyframes[0].Color;
            return true;
        }

        float sampleTime = timeOfDay == LitProfileReader.TimeUnitsPerDay ? 0f : timeOfDay;
        for (int index = 0; index < _evaluationKeyframes.Length; index++)
        {
            LitColorKeyframe current = _evaluationKeyframes[index];
            if (sampleTime == current.TimeOfDay)
            {
                color = current.Color;
                return true;
            }

            if (sampleTime < current.TimeOfDay)
            {
                LitColorKeyframe previous = index == 0
                    ? _evaluationKeyframes[^1]
                    : _evaluationKeyframes[index - 1];
                float previousTime = index == 0
                    ? previous.TimeOfDay - LitProfileReader.TimeUnitsPerDay
                    : previous.TimeOfDay;
                color = Interpolate(previous, previousTime, current, current.TimeOfDay, sampleTime);
                return true;
            }
        }

        LitColorKeyframe last = _evaluationKeyframes[^1];
        LitColorKeyframe first = _evaluationKeyframes[0];
        color = Interpolate(
            last,
            last.TimeOfDay,
            first,
            first.TimeOfDay + LitProfileReader.TimeUnitsPerDay,
            sampleTime);
        return true;
    }

    public Vector3 Evaluate(float timeOfDay)
    {
        if (!TryEvaluate(timeOfDay, out Vector3 color))
            throw new InvalidOperationException($"LIT track {Index} has no keyframes to evaluate.");

        return color;
    }

    private static Vector3 Interpolate(
        LitColorKeyframe previous,
        float previousTime,
        LitColorKeyframe next,
        float nextTime,
        float sampleTime)
    {
        float span = nextTime - previousTime;
        if (span <= 0f)
            return next.Color;

        float amount = Math.Clamp((sampleTime - previousTime) / span, 0f, 1f);
        return Vector3.Lerp(previous.Color, next.Color, amount);
    }
}

internal static class LitReadOnly
{
    public static ReadOnlyCollection<T> Copy<T>(IEnumerable<T> values)
    {
        return Array.AsReadOnly(values.ToArray());
    }
}
