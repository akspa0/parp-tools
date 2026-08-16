using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;

namespace WowViewer.Core.IO.AssetReferences;

/// <summary>Whether a model's format era is known to read, known blocked, or not yet measured either way.</summary>
public enum ModelRouteStatus
{
    /// <summary>This era reads today (Spec 154 measured it, or it needs no fix).</summary>
    Readable,

    /// <summary>Spec 154 measured this era as broken. Not attempted — the failure is already known.</summary>
    Blocked,

    /// <summary>Neither confirmed readable nor confirmed blocked. Attempted normally; a failure here is
    /// reported as an ordinary unreadable asset, not manufactured into a blocked-route claim.</summary>
    Unknown,
}

public readonly record struct ModelRouteClassification(ModelRouteStatus Status, string RouteLabel, string? Reason);

/// <summary>
/// Classifies a model's format era against Spec 154's measured findings, before any attempt to fully
/// parse it. This exists so a build whose model route is known not to work reports thousands of
/// identical, already-understood blocked-route outcomes as one aggregated fact — never as noise
/// indistinguishable from unrelated per-asset corruption.
/// </summary>
/// <remarks>
/// Era detection itself is <see cref="M2ModelReaderDispatcher.DetectEra"/> — this type adds no decoding
/// of its own. It only maps the returned era (or the dispatcher's own refusal) onto what Spec 154 actually
/// measured. An era Spec 154 did not measure either way stays <see cref="ModelRouteStatus.Unknown"/>
/// rather than being guessed into either bucket.
/// </remarks>
public static class ModelRouteClassifier
{
    public static ModelRouteClassification Classify(byte[] modelBytes, string sourcePath)
    {
        try
        {
            M2Era1121EraTag era = M2ModelReaderDispatcher.DetectEra(modelBytes, sourcePath);
            return era switch
            {
                M2Era1121EraTag.Mdlx =>
                    Readable("MDLX"),
                M2Era1121EraTag.Md20_3X_V108 =>
                    Readable("MD20 0x108"),
                M2Era1121EraTag.Md20_1X_V100_Era100 => Blocked(
                    "MD20 0x100 (1.0.0 Era100 layout)",
                    "Spec 154: M2Era100ModelReader uses the wrong fallback bone-read constants; bones=0."),
                M2Era1121EraTag.Md20_4X_V109 => Blocked(
                    "MD20 0x109+ (4.x)",
                    "Spec 154: unhandled camera-parsing exception measured at 4.0.0.11927; also beyond " +
                    "this project's declared 4.0.0 scope ceiling."),
                M2Era1121EraTag.Md20_1X_V100 =>
                    Unknown("MD20 0x100 (non-Era100 layout)"),
                M2Era1121EraTag.Md20_1X_V101 =>
                    Unknown("MD20 0x101"),
                _ => Unknown("unrecognized era"),
            };
        }
        catch (NotSupportedException ex)
        {
            // M2ModelReaderDispatcher.DetectEra itself refuses MD20 0x102-0x107 outright — the
            // exception IS the classification for this range, measured directly at 3.0.1.8303.
            return Blocked("MD20 0x102-0x107 (2.x TBC era)", ex.Message);
        }
        catch (Exception ex) when (ex is InvalidDataException or ArgumentException)
        {
            // Too small, bad magic, or otherwise not even classifiable. Not a blocked route — let the
            // normal read path produce its own honest per-asset failure.
            return Unknown($"undetectable: {ex.Message}");
        }
    }

    private static ModelRouteClassification Readable(string routeLabel)
        => new(ModelRouteStatus.Readable, routeLabel, null);

    private static ModelRouteClassification Blocked(string routeLabel, string reason)
        => new(ModelRouteStatus.Blocked, routeLabel, reason);

    private static ModelRouteClassification Unknown(string routeLabel)
        => new(ModelRouteStatus.Unknown, routeLabel, null);
}
