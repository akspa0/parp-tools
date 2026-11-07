using System;
using System.Numerics;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// A measured placement correction applied to shape-match proposals. The operator-reported
/// bias is a small offset along -X; keeping it as a separate, visible value means the matcher
/// math is not silently rewritten and the correction can be inspected, reset, or re-measured.
/// </summary>
public readonly record struct Pm4CentroidCalibration(
    Vector3 Offset,
    string Source = "uncalibrated")
{
    public static readonly Pm4CentroidCalibration None = new(Vector3.Zero, "uncalibrated");

    public bool IsIdentity => Offset.LengthSquared() < 1e-8f;

    /// <summary>Applies the correction exactly once to a solved placement position.</summary>
    public Vector3 Apply(Vector3 solvedPosition) => solvedPosition + Offset;

    /// <summary>Applies the correction at most once based on <paramref name="calibrationApplied"/>.</summary>
    public Vector3 Apply(Vector3 solvedPosition, bool calibrationApplied) =>
        calibrationApplied ? solvedPosition : solvedPosition + Offset;

    /// <summary>Returns a calibration with the offset negated, for sign-error diagnosis.</summary>
    public Pm4CentroidCalibration Inverted() => this with { Offset = -Offset, Source = Source + " (inverted)" };

    public Pm4CentroidCalibration WithOffset(Vector3 offset, string source) => new(offset, source);
}
