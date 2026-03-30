using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Research;

public static class Pm4ResearchAnalyzer
{
    private static readonly string[] KnownTypedSignatures =
    {
        "MVER",
        "MSHD",
        "MSLK",
        "MSPV",
        "MSPI",
        "MSVT",
        "MSVI",
        "MSUR",
        "MSCN",
        "MPRL",
        "MPRR",
        "MDBH",
        "MDBI",
        "MDBF",
        "MDOS",
        "MDSF"
    };

    public static Pm4AnalysisReport Analyze(Pm4ResearchDocument document)
    {
        IReadOnlyList<Vector3> mprlPositions = document.KnownChunks.Mprl.Select(static entry => entry.Position).ToList();

        return new Pm4AnalysisReport(
            document.SourcePath,
            document.Version,
            document.Chunks.Select(static chunk => new Pm4ChunkSummary(chunk.Signature, chunk.Size)).ToList(),
            document.Chunks
                .Where(static chunk => !KnownTypedSignatures.Contains(chunk.Signature, StringComparer.Ordinal))
                .Select(static chunk => $"{chunk.Signature}:{chunk.Size}")
                .ToList(),
            BuildVectorSetSummary("MSPV", document.KnownChunks.Mspv),
            BuildVectorSetSummary("MSVT", document.KnownChunks.Msvt),
            BuildVectorSetSummary("MSCN", document.KnownChunks.Mscn),
            BuildVectorSetSummary("MPRL", mprlPositions),
            BuildMprlSummary(document.KnownChunks.Mprl),
            Pm4TerminologyCatalog.ForInspectReport(),
            BuildCk24Summaries(document.KnownChunks.Msur),
            BuildResearchNotes(document.KnownChunks.Msur),
            document.Diagnostics);
    }

    private static IReadOnlyList<string> BuildResearchNotes(IReadOnlyList<Pm4MsurEntry> surfaces)
    {
        if (!surfaces.Any(static surface => surface.Ck24 != 0 && surface.Ck24ObjectId != 0))
            return Array.Empty<string>();

        return
        [
            "CK24 low-16 object values, read as integers, remain a plausible UniqueID candidate for the represented PM4 objects. Current evidence is value-range alignment only; correlate against real placed-object datasets before treating that identity mapping as confirmed."
        ];
    }

    private static Pm4VectorSetSummary BuildVectorSetSummary(string name, IReadOnlyList<Vector3> points)
    {
        if (points.Count == 0)
            return new Pm4VectorSetSummary(name, 0, null, null, Array.Empty<Pm4QuadrantSummary>());

        Pm4Bounds3 bounds = ComputeBounds(points)!;
        Vector3 centroid = Vector3.Zero;
        for (int index = 0; index < points.Count; index++)
            centroid += points[index];

        centroid /= points.Count;

        var quadrants = new[]
        {
            ComputeQuadrants("XY", points, static point => point.X, static point => point.Y, bounds.Min.X, bounds.Max.X, bounds.Min.Y, bounds.Max.Y),
            ComputeQuadrants("XZ", points, static point => point.X, static point => point.Z, bounds.Min.X, bounds.Max.X, bounds.Min.Z, bounds.Max.Z),
            ComputeQuadrants("YZ", points, static point => point.Y, static point => point.Z, bounds.Min.Y, bounds.Max.Y, bounds.Min.Z, bounds.Max.Z)
        };

        return new Pm4VectorSetSummary(name, points.Count, bounds, centroid, quadrants);
    }

    private static Pm4QuadrantSummary ComputeQuadrants(
        string plane,
        IReadOnlyList<Vector3> points,
        Func<Vector3, float> axisA,
        Func<Vector3, float> axisB,
        float minA,
        float maxA,
        float minB,
        float maxB)
    {
        float midA = (minA + maxA) * 0.5f;
        float midB = (minB + maxB) * 0.5f;
        int lowLow = 0;
        int lowHigh = 0;
        int highLow = 0;
        int highHigh = 0;

        for (int index = 0; index < points.Count; index++)
        {
            bool highA = axisA(points[index]) >= midA;
            bool highB = axisB(points[index]) >= midB;

            if (highA)
            {
                if (highB)
                    highHigh++;
                else
                    highLow++;
            }
            else
            {
                if (highB)
                    lowHigh++;
                else
                    lowLow++;
            }
        }

        return new Pm4QuadrantSummary(plane, midA, midB, lowLow, lowHigh, highLow, highHigh);
    }

    private static Pm4MprlSummary BuildMprlSummary(IReadOnlyList<Pm4MprlEntry> entries)
    {
        if (entries.Count == 0)
            return new Pm4MprlSummary(0, 0, 0, null, null, null, null);

        int normalCount = 0;
        int terminatorCount = 0;
        short floorMin = short.MaxValue;
        short floorMax = short.MinValue;
        float rotationMin = float.MaxValue;
        float rotationMax = float.MinValue;

        for (int index = 0; index < entries.Count; index++)
        {
            Pm4MprlEntry entry = entries[index];
            if (entry.Unk16 == 0)
                normalCount++;
            else
                terminatorCount++;

            if (entry.Unk14 < floorMin)
                floorMin = entry.Unk14;
            if (entry.Unk14 > floorMax)
                floorMax = entry.Unk14;

            float degrees = entry.Unk04 * (360f / 65536f);
            if (degrees < rotationMin)
                rotationMin = degrees;
            if (degrees > rotationMax)
                rotationMax = degrees;
        }

        return new Pm4MprlSummary(
            entries.Count,
            normalCount,
            terminatorCount,
            floorMin,
            floorMax,
            rotationMin,
            rotationMax);
    }

    private static IReadOnlyList<Pm4Ck24Summary> BuildCk24Summaries(IReadOnlyList<Pm4MsurEntry> surfaces)
    {
        return surfaces
            .Where(static surface => surface.Ck24 != 0)
            .GroupBy(static surface => surface.Ck24)
            .Select(group => new Pm4Ck24Summary(
                group.Key,
                group.First().Ck24Type,
                group.First().Ck24ObjectId,
                group.Count(),
                group.Sum(static surface => surface.IndexCount),
                group.Average(static surface => surface.Height),
                group.Select(static surface => surface.MdosIndex).Distinct().Count()))
            .OrderByDescending(static summary => summary.SurfaceCount)
            .ThenBy(static summary => summary.Ck24)
            .Take(32)
            .ToList();
    }

    private static Pm4Bounds3? ComputeBounds(IReadOnlyList<Vector3> points)
    {
        if (points.Count == 0)
            return null;

        Vector3 min = points[0];
        Vector3 max = points[0];
        for (int index = 1; index < points.Count; index++)
        {
            min = Vector3.Min(min, points[index]);
            max = Vector3.Max(max, points[index]);
        }

        return new Pm4Bounds3(min, max);
    }
}