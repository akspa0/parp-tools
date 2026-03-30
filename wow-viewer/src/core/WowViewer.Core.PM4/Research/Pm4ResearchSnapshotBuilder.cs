using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Research;

public static class Pm4ResearchSnapshotBuilder
{
    public static Pm4ExplorationSnapshot CreateSnapshot(Pm4ResearchDocument document)
    {
        return new Pm4ExplorationSnapshot(
            document.Version,
            document.Chunks.Count,
            document.KnownChunks.Mslk.Count,
            document.KnownChunks.Mspv.Count,
            document.KnownChunks.Mspi.Count,
            document.KnownChunks.Msvt.Count,
            document.KnownChunks.Msvi.Count,
            document.KnownChunks.Msur.Count,
            document.KnownChunks.Mscn.Count,
            document.KnownChunks.Mprl.Count,
            document.KnownChunks.Mprr.Count,
            ComputeBounds(document.KnownChunks.Msvt),
            ComputeBounds(document.KnownChunks.Mscn),
            ComputeBounds(document.KnownChunks.Mprl.Select(static entry => entry.Position).ToList()),
            document.Diagnostics);
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