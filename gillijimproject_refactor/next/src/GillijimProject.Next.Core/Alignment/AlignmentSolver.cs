using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace GillijimProject.Next.Core.Alignment;

public readonly record struct AlignmentResult(
    bool FlipX,
    bool FlipY,
    int RotZDeg,
    double TranslateX,
    double TranslateY,
    double Score,
    bool NegativeParity)
{
    public override string ToString()
        => $"flipX={FlipX} flipY={FlipY} rotZ={RotZDeg} tx={TranslateX.ToString(CultureInfo.InvariantCulture)} ty={TranslateY.ToString(CultureInfo.InvariantCulture)} score={Score:F6} parity={(NegativeParity ? "-" : "+")}";
}

public static class AlignmentSolver
{
    public static AlignmentResult SolveToTileBounds(
        IReadOnlyList<(double X, double Y)> srcPoints,
        (double MinX, double MinY, double MaxX, double MaxY) dstBounds,
        int sampleTarget = 4000)
    {
        if (srcPoints == null || srcPoints.Count == 0) throw new ArgumentException("srcPoints empty");
        var (minX, minY, maxX, maxY) = dstBounds;
        var dstCx = (minX + maxX) * 0.5;
        var dstCy = (minY + maxY) * 0.5;

        var candList = new List<(bool fx, bool fy, int rz)>
        {
            (false,false,0),(false,false,90),(false,false,180),(false,false,270),
            (true,false,0),(true,false,90),(true,false,180),(true,false,270),
            (false,true,0),(false,true,90),(false,true,180),(false,true,270)
        };

        // Build a sample of points (uniform stride)
        int stride = Math.Max(1, srcPoints.Count / Math.Max(1, sampleTarget));
        var sample = new List<(double X, double Y)>(Math.Min(srcPoints.Count, sampleTarget));
        for (int i = 0; i < srcPoints.Count; i += stride) sample.Add(srcPoints[i]);

        AlignmentResult best = default;
        double bestScore = double.NegativeInfinity;

        foreach (var (fx, fy, rz) in candList)
        {
            // Transform sample, compute center, derive translation, then score by in-bounds fraction
            var tmp = new List<(double X, double Y)>(sample.Count);
            double sx = 0, sy = 0;
            foreach (var (x0, y0) in sample)
            {
                var (x1, y1) = ApplyD4(x0, y0, fx, fy, rz);
                sx += x1; sy += y1;
                tmp.Add((x1, y1));
            }
            double scx = sx / tmp.Count, scy = sy / tmp.Count;
            double tx = dstCx - scx, ty = dstCy - scy;

            // Score: fraction inside bounds, minus overflow penalty
            int inside = 0; double overflow = 0;
            foreach (var (x1, y1) in tmp)
            {
                double x = x1 + tx, y = y1 + ty;
                bool inX = x >= minX && x <= maxX;
                bool inY = y >= minY && y <= maxY;
                if (inX && inY) inside++;
                else
                {
                    double dx = 0, dy = 0;
                    if (!inX) dx = x < minX ? (minX - x) : (x - maxX);
                    if (!inY) dy = y < minY ? (minY - y) : (y - maxY);
                    overflow += dx + dy;
                }
            }
            double coverage = (double)inside / Math.Max(1, tmp.Count);
            double score = coverage - 0.0001 * overflow;
            if (score > bestScore)
            {
                bool negativeParity = ((fx ? 1 : 0) + (fy ? 1 : 0)) % 2 == 1;
                bestScore = score;
                best = new AlignmentResult(fx, fy, rz, tx, ty, score, negativeParity);
            }
        }

        return best;
    }

    public static (double X, double Y) ApplyD4(double x, double y, bool flipX, bool flipY, int rot)
    {
        if (flipX) x = -x;
        if (flipY) y = -y;
        rot = ((rot % 360) + 360) % 360;
        return rot switch
        {
            0 => (x, y),
            90 => (-y, x),
            180 => (-x, -y),
            270 => (y, -x),
            _ => (x, y)
        };
    }
}
