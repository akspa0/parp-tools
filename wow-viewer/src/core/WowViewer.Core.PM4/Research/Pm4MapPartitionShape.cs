using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Research;

internal static class Pm4MapPartitionShape
{
    // Fan triangulation is a measurement convention, not a claim about source asset topology.
    public static Pm4MapShape Measure(IEnumerable<(Vector3 A, Vector3 B, Vector3 C)> input)
    {
        var triangles = new List<(Vector3 A, Vector3 B, Vector3 C)>();
        int invalid = 0;
        foreach (var t in input)
        {
            if (!Finite(t.A) || !Finite(t.B) || !Finite(t.C)
                || Vector3.Cross(t.B - t.A, t.C - t.A).LengthSquared() <= 1e-12f)
                invalid++;
            else
                triangles.Add(t);
        }
        if (triangles.Count == 0)
            return new Pm4MapShape(0, invalid, Vector3.Zero, 0, null, new Dictionary<string, int>());

        // Double accumulation avoids loss of small local features at large map coordinates.
        double x = 0, y = 0, z = 0;
        foreach (var t in triangles)
            foreach (var p in new[] { t.A, t.B, t.C }) { x += p.X; y += p.Y; z += p.Z; }
        double n = triangles.Count * 3d;
        x /= n; y /= n; z /= n;
        double square = 0;
        foreach (var t in triangles)
            foreach (var p in new[] { t.A, t.B, t.C })
                square += (p.X - x) * (p.X - x) + (p.Y - y) * (p.Y - y) + (p.Z - z) * (p.Z - z);
        double radius = Math.Sqrt(square / n);
        var histogram = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var t in triangles)
        {
            double[] edges = [Vector3.Distance(t.A, t.B), Vector3.Distance(t.B, t.C), Vector3.Distance(t.C, t.A)];
            Array.Sort(edges);
            double cx = (t.A.X + (double)t.B.X + t.C.X) / 3 - x;
            double cy = (t.A.Y + (double)t.B.Y + t.C.Y) / 3 - y;
            double cz = (t.A.Z + (double)t.B.Z + t.C.Z) / 3 - z;
            // Quantized dimensionless lengths plus triangle-centroid radial arrangement.
            string key = string.Join('_', edges.Select(e => Bin(e / radius))) + "_" + Bin(Math.Sqrt(cx * cx + cy * cy + cz * cz) / radius);
            histogram[key] = histogram.GetValueOrDefault(key) + 1;
        }
        string text = string.Join('|', histogram.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}:{p.Value}"));
        string signature = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
        return new Pm4MapShape(triangles.Count, invalid, new Vector3((float)x, (float)y, (float)z),
            (float)radius, signature, histogram);
    }

    private static string Bin(double value) => Math.Round(value / 0.02, MidpointRounding.ToEven).ToString(CultureInfo.InvariantCulture);
    private static bool Finite(Vector3 p) => float.IsFinite(p.X) && float.IsFinite(p.Y) && float.IsFinite(p.Z);
}
