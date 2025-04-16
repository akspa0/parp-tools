using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using WoWToolbox.Core.WMO;

namespace WoWToolbox.MSCNExplorer.Analysis
{
    public class ProximityResult
    {
        public Vector3 Point { get; set; }
        public float MinDistance { get; set; }
        public int ClosestVertexIndex { get; set; }
    }

    public class MscnMeshComparisonAnalyzer
    {
        public List<Vector3> MscnPoints { get; private set; } = new();
        public WmoGroupMesh Mesh { get; set; } = null!;
        public List<ProximityResult> Results { get; private set; } = new();

        public MscnMeshComparisonAnalyzer() { }

        public void LoadMscnPoints(string filePath)
        {
            MscnPoints.Clear();
            foreach (var line in File.ReadLines(filePath))
            {
                var trimmed = line.Trim();
                if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                    continue;
                var parts = trimmed.Split(new[] { ' ', ',' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 3)
                    continue;
                if (float.TryParse(parts[0], out float x) &&
                    float.TryParse(parts[1], out float y) &&
                    float.TryParse(parts[2], out float z))
                {
                    MscnPoints.Add(new Vector3(x, y, z));
                }
            }
        }

        public void LoadWmoMesh(string wmoGroupFilePath)
        {
            using var stream = File.OpenRead(wmoGroupFilePath);
            Mesh = WmoGroupMesh.LoadFromStream(stream);
        }

        public void AnalyzeProximity(float threshold = 0.1f)
        {
            if (Mesh == null || Mesh.Vertices.Count == 0)
                throw new InvalidOperationException("Mesh must be loaded before analysis.");
            Results.Clear();
            float minDist = float.MaxValue, maxDist = float.MinValue, sumDist = 0;
            int withinThreshold = 0;
            for (int i = 0; i < MscnPoints.Count; i++)
            {
                var point = MscnPoints[i];
                float bestDist = float.MaxValue;
                int bestIdx = -1;
                for (int v = 0; v < Mesh.Vertices.Count; v++)
                {
                    float dist = Vector3.Distance(point, Mesh.Vertices[v].Position);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx = v;
                    }
                }
                Results.Add(new ProximityResult { Point = point, MinDistance = bestDist, ClosestVertexIndex = bestIdx });
                if (bestDist < minDist) minDist = bestDist;
                if (bestDist > maxDist) maxDist = bestDist;
                sumDist += bestDist;
                if (bestDist <= threshold) withinThreshold++;
            }
            float avgDist = Results.Count > 0 ? sumDist / Results.Count : 0;
            float pctWithin = Results.Count > 0 ? (withinThreshold * 100.0f / Results.Count) : 0;
            Console.WriteLine($"MSCN-to-mesh vertex proximity:");
            Console.WriteLine($"  Points analyzed: {Results.Count}");
            Console.WriteLine($"  Min distance: {minDist:F4}");
            Console.WriteLine($"  Max distance: {maxDist:F4}");
            Console.WriteLine($"  Avg distance: {avgDist:F4}");
            Console.WriteLine($"  % within {threshold:F3}: {pctWithin:F2}%");
        }

        public void ExportResults(string outputPath)
        {
            // TODO: Export analysis results (CSV, JSON, etc.)
            throw new NotImplementedException();
        }

        // (Optional) Visual overlay/export hooks
    }
} 