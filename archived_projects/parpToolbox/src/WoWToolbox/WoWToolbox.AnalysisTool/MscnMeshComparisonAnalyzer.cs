using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using WoWToolbox.Core.WMO;
using WoWToolbox.Core.Models;

namespace WoWToolbox.AnalysisTool
{
    public class MscnMeshComparisonAnalyzer
    {
        public List<Vector3> MscnPoints { get; private set; } = new();
        public MeshData? Mesh { get; private set; }

        public MscnMeshComparisonAnalyzer()
        {
            Mesh = null;
        }

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
            var wmoGroupMesh = WmoGroupMesh.LoadFromStream(stream, wmoGroupFilePath);
            Mesh = WmoGroupMeshToMeshData(wmoGroupMesh);
        }

        private static MeshData WmoGroupMeshToMeshData(WmoGroupMesh mesh)
        {
            var md = new MeshData();
            if (mesh == null) return md;
            foreach (var v in mesh.Vertices)
                md.Vertices.Add(v.Position);
            foreach (var tri in mesh.Triangles)
            {
                md.Indices.Add(tri.Index0);
                md.Indices.Add(tri.Index1);
                md.Indices.Add(tri.Index2);
            }
            return md;
        }

        public void AnalyzeProximity(float threshold = 0.1f)
        {
            // TODO: For each MSCN point, compute distance to nearest mesh vertex/edge/triangle
            // Output statistics: histogram, % within threshold, etc.
            throw new NotImplementedException();
        }

        public void ExportResults(string outputPath)
        {
            // TODO: Export analysis results (CSV, JSON, etc.)
            throw new NotImplementedException();
        }

        // (Optional) Visual overlay/export hooks
    }
} 