using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Services.WMO;
using Microsoft.Extensions.Logging;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// PM4-to-WMO spatial correlation and matching system
    /// Matches PM4 navigation data buildings with WMO render geometry
    /// </summary>
    public class Pm4WmoMatcher
    {
        private readonly ILogger<Pm4WmoMatcher> _logger;
        private readonly IWmoLoader _wmoLoader;

        // Coordinate system constants
        private const float MSCN_SCALE_FACTOR = 1.0f / 4096.0f;
        private const float CORRELATION_DISTANCE_THRESHOLD = 100.0f; // WoW units
        private const float GEOMETRY_SIMILARITY_THRESHOLD = 0.7f; // 70% similarity

        public Pm4WmoMatcher(ILogger<Pm4WmoMatcher> logger, IWmoLoader wmoLoader)
        {
            _logger = logger;
            _wmoLoader = wmoLoader;
        }

        /// <summary>
        /// PM4 building data extracted from spatial clustering
        /// </summary>
        public class Pm4Building
        {
            public string BuildingId { get; set; }
            public uint SurfaceKey { get; set; }
            public int TriangleCount { get; set; }
            public int VertexCount { get; set; }
            public List<Vector3> MscnVertices { get; set; } = new List<Vector3>();
            public Vector3 CenterPoint { get; set; }
            public Vector3 BoundingBoxMin { get; set; }
            public Vector3 BoundingBoxMax { get; set; }
            public float SurfaceArea { get; set; }
        }

        /// <summary>
        /// WMO geometry data for correlation
        /// </summary>
        public class WmoGeometry
        {
            public string WmoPath { get; set; }
            public List<Vector3> Vertices { get; set; } = new List<Vector3>();
            public List<uint> Indices { get; set; } = new List<uint>();
            public Vector3 CenterPoint { get; set; }
            public Vector3 BoundingBoxMin { get; set; }
            public Vector3 BoundingBoxMax { get; set; }
            public float SurfaceArea { get; set; }
            public int TriangleCount => Indices.Count / 3;
        }

        /// <summary>
        /// PM4-WMO correlation result
        /// </summary>
        public class CorrelationResult
        {
            public Pm4Building Pm4Building { get; set; }
            public WmoGeometry WmoGeometry { get; set; }
            public float SpatialDistance { get; set; }
            public float GeometricSimilarity { get; set; }
            public float OverallScore { get; set; }
            public bool IsValidMatch => OverallScore > GEOMETRY_SIMILARITY_THRESHOLD;
        }

        /// <summary>
        /// Perform PM4-to-WMO correlation analysis
        /// </summary>
        public List<CorrelationResult> Correlate(
            List<Pm4Building> pm4Buildings,
            string wmoFilePath)
        {
            _logger.LogInformation("Starting PM4-WMO correlation analysis");
            _logger.LogInformation($"PM4 Buildings: {pm4Buildings.Count}");
            _logger.LogInformation($"WMO File: {wmoFilePath}");

            // Load WMO geometry
            var wmoGeometry = LoadWmoGeometry(wmoFilePath);
            if (wmoGeometry == null)
            {
                _logger.LogError($"Failed to load WMO geometry from {wmoFilePath}");
                return new List<CorrelationResult>();
            }

            _logger.LogInformation($"WMO Geometry loaded: {wmoGeometry.TriangleCount} triangles, {wmoGeometry.Vertices.Count} vertices");

            var results = new List<CorrelationResult>();

            foreach (var pm4Building in pm4Buildings)
            {
                var correlation = CalculateCorrelation(pm4Building, wmoGeometry);
                results.Add(correlation);

                if (correlation.IsValidMatch)
                {
                    _logger.LogDebug($"MATCH FOUND: Building {pm4Building.BuildingId} -> WMO (Score: {correlation.OverallScore:F3})");
                }
            }

            // Sort by overall score (best matches first)
            results = results.OrderByDescending(r => r.OverallScore).ToList();

            _logger.LogInformation($"Correlation complete. Valid matches: {results.Count(r => r.IsValidMatch)}/{results.Count}");
            return results;
        }

        /// <summary>
        /// Load WMO geometry using the existing IWmoLoader
        /// </summary>
        private WmoGeometry LoadWmoGeometry(string wmoFilePath)
        {
            try
            {
                var (textures, groups) = _wmoLoader.Load(wmoFilePath);
                if (groups == null || !groups.Any())
                {
                    _logger.LogWarning($"No WMO groups found in {wmoFilePath}");
                    return null;
                }

                var geometry = new WmoGeometry
                {
                    WmoPath = wmoFilePath,
                    Vertices = new List<Vector3>(),
                    Indices = new List<uint>()
                };

                // Combine all WMO group vertices and faces
                foreach (var group in groups)
                {
                    if (group.Vertices?.Any() == true)
                    {
                        int vertexOffset = geometry.Vertices.Count;
                        ((List<Vector3>)geometry.Vertices).AddRange(group.Vertices);

                        // Add indices with vertex offset
                        if (group.Faces?.Any() == true)
                        {
                            foreach (var face in group.Faces)
                            {
                                ((List<uint>)geometry.Indices).Add((uint)(face.Item1 + vertexOffset));
                                ((List<uint>)geometry.Indices).Add((uint)(face.Item2 + vertexOffset));
                                ((List<uint>)geometry.Indices).Add((uint)(face.Item3 + vertexOffset));
                            }
                        }
                    }
                }

                // Calculate WMO bounding box and center
                CalculateGeometryProperties(geometry);

                return geometry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error loading WMO geometry from {wmoFilePath}");
                return null;
            }
        }

        /// <summary>
        /// Calculate spatial correlation between PM4 building and WMO geometry
        /// </summary>
        private CorrelationResult CalculateCorrelation(Pm4Building pm4Building, WmoGeometry wmoGeometry)
        {
            // Calculate spatial distance between centers
            var spatialDistance = Vector3.Distance(pm4Building.CenterPoint, wmoGeometry.CenterPoint);

            // Calculate geometric similarity metrics
            var geometricSimilarity = CalculateGeometricSimilarity(pm4Building, wmoGeometry);

            // Combine scores with weights
            var spatialScore = Math.Max(0, 1.0f - (spatialDistance / CORRELATION_DISTANCE_THRESHOLD));
            var overallScore = (spatialScore * 0.4f) + (geometricSimilarity * 0.6f);

            return new CorrelationResult
            {
                Pm4Building = pm4Building,
                WmoGeometry = wmoGeometry,
                SpatialDistance = spatialDistance,
                GeometricSimilarity = geometricSimilarity,
                OverallScore = overallScore
            };
        }

        /// <summary>
        /// Calculate geometric similarity between PM4 and WMO data
        /// </summary>
        private float CalculateGeometricSimilarity(Pm4Building pm4Building, WmoGeometry wmoGeometry)
        {
            var similarities = new List<float>();

            // Triangle count similarity
            var triangleRatio = Math.Min(pm4Building.TriangleCount, wmoGeometry.TriangleCount) / 
                               (float)Math.Max(pm4Building.TriangleCount, wmoGeometry.TriangleCount);
            similarities.Add(triangleRatio);

            // Vertex count similarity  
            var vertexRatio = Math.Min(pm4Building.VertexCount, wmoGeometry.Vertices.Count) / 
                             (float)Math.Max(pm4Building.VertexCount, wmoGeometry.Vertices.Count);
            similarities.Add(vertexRatio);

            // Bounding box size similarity
            var pm4Size = pm4Building.BoundingBoxMax - pm4Building.BoundingBoxMin;
            var wmoSize = wmoGeometry.BoundingBoxMax - wmoGeometry.BoundingBoxMin;
            var sizeRatio = Math.Min(pm4Size.Length(), wmoSize.Length()) / 
                           Math.Max(pm4Size.Length(), wmoSize.Length());
            similarities.Add(sizeRatio);

            // Surface area similarity (if calculated)
            if (pm4Building.SurfaceArea > 0 && wmoGeometry.SurfaceArea > 0)
            {
                var areaRatio = Math.Min(pm4Building.SurfaceArea, wmoGeometry.SurfaceArea) / 
                               Math.Max(pm4Building.SurfaceArea, wmoGeometry.SurfaceArea);
                similarities.Add(areaRatio);
            }

            return similarities.Average();
        }

        /// <summary>
        /// Calculate geometry properties (bounding box, center, surface area)
        /// </summary>
        private void CalculateGeometryProperties(WmoGeometry geometry)
        {
            if (!geometry.Vertices.Any()) return;

            // Calculate bounding box
            geometry.BoundingBoxMin = new Vector3(
                geometry.Vertices.Min(v => v.X),
                geometry.Vertices.Min(v => v.Y),
                geometry.Vertices.Min(v => v.Z)
            );

            geometry.BoundingBoxMax = new Vector3(
                geometry.Vertices.Max(v => v.X),
                geometry.Vertices.Max(v => v.Y),
                geometry.Vertices.Max(v => v.Z)
            );

            // Calculate center point
            geometry.CenterPoint = (geometry.BoundingBoxMin + geometry.BoundingBoxMax) / 2.0f;

            // Calculate approximate surface area from triangles
            geometry.SurfaceArea = CalculateSurfaceArea(geometry.Vertices, geometry.Indices);
        }

        /// <summary>
        /// Calculate surface area from triangle mesh
        /// </summary>
        private float CalculateSurfaceArea(List<Vector3> vertices, List<uint> indices)
        {
            float totalArea = 0.0f;

            for (int i = 0; i < indices.Count; i += 3)
            {
                if (i + 2 < indices.Count)
                {
                    var v1 = vertices[(int)indices[i]];
                    var v2 = vertices[(int)indices[i + 1]];
                    var v3 = vertices[(int)indices[i + 2]];

                    // Calculate triangle area using cross product
                    var edge1 = v2 - v1;
                    var edge2 = v3 - v1;
                    var cross = Vector3.Cross(edge1, edge2);
                    totalArea += cross.Length() * 0.5f;
                }
            }

            return totalArea;
        }

        /// <summary>
        /// Transform MSCN coordinates to world coordinates
        /// </summary>
        public static Vector3 TransformMscnToWorld(Vector3 mscnVertex)
        {
            return mscnVertex * MSCN_SCALE_FACTOR;
        }

        /// <summary>
        /// Create PM4 building data from spatial clustering results
        /// </summary>
        public static Pm4Building CreatePm4Building(string buildingId, uint surfaceKey, 
            int triangleCount, int vertexCount, List<Vector3> mscnVertices)
        {
            var building = new Pm4Building
            {
                BuildingId = buildingId,
                SurfaceKey = surfaceKey,
                TriangleCount = triangleCount,
                VertexCount = vertexCount,
                MscnVertices = mscnVertices.Select(TransformMscnToWorld).ToList()
            };

            // Calculate bounding box and center from MSCN vertices
            if (building.MscnVertices.Any())
            {
                building.BoundingBoxMin = new Vector3(
                    building.MscnVertices.Min(v => v.X),
                    building.MscnVertices.Min(v => v.Y),
                    building.MscnVertices.Min(v => v.Z)
                );

                building.BoundingBoxMax = new Vector3(
                    building.MscnVertices.Max(v => v.X),
                    building.MscnVertices.Max(v => v.Y),
                    building.MscnVertices.Max(v => v.Z)
                );

                building.CenterPoint = (building.BoundingBoxMin + building.BoundingBoxMax) / 2.0f;
            }

            return building;
        }
    }
}