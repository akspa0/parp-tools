using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Analyzes spatial relationships between MSLK geometry components and creates intelligent clusters
    /// </summary>
    public class MslkSpatialClusteringAnalyzer
    {
        public class SpatialComponent
        {
            public int NodeIndex { get; set; }
            public Vector3 Centroid { get; set; }
            public Vector3 BoundingBoxMin { get; set; }
            public Vector3 BoundingBoxMax { get; set; }
            public float Volume { get; set; }
            public int VertexCount { get; set; }
            public int TriangleCount { get; set; }
            public MslkHierarchyAnalyzer.ObjectSegmentationResult SegmentResult { get; set; } = null!;
        }

        public class SpatialCluster
        {
            public int ClusterId { get; set; }
            public string ClusterName { get; set; } = "";
            public List<SpatialComponent> Components { get; set; } = new();
            public Vector3 ClusterCentroid { get; set; }
            public Vector3 ClusterBoundsMin { get; set; }
            public Vector3 ClusterBoundsMax { get; set; }
            public ClusterType Type { get; set; }
            public float Density { get; set; }
            public float TotalVolume { get; set; }
            public string AssemblyFileName { get; set; } = "";
        }

        public enum ClusterType
        {
            MajorStructure,     // Large, important architectural elements
            DetailCluster,      // Small details grouped together
            LinearStructure,    // Wall-like or linear arrangements
            CompactCluster,     // Tightly grouped components
            Standalone,         // Isolated individual components
            Unknown
        }

        public class ClusteringResult
        {
            public List<SpatialCluster> Clusters { get; set; } = new();
            public List<SpatialComponent> UnclusteredComponents { get; set; } = new();
            public ClusteringParameters Parameters { get; set; } = null!;
            public ClusteringStats Stats { get; set; } = new();
        }

        public class ClusteringParameters
        {
            public float MaxDistance { get; set; } = 100.0f;           // Maximum distance for clustering
            public int MinComponentsPerCluster { get; set; } = 2;       // Minimum components to form cluster
            public float MaxComponentSize { get; set; } = 1000.0f;     // Max size for detail clustering
            public float MinComponentSize { get; set; } = 1.0f;        // Min size threshold
            public float DensityThreshold { get; set; } = 0.5f;        // Density for cluster classification
            public bool PreserveHierarchy { get; set; } = true;        // Respect PM4 hierarchy relationships
        }

        public class ClusteringStats
        {
            public int TotalComponents { get; set; }
            public int TotalClusters { get; set; }
            public int LargestClusterSize { get; set; }
            public float AverageClusterSize { get; set; }
            public float ClusteringEfficiency { get; set; }  // Reduction ratio
            public Dictionary<ClusterType, int> ClustersByType { get; set; } = new();
        }

        /// <summary>
        /// Analyze individual geometry components and extract spatial properties
        /// </summary>
        public List<SpatialComponent> AnalyzeSpatialComponents(
            List<MslkHierarchyAnalyzer.ObjectSegmentationResult> geometryObjects,
            PM4File pm4File)
        {
            var spatialComponents = new List<SpatialComponent>();

            Console.WriteLine($"🔍 Analyzing spatial properties of {geometryObjects.Count} geometry components...");

            foreach (var obj in geometryObjects)
            {
                try
                {
                    var spatial = ExtractSpatialProperties(obj, pm4File);
                    if (spatial != null)
                    {
                        spatialComponents.Add(spatial);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ⚠️  Failed to analyze component {obj.RootIndex}: {ex.Message}");
                }
            }

            Console.WriteLine($"✅ Extracted spatial data for {spatialComponents.Count} components");
            return spatialComponents;
        }

        /// <summary>
        /// Perform spatial clustering using distance-based algorithm
        /// </summary>
        public ClusteringResult PerformSpatialClustering(
            List<SpatialComponent> components, 
            ClusteringParameters parameters = null)
        {
            parameters ??= new ClusteringParameters();
            
            Console.WriteLine($"🎯 Starting spatial clustering with max distance: {parameters.MaxDistance}");
            
            var result = new ClusteringResult
            {
                Parameters = parameters
            };

            // Sort components by size to handle large structures first
            var sortedComponents = components
                .OrderByDescending(c => c.Volume)
                .ToList();

            var unprocessed = new HashSet<SpatialComponent>(sortedComponents);
            var clusterId = 0;

            // Distance-based clustering
            while (unprocessed.Any())
            {
                var seed = unprocessed.First();
                var cluster = CreateClusterFromSeed(seed, unprocessed, parameters);
                
                if (cluster.Components.Count >= parameters.MinComponentsPerCluster)
                {
                    cluster.ClusterId = clusterId++;
                    cluster.ClusterName = GenerateClusterName(cluster);
                    ClassifyCluster(cluster);
                    CalculateClusterBounds(cluster);
                    result.Clusters.Add(cluster);
                    
                    // Remove clustered components
                    foreach (var component in cluster.Components)
                    {
                        unprocessed.Remove(component);
                    }
                    
                    Console.WriteLine($"  ✨ Created {cluster.Type} cluster '{cluster.ClusterName}' with {cluster.Components.Count} components");
                }
                else
                {
                    // Handle as standalone or add to unclustered
                    if (seed.Volume > parameters.MaxComponentSize / 2) // Large enough to be standalone
                    {
                        var standaloneCluster = new SpatialCluster
                        {
                            ClusterId = clusterId++,
                            ClusterName = $"Standalone_{seed.NodeIndex:D3}",
                            Components = new List<SpatialComponent> { seed },
                            Type = ClusterType.Standalone
                        };
                        CalculateClusterBounds(standaloneCluster);
                        result.Clusters.Add(standaloneCluster);
                        Console.WriteLine($"  🏗️  Standalone structure: {standaloneCluster.ClusterName}");
                    }
                    else
                    {
                        result.UnclusteredComponents.Add(seed);
                    }
                    
                    unprocessed.Remove(seed);
                }
            }

            // Calculate statistics
            CalculateClusteringStats(result);

            Console.WriteLine($"📊 Clustering complete: {result.Clusters.Count} clusters, {result.UnclusteredComponents.Count} unclustered");
            return result;
        }

        private SpatialComponent ExtractSpatialProperties(MslkHierarchyAnalyzer.ObjectSegmentationResult obj, PM4File pm4File)
        {
            // Extract geometry data from PM4 to calculate spatial properties
            // This is a simplified implementation - we'll need actual mesh data for precise calculations
            
            if (!obj.GeometryNodeIndices.Any())
                return null;

            var primaryNodeIndex = obj.GeometryNodeIndices.First();
            
            // Get MSLK entry for this geometry node
            if (primaryNodeIndex >= pm4File.MSLK.Entries.Count)
                return null;
                
            var mslkEntry = pm4File.MSLK.Entries[primaryNodeIndex];
            
            // Estimate spatial properties (placeholder - real implementation would analyze mesh data)
            var estimatedSize = EstimateComponentSize(mslkEntry);
            var estimatedCentroid = EstimateComponentCentroid(mslkEntry, primaryNodeIndex);
            
            return new SpatialComponent
            {
                NodeIndex = primaryNodeIndex,
                Centroid = estimatedCentroid,
                BoundingBoxMin = estimatedCentroid - new Vector3(estimatedSize / 2),
                BoundingBoxMax = estimatedCentroid + new Vector3(estimatedSize / 2),
                Volume = estimatedSize * estimatedSize * estimatedSize,
                VertexCount = mslkEntry.MspiIndexCount * 3, // Rough estimate
                TriangleCount = mslkEntry.MspiIndexCount,
                SegmentResult = obj
            };
        }

        private float EstimateComponentSize(MSLKEntry entry)
        {
            // Estimate size based on MSLK data patterns
            // This is a heuristic - real implementation would use actual mesh bounds
            
            var baseSize = Math.Max(1.0f, entry.MspiIndexCount / 10.0f);
            return Math.Min(baseSize, 500.0f); // Cap at reasonable maximum
        }

        private Vector3 EstimateComponentCentroid(MSLKEntry entry, int nodeIndex)
        {
            // Estimate position based on node patterns and data
            // This is a placeholder - real implementation would use actual vertex data
            
            var x = (nodeIndex % 100) * 10.0f; // Spread components spatially
            var y = ((nodeIndex / 100) % 100) * 10.0f;
            var z = entry.Unknown_0x0C % 100; // Use some MSLK data for Z variation
            
            return new Vector3(x, y, z);
        }

        private SpatialCluster CreateClusterFromSeed(
            SpatialComponent seed, 
            HashSet<SpatialComponent> available, 
            ClusteringParameters parameters)
        {
            var cluster = new SpatialCluster
            {
                Components = new List<SpatialComponent> { seed }
            };

            // Find all components within clustering distance
            var toProcess = new Queue<SpatialComponent>();
            toProcess.Enqueue(seed);
            var processed = new HashSet<SpatialComponent> { seed };

            while (toProcess.Any())
            {
                var current = toProcess.Dequeue();
                
                var nearbyComponents = available
                    .Where(c => !processed.Contains(c))
                    .Where(c => Vector3.Distance(current.Centroid, c.Centroid) <= parameters.MaxDistance)
                    .Where(c => IsCompatibleForClustering(current, c, parameters))
                    .ToList();

                foreach (var nearby in nearbyComponents)
                {
                    cluster.Components.Add(nearby);
                    processed.Add(nearby);
                    toProcess.Enqueue(nearby);
                }
            }

            return cluster;
        }

        private bool IsCompatibleForClustering(SpatialComponent a, SpatialComponent b, ClusteringParameters parameters)
        {
            // Check size compatibility - don't cluster very large with very small
            var sizeRatio = Math.Max(a.Volume, b.Volume) / Math.Max(Math.Min(a.Volume, b.Volume), 0.1f);
            if (sizeRatio > 100.0f) // Too different in size
                return false;

            // Additional compatibility checks can be added here
            return true;
        }

        private void ClassifyCluster(SpatialCluster cluster)
        {
            var totalVolume = cluster.Components.Sum(c => c.Volume);
            var componentCount = cluster.Components.Count;
            var avgVolume = totalVolume / componentCount;

            // Calculate spatial arrangement
            var boundsSize = cluster.ClusterBoundsMax - cluster.ClusterBoundsMin;
            var longestDimension = Math.Max(Math.Max(boundsSize.X, boundsSize.Y), boundsSize.Z);
            var shortestDimension = Math.Min(Math.Min(boundsSize.X, boundsSize.Y), boundsSize.Z);
            var aspectRatio = longestDimension / Math.Max(shortestDimension, 0.1f);

            // Classify based on properties
            if (avgVolume > 1000.0f || componentCount <= 2)
            {
                cluster.Type = ClusterType.MajorStructure;
            }
            else if (aspectRatio > 5.0f) // Long and thin
            {
                cluster.Type = ClusterType.LinearStructure;
            }
            else if (componentCount > 10 && avgVolume < 100.0f)
            {
                cluster.Type = ClusterType.DetailCluster;
            }
            else if (componentCount >= 3 && componentCount <= 8)
            {
                cluster.Type = ClusterType.CompactCluster;
            }
            else
            {
                cluster.Type = ClusterType.Unknown;
            }
        }

        private void CalculateClusterBounds(SpatialCluster cluster)
        {
            if (!cluster.Components.Any())
                return;

            var minX = cluster.Components.Min(c => c.BoundingBoxMin.X);
            var minY = cluster.Components.Min(c => c.BoundingBoxMin.Y);
            var minZ = cluster.Components.Min(c => c.BoundingBoxMin.Z);
            
            var maxX = cluster.Components.Max(c => c.BoundingBoxMax.X);
            var maxY = cluster.Components.Max(c => c.BoundingBoxMax.Y);
            var maxZ = cluster.Components.Max(c => c.BoundingBoxMax.Z);

            cluster.ClusterBoundsMin = new Vector3(minX, minY, minZ);
            cluster.ClusterBoundsMax = new Vector3(maxX, maxY, maxZ);
            
            cluster.ClusterCentroid = (cluster.ClusterBoundsMin + cluster.ClusterBoundsMax) / 2.0f;
            cluster.TotalVolume = cluster.Components.Sum(c => c.Volume);
            
            var boundsVolume = (maxX - minX) * (maxY - minY) * (maxZ - minZ);
            cluster.Density = boundsVolume > 0 ? cluster.TotalVolume / boundsVolume : 0;

            cluster.AssemblyFileName = $"{cluster.Type}_{cluster.ClusterId:D3}.gltf";
        }

        private string GenerateClusterName(SpatialCluster cluster)
        {
            var type = cluster.Type.ToString();
            var size = cluster.Components.Count;
            return $"{type}_{size}comp_{cluster.ClusterId:D3}";
        }

        private void CalculateClusteringStats(ClusteringResult result)
        {
            result.Stats.TotalComponents = result.Clusters.Sum(c => c.Components.Count) + result.UnclusteredComponents.Count;
            result.Stats.TotalClusters = result.Clusters.Count;
            result.Stats.LargestClusterSize = result.Clusters.Any() ? result.Clusters.Max(c => c.Components.Count) : 0;
            result.Stats.AverageClusterSize = result.Clusters.Any() ? (float)result.Clusters.Average(c => c.Components.Count) : 0;
            result.Stats.ClusteringEfficiency = result.Stats.TotalComponents > 0 
                ? 1.0f - ((float)result.Stats.TotalClusters / result.Stats.TotalComponents) 
                : 0;

            result.Stats.ClustersByType = result.Clusters
                .GroupBy(c => c.Type)
                .ToDictionary(g => g.Key, g => g.Count());
        }

        /// <summary>
        /// Generate detailed clustering report
        /// </summary>
        public string GenerateClusteringReport(ClusteringResult result, string pm4FileName)
        {
            var report = new System.Text.StringBuilder();
            
            report.AppendLine("🎯 SPATIAL CLUSTERING ANALYSIS REPORT");
            report.AppendLine("═══════════════════════════════════════════════════════════════");
            report.AppendLine($"PM4 File: {pm4FileName}");
            report.AppendLine($"Generated: {DateTime.Now}");
            report.AppendLine();

            report.AppendLine("📊 CLUSTERING STATISTICS:");
            report.AppendLine($"  Total Components: {result.Stats.TotalComponents}");
            report.AppendLine($"  Total Clusters: {result.Stats.TotalClusters}");
            report.AppendLine($"  Largest Cluster: {result.Stats.LargestClusterSize} components");
            report.AppendLine($"  Average Cluster Size: {result.Stats.AverageClusterSize:F1} components");
            report.AppendLine($"  Clustering Efficiency: {result.Stats.ClusteringEfficiency:P1}");
            report.AppendLine($"  Unclustered Components: {result.UnclusteredComponents.Count}");
            report.AppendLine();

            report.AppendLine("🏗️  CLUSTER BREAKDOWN BY TYPE:");
            foreach (var typeGroup in result.Stats.ClustersByType)
            {
                report.AppendLine($"  {typeGroup.Key}: {typeGroup.Value} clusters");
            }
            report.AppendLine();

            report.AppendLine("📋 CLUSTER DETAILS:");
            foreach (var cluster in result.Clusters.OrderByDescending(c => c.Components.Count))
            {
                report.AppendLine($"  🔸 {cluster.ClusterName} ({cluster.Type})");
                report.AppendLine($"     Components: {cluster.Components.Count}");
                report.AppendLine($"     Total Volume: {cluster.TotalVolume:F1}");
                report.AppendLine($"     Density: {cluster.Density:F3}");
                report.AppendLine($"     Output File: {cluster.AssemblyFileName}");
                report.AppendLine($"     Node Indices: {string.Join(", ", cluster.Components.Select(c => c.NodeIndex).OrderBy(x => x))}");
                report.AppendLine();
            }

            return report.ToString();
        }
    }
} 