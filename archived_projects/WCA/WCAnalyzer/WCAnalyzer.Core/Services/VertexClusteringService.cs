using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for clustering vertices in 3D mesh data based on spatial proximity.
    /// </summary>
    public class VertexClusteringService
    {
        private readonly ILogger<VertexClusteringService>? _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="VertexClusteringService"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        public VertexClusteringService(ILogger<VertexClusteringService>? logger = null)
        {
            _logger = logger;
        }
        
        /// <summary>
        /// Clusters vertices based on spatial proximity using K-means clustering.
        /// </summary>
        /// <param name="vertices">The vertices to cluster</param>
        /// <param name="clusterCount">The target number of clusters. If null, will be estimated.</param>
        /// <param name="maxIterations">Maximum number of iterations for K-means</param>
        /// <returns>A dictionary mapping each vertex index to its cluster ID</returns>
        public Dictionary<int, int> ClusterVertices(List<Vector3> vertices, int? clusterCount = null, int maxIterations = 100)
        {
            if (vertices == null || vertices.Count == 0)
            {
                _logger?.LogWarning("No vertices provided for clustering");
                return new Dictionary<int, int>();
            }
            
            // Convert vertices to matrix format for math operations
            var points = Matrix<double>.Build.Dense(vertices.Count, 3);
            for (int i = 0; i < vertices.Count; i++)
            {
                points[i, 0] = vertices[i].X;
                points[i, 1] = vertices[i].Y;
                points[i, 2] = vertices[i].Z;
            }
            
            // Determine optimal cluster count if not specified
            int k = clusterCount ?? EstimateOptimalClusterCount(vertices);
            k = Math.Min(k, vertices.Count); // Can't have more clusters than points
            
            _logger?.LogInformation("Clustering {Count} vertices into {ClusterCount} clusters", vertices.Count, k);
            
            // Perform K-means clustering
            var clusters = KMeans(points, k, maxIterations);
            
            // Map vertex indices to cluster IDs
            var vertexClusterMap = new Dictionary<int, int>();
            for (int i = 0; i < vertices.Count; i++)
            {
                vertexClusterMap[i] = clusters[i];
            }
            
            return vertexClusterMap;
        }
        
        /// <summary>
        /// Groups triangle faces by the clusters their vertices belong to.
        /// </summary>
        /// <param name="indices">The triangle indices (every 3 indices form a triangle)</param>
        /// <param name="vertexClusterMap">Mapping of vertex indices to cluster IDs</param>
        /// <returns>A dictionary mapping cluster IDs to lists of triangle indices</returns>
        public Dictionary<int, List<uint>> GroupTrianglesByCluster(List<uint> indices, Dictionary<int, int> vertexClusterMap)
        {
            if (indices == null || indices.Count == 0 || vertexClusterMap == null || vertexClusterMap.Count == 0)
            {
                _logger?.LogWarning("No indices or vertex cluster mapping provided for triangle grouping");
                return new Dictionary<int, List<uint>>();
            }
            
            var trianglesByCluster = new Dictionary<int, List<uint>>();
            int triangleCount = indices.Count / 3;
            
            for (int i = 0; i < triangleCount; i++)
            {
                int baseIndex = i * 3;
                uint v1 = indices[baseIndex];
                uint v2 = indices[baseIndex + 1];
                uint v3 = indices[baseIndex + 2];
                
                // Determine the dominant cluster for this triangle
                // Strategy: use the most common cluster among the 3 vertices
                var clusterCounts = new Dictionary<int, int>();
                
                if (vertexClusterMap.TryGetValue((int)v1, out int c1))
                    clusterCounts[c1] = clusterCounts.GetValueOrDefault(c1) + 1;
                
                if (vertexClusterMap.TryGetValue((int)v2, out int c2))
                    clusterCounts[c2] = clusterCounts.GetValueOrDefault(c2) + 1;
                
                if (vertexClusterMap.TryGetValue((int)v3, out int c3))
                    clusterCounts[c3] = clusterCounts.GetValueOrDefault(c3) + 1;
                
                int dominantCluster = clusterCounts.OrderByDescending(kv => kv.Value).First().Key;
                
                // Add this triangle to the appropriate cluster
                if (!trianglesByCluster.ContainsKey(dominantCluster))
                    trianglesByCluster[dominantCluster] = new List<uint>();
                
                trianglesByCluster[dominantCluster].Add(v1);
                trianglesByCluster[dominantCluster].Add(v2);
                trianglesByCluster[dominantCluster].Add(v3);
            }
            
            _logger?.LogInformation("Grouped {TriangleCount} triangles into {ClusterCount} clusters", 
                triangleCount, trianglesByCluster.Count);
            
            return trianglesByCluster;
        }
        
        /// <summary>
        /// Implements K-means clustering algorithm on 3D points.
        /// </summary>
        /// <param name="points">Matrix of points (rows=points, columns=dimensions)</param>
        /// <param name="k">Number of clusters</param>
        /// <param name="maxIterations">Maximum number of iterations</param>
        /// <returns>Array of cluster assignments for each point</returns>
        private int[] KMeans(Matrix<double> points, int k, int maxIterations)
        {
            int n = points.RowCount;
            int dim = points.ColumnCount;
            
            // Initialize centroids with k-means++ method
            var centroids = InitializeCentroids(points, k);
            var clusters = new int[n];
            var oldClusters = new int[n];
            
            bool converged = false;
            int iteration = 0;
            
            while (!converged && iteration < maxIterations)
            {
                // Assign points to nearest centroid
                for (int i = 0; i < n; i++)
                {
                    var point = points.Row(i);
                    double minDistance = double.MaxValue;
                    int closestCentroid = 0;
                    
                    for (int j = 0; j < k; j++)
                    {
                        var centroid = centroids.Row(j);
                        double distance = Distance(point, centroid);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            closestCentroid = j;
                        }
                    }
                    
                    clusters[i] = closestCentroid;
                }
                
                // Check for convergence
                converged = true;
                for (int i = 0; i < n; i++)
                {
                    if (clusters[i] != oldClusters[i])
                    {
                        converged = false;
                        break;
                    }
                }
                
                if (converged)
                    break;
                
                // Copy current clusters to old clusters
                Array.Copy(clusters, oldClusters, n);
                
                // Update centroids
                var newCentroids = Matrix<double>.Build.Dense(k, dim);
                var counts = new int[k];
                
                for (int i = 0; i < n; i++)
                {
                    int cluster = clusters[i];
                    counts[cluster]++;
                    
                    for (int j = 0; j < dim; j++)
                    {
                        newCentroids[cluster, j] += points[i, j];
                    }
                }
                
                for (int i = 0; i < k; i++)
                {
                    if (counts[i] > 0)
                    {
                        for (int j = 0; j < dim; j++)
                        {
                            newCentroids[i, j] /= counts[i];
                        }
                    }
                    else
                    {
                        // Handle empty cluster by assigning a random point
                        int randomIndex = new Random().Next(0, n);
                        for (int j = 0; j < dim; j++)
                        {
                            newCentroids[i, j] = points[randomIndex, j];
                        }
                    }
                }
                
                centroids = newCentroids;
                iteration++;
            }
            
            _logger?.LogDebug("K-means clustering completed in {Iterations} iterations", iteration);
            
            return clusters;
        }
        
        /// <summary>
        /// Initializes centroids using the k-means++ method for better initial placement.
        /// </summary>
        /// <param name="points">Matrix of points</param>
        /// <param name="k">Number of clusters</param>
        /// <returns>Matrix of initial centroids</returns>
        private Matrix<double> InitializeCentroids(Matrix<double> points, int k)
        {
            int n = points.RowCount;
            int dim = points.ColumnCount;
            var random = new Random();
            var centroids = Matrix<double>.Build.Dense(k, dim);
            
            // Choose first centroid randomly
            int firstIndex = random.Next(0, n);
            var firstCentroid = points.Row(firstIndex);
            for (int j = 0; j < dim; j++)
            {
                centroids[0, j] = firstCentroid[j];
            }
            
            // Choose remaining centroids
            for (int i = 1; i < k; i++)
            {
                var distances = new double[n];
                
                // Calculate minimum distance from each point to any existing centroid
                for (int j = 0; j < n; j++)
                {
                    var point = points.Row(j);
                    double minDistance = double.MaxValue;
                    
                    for (int c = 0; c < i; c++)
                    {
                        var centroid = centroids.Row(c);
                        double distance = Distance(point, centroid);
                        minDistance = Math.Min(minDistance, distance);
                    }
                    
                    distances[j] = minDistance;
                }
                
                // Choose next centroid with probability proportional to squared distance
                double totalWeight = distances.Sum();
                double threshold = random.NextDouble() * totalWeight;
                double cumulativeWeight = 0;
                int nextCentroidIndex = 0;
                
                for (int j = 0; j < n; j++)
                {
                    cumulativeWeight += distances[j];
                    if (cumulativeWeight >= threshold)
                    {
                        nextCentroidIndex = j;
                        break;
                    }
                }
                
                var nextCentroid = points.Row(nextCentroidIndex);
                for (int j = 0; j < dim; j++)
                {
                    centroids[i, j] = nextCentroid[j];
                }
            }
            
            return centroids;
        }
        
        /// <summary>
        /// Calculates the squared Euclidean distance between two vectors.
        /// </summary>
        /// <param name="a">First vector</param>
        /// <param name="b">Second vector</param>
        /// <returns>The squared Euclidean distance</returns>
        private double Distance(MathNet.Numerics.LinearAlgebra.Vector<double> a, MathNet.Numerics.LinearAlgebra.Vector<double> b)
        {
            return (a - b).PointwiseMultiply(a - b).Sum();
        }
        
        /// <summary>
        /// Estimates the optimal number of clusters for the given vertices.
        /// </summary>
        /// <param name="vertices">The vertices to analyze</param>
        /// <returns>Estimated optimal number of clusters</returns>
        public int EstimateOptimalClusterCount(List<Vector3> vertices)
        {
            if (vertices == null || vertices.Count == 0)
                return 1;

            // Use square root heuristic as a simple estimation
            int k = (int)Math.Sqrt(vertices.Count / 2);
            return Math.Max(1, Math.Min(k, vertices.Count));
        }
    }
} 