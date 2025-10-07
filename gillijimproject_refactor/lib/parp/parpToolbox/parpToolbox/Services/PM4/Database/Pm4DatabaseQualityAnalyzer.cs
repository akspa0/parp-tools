using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Analyzes PM4 database for data quality issues and parsing errors.
    /// </summary>
    public class Pm4DatabaseQualityAnalyzer
    {
        private readonly string _databasePath;

        public Pm4DatabaseQualityAnalyzer(string databasePath)
        {
            _databasePath = databasePath;
        }

        /// <summary>
        /// Performs comprehensive data quality analysis on the PM4 database.
        /// </summary>
        public async Task<QualityAnalysisReport> AnalyzeDataQualityAsync()
        {
            using var context = new Pm4DatabaseContext(_databasePath);
            
            ConsoleLogger.WriteLine("[QUALITY] Starting comprehensive data quality analysis...");
            
            var report = new QualityAnalysisReport();
            
            // Analyze coordinate data quality
            await AnalyzeCoordinateQualityAsync(context, report);
            
            // Analyze placement data (MPRL)
            await AnalyzePlacementDataAsync(context, report);
            
            // Analyze links data (MSLK)
            await AnalyzeLinksDataAsync(context, report);
            
            // Analyze field completeness
            await AnalyzeFieldCompletenessAsync(context, report);
            
            // Analyze raw chunk coverage
            await AnalyzeRawChunkCoverageAsync(context, report);
            
            ConsoleLogger.WriteLine("[QUALITY] Data quality analysis complete.");
            return report;
        }

        private async Task AnalyzeCoordinateQualityAsync(Pm4DatabaseContext context, QualityAnalysisReport report)
        {
            ConsoleLogger.WriteLine("[QUALITY] Analyzing coordinate data quality...");
            
            // Analyze surfaces coordinate data
            var surfaces = await context.Surfaces.ToListAsync();
            if (surfaces.Any())
            {
                var coords = new List<float>();
                
                foreach (var surface in surfaces)
                {
                    coords.AddRange(new[] { surface.BoundsMinX, surface.BoundsMinY, surface.BoundsMinZ });
                    coords.AddRange(new[] { surface.BoundsMaxX, surface.BoundsMaxY, surface.BoundsMaxZ });
                    coords.AddRange(new[] { surface.BoundsCenterX, surface.BoundsCenterY, surface.BoundsCenterZ });
                }
                
                var coordStats = CalculateCoordinateStats(coords);
                report.SurfaceCoordinateStats = coordStats;
                
                // Identify outliers (likely non-coordinate data)
                var outliers = coords.Where(c => Math.Abs(c) > 100000 || float.IsInfinity(c) || float.IsNaN(c)).ToList();
                report.SurfaceOutliers = outliers;
                
                ConsoleLogger.WriteLine($"[QUALITY] Surfaces: {coords.Count} coordinate values, {outliers.Count} outliers detected");
            }
            
            // Analyze vertices coordinate data
            var vertices = await context.Vertices.Take(1000).ToListAsync(); // Sample for performance
            if (vertices.Any())
            {
                var vertexCoords = vertices.SelectMany(v => new[] { v.X, v.Y, v.Z }).ToList();
                var vertexStats = CalculateCoordinateStats(vertexCoords);
                report.VertexCoordinateStats = vertexStats;
                
                ConsoleLogger.WriteLine($"[QUALITY] Vertices: {vertexCoords.Count} coordinate values analyzed (sample)");
            }
        }

        private async Task AnalyzePlacementDataAsync(Pm4DatabaseContext context, QualityAnalysisReport report)
        {
            ConsoleLogger.WriteLine("[QUALITY] Analyzing placement data...");
            
            var placements = await context.Placements.ToListAsync();
            report.TotalPlacements = placements.Count;
            
            if (placements.Any())
            {
                var nonZeroPlacements = placements.Where(p => 
                    p.PositionX != 0 || p.PositionY != 0 || p.PositionZ != 0).Count();
                    
                report.NonZeroPlacements = nonZeroPlacements;
                report.ZeroPlacementPercentage = (placements.Count - nonZeroPlacements) * 100.0 / placements.Count;
                
                ConsoleLogger.WriteLine($"[QUALITY] Placements: {placements.Count} total, {nonZeroPlacements} non-zero ({report.ZeroPlacementPercentage:F1}% are zero)");
            }
        }

        private async Task AnalyzeLinksDataAsync(Pm4DatabaseContext context, QualityAnalysisReport report)
        {
            ConsoleLogger.WriteLine("[QUALITY] Analyzing links data...");
            
            var links = await context.Links.ToListAsync();
            report.TotalLinks = links.Count;
            
            if (links.Any())
            {
                // Check for valid parent/reference relationships
                var validParentRefs = links.Where(l => 
                    !string.IsNullOrEmpty(l.RawFieldsJson) && l.RawFieldsJson != "{}").Count();
                    
                report.LinksWithValidData = validParentRefs;
                
                ConsoleLogger.WriteLine($"[QUALITY] Links: {links.Count} total, {validParentRefs} with valid field data");
            }
        }

        private async Task AnalyzeFieldCompletenessAsync(Pm4DatabaseContext context, QualityAnalysisReport report)
        {
            ConsoleLogger.WriteLine("[QUALITY] Analyzing field completeness...");
            
            // Check for empty/default JSON fields
            var emptyLinkFields = await context.Links.CountAsync(l => l.RawFieldsJson == "{}");
            var emptyPlacementFields = await context.Placements.CountAsync(p => p.RawFieldsJson == "{}");
            
            report.EmptyLinkFields = emptyLinkFields;
            report.EmptyPlacementFields = emptyPlacementFields;
            
            ConsoleLogger.WriteLine($"[QUALITY] Empty fields: {emptyLinkFields} links, {emptyPlacementFields} placements");
        }

        private async Task AnalyzeRawChunkCoverageAsync(Pm4DatabaseContext context, QualityAnalysisReport report)
        {
            ConsoleLogger.WriteLine("[QUALITY] Analyzing raw chunk coverage...");
            
            var rawChunks = await context.RawChunks.ToListAsync();
            report.TotalRawChunks = rawChunks.Count;
            
            if (rawChunks.Any())
            {
                var chunkTypes = rawChunks.GroupBy(c => c.ChunkType)
                    .ToDictionary(g => g.Key, g => g.Count());
                    
                report.ChunkTypeCounts = chunkTypes;
                report.TotalRawDataSize = rawChunks.Sum(c => c.ChunkSize);
                
                ConsoleLogger.WriteLine($"[QUALITY] Raw chunks: {rawChunks.Count} total, {report.TotalRawDataSize:N0} bytes");
                foreach (var (type, count) in chunkTypes)
                {
                    ConsoleLogger.WriteLine($"[QUALITY]   {type}: {count} chunks");
                }
            }
        }

        private CoordinateStats CalculateCoordinateStats(List<float> coordinates)
        {
            if (!coordinates.Any()) return new CoordinateStats();
            
            var validCoords = coordinates.Where(c => !float.IsInfinity(c) && !float.IsNaN(c)).ToList();
            if (!validCoords.Any()) return new CoordinateStats();
            
            return new CoordinateStats
            {
                Count = coordinates.Count,
                ValidCount = validCoords.Count,
                Min = validCoords.Min(),
                Max = validCoords.Max(),
                Average = validCoords.Average(),
                StandardDeviation = Math.Sqrt(validCoords.Average(c => Math.Pow(c - validCoords.Average(), 2)))
            };
        }
    }

    public class QualityAnalysisReport
    {
        public CoordinateStats? SurfaceCoordinateStats { get; set; }
        public CoordinateStats? VertexCoordinateStats { get; set; }
        public List<float> SurfaceOutliers { get; set; } = new();
        
        public int TotalPlacements { get; set; }
        public int NonZeroPlacements { get; set; }
        public double ZeroPlacementPercentage { get; set; }
        
        public int TotalLinks { get; set; }
        public int LinksWithValidData { get; set; }
        
        public int EmptyLinkFields { get; set; }
        public int EmptyPlacementFields { get; set; }
        
        public int TotalRawChunks { get; set; }
        public long TotalRawDataSize { get; set; }
        public Dictionary<string, int> ChunkTypeCounts { get; set; } = new();
    }

    public class CoordinateStats
    {
        public int Count { get; set; }
        public int ValidCount { get; set; }
        public float Min { get; set; }
        public float Max { get; set; }
        public double Average { get; set; }
        public double StandardDeviation { get; set; }
    }
}
