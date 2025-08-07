using Microsoft.EntityFrameworkCore;
using ParpToolbox.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Decodes Surface "bounds" fields as linkage containers rather than spatial bounds.
    /// Tests the hypothesis that bounds fields encode cross-tile references and metadata.
    /// </summary>
    public class Pm4SurfaceBoundsDecoder
    {
        private readonly string _databasePath;
        private readonly string _outputDirectory;
        
        public Pm4SurfaceBoundsDecoder(string databasePath, string outputDirectory)
        {
            _databasePath = databasePath ?? throw new ArgumentNullException(nameof(databasePath));
            _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));
            
            if (!Directory.Exists(_outputDirectory))
            {
                Directory.CreateDirectory(_outputDirectory);
            }
        }
        
        /// <summary>
        /// Main analysis entry point for bounds decoding validation.
        /// </summary>
        public async Task AnalyzeAsync()
        {
            ConsoleLogger.WriteLine("[BOUNDS DECODER] Starting Surface bounds decoding analysis...");

            using var context = new Pm4DatabaseContext(_databasePath);

            var surfaces = await context.Surfaces.ToListAsync();

            if (!surfaces.Any())
            {
                ConsoleLogger.WriteLine("No surfaces found in database.");
                return;
            }

            ConsoleLogger.WriteLine($"[BOUNDS DECODER] Analyzing {surfaces.Count} surface records...");

            // Test 1: Validate field overloading hypothesis
            await ValidateFieldOverloading(surfaces);
            
            // Test 2: Decode tile/chunk reference IDs
            await DecodeTileReferences(surfaces);
            
            // Test 3: Analyze cross-tile vertex indices
            await AnalyzeCrossTileVertices(surfaces);
            
            // Test 4: Interpret direction vectors
            await AnalyzeDirectionVectors(surfaces);
            
            // Test 5: Test linkage-based organization
            await TestLinkageBasedOrganization(surfaces);

            ConsoleLogger.WriteLine($"[BOUNDS DECODER] Analysis complete. Reports saved to: {_outputDirectory}");
        }

        /// <summary>
        /// Test 1: Validate the field overloading hypothesis
        /// BoundsCenterX = MsviFirstIndex, BoundsCenterY = IndexCount, BoundsCenterZ = GroupKey
        /// </summary>
        private async Task ValidateFieldOverloading(List<Pm4Surface> surfaces)
        {
            ConsoleLogger.WriteLine("[BOUNDS DECODER] TEST 1: Validating field overloading hypothesis...");
            
            var overloadingValidation = surfaces.Select(s => new
            {
                Id = s.Id,
                MsviFirstIndex = s.MsviFirstIndex,
                BoundsCenterX = s.BoundsCenterX,
                IndexCount = s.IndexCount,
                BoundsCenterY = s.BoundsCenterY,
                GroupKey = s.GroupKey,
                BoundsCenterZ = s.BoundsCenterZ,
                CenterXMatch = Math.Abs(s.BoundsCenterX - s.MsviFirstIndex) < 0.001f,
                CenterYMatch = Math.Abs(s.BoundsCenterY - s.IndexCount) < 0.001f,
                CenterZMatch = Math.Abs(s.BoundsCenterZ - s.GroupKey) < 0.001f
            }).ToList();

            var centerXMatches = overloadingValidation.Count(v => v.CenterXMatch);
            var centerYMatches = overloadingValidation.Count(v => v.CenterYMatch);
            var centerZMatches = overloadingValidation.Count(v => v.CenterZMatch);

            ConsoleLogger.WriteLine($"  BoundsCenterX = MsviFirstIndex: {centerXMatches}/{surfaces.Count} matches ({centerXMatches * 100.0 / surfaces.Count:F1}%)");
            ConsoleLogger.WriteLine($"  BoundsCenterY = IndexCount: {centerYMatches}/{surfaces.Count} matches ({centerYMatches * 100.0 / surfaces.Count:F1}%)");
            ConsoleLogger.WriteLine($"  BoundsCenterZ = GroupKey: {centerZMatches}/{surfaces.Count} matches ({centerZMatches * 100.0 / surfaces.Count:F1}%)");

            // Save validation results
            var csvPath = Path.Combine(_outputDirectory, "field_overloading_validation.csv");
            var csvContent = "Id,MsviFirstIndex,BoundsCenterX,CenterXMatch,IndexCount,BoundsCenterY,CenterYMatch,GroupKey,BoundsCenterZ,CenterZMatch\n" +
                string.Join("\n", overloadingValidation.Select(v => 
                    $"{v.Id},{v.MsviFirstIndex},{v.BoundsCenterX},{v.CenterXMatch},{v.IndexCount},{v.BoundsCenterY},{v.CenterYMatch},{v.GroupKey},{v.BoundsCenterZ},{v.CenterZMatch}"));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Validation results saved to: {csvPath}");
        }

        /// <summary>
        /// Test 2: Decode BoundsMaxZ as tile/chunk reference IDs
        /// </summary>
        private async Task DecodeTileReferences(List<Pm4Surface> surfaces)
        {
            ConsoleLogger.WriteLine("[BOUNDS DECODER] TEST 2: Decoding tile/chunk reference IDs...");

            var tileRefs = surfaces
                .GroupBy(s => s.BoundsMaxZ)
                .Select(g => new
                {
                    TileRefId = g.Key,
                    Count = g.Count(),
                    HexValue = $"0x{(uint)g.Key:X8}",
                    // Potential coordinate decoding (various interpretations)
                    AsBytes = new byte[] { 
                        (byte)((uint)g.Key & 0xFF), 
                        (byte)(((uint)g.Key >> 8) & 0xFF), 
                        (byte)(((uint)g.Key >> 16) & 0xFF), 
                        (byte)(((uint)g.Key >> 24) & 0xFF) 
                    },
                    AsCoords = $"({(uint)g.Key & 0xFF}, {((uint)g.Key >> 8) & 0xFF}, {((uint)g.Key >> 16) & 0xFF}, {((uint)g.Key >> 24) & 0xFF})"
                })
                .OrderByDescending(x => x.Count)
                .ToList();

            ConsoleLogger.WriteLine($"  Found {tileRefs.Count} unique tile/chunk reference IDs:");
            foreach (var tileRef in tileRefs.Take(10))
            {
                ConsoleLogger.WriteLine($"    {tileRef.TileRefId} ({tileRef.HexValue}): {tileRef.Count} surfaces, coords: {tileRef.AsCoords}");
            }

            // Save tile reference analysis
            var csvPath = Path.Combine(_outputDirectory, "tile_references_analysis.csv");
            var csvContent = "TileRefId,HexValue,Count,Percentage,PotentialCoords\n" +
                string.Join("\n", tileRefs.Select(t => 
                    $"{t.TileRefId},{t.HexValue},{t.Count},{t.Count * 100.0 / surfaces.Count:F2},{t.AsCoords}"));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Tile reference analysis saved to: {csvPath}");
        }

        /// <summary>
        /// Test 3: Analyze BoundsMaxX/Y as cross-tile vertex indices
        /// </summary>
        private async Task AnalyzeCrossTileVertices(List<Pm4Surface> surfaces)
        {
            ConsoleLogger.WriteLine("[BOUNDS DECODER] TEST 3: Analyzing cross-tile vertex indices...");

            var vertexIndices = surfaces.Select(s => new
            {
                Id = s.Id,
                BoundsMaxX = s.BoundsMaxX,
                BoundsMaxY = s.BoundsMaxY,
                BoundsMaxZ = s.BoundsMaxZ,
                GroupKey = s.GroupKey,
                // Potential vertex index interpretations
                MaxXAsInt = (int)s.BoundsMaxX,
                MaxYAsInt = (int)s.BoundsMaxY,
                MaxXAsUInt = (uint)s.BoundsMaxX,
                MaxYAsUInt = (uint)s.BoundsMaxY
            }).ToList();

            // Analyze patterns in vertex indices
            var maxXStats = new
            {
                Min = vertexIndices.Min(v => v.MaxXAsInt),
                Max = vertexIndices.Max(v => v.MaxXAsInt),
                Average = vertexIndices.Average(v => v.MaxXAsInt),
                Positive = vertexIndices.Count(v => v.MaxXAsInt > 0),
                Negative = vertexIndices.Count(v => v.MaxXAsInt < 0),
                LargeValues = vertexIndices.Count(v => Math.Abs(v.MaxXAsInt) > 100000)
            };

            var maxYStats = new
            {
                Min = vertexIndices.Min(v => v.MaxYAsInt),
                Max = vertexIndices.Max(v => v.MaxYAsInt),
                Average = vertexIndices.Average(v => v.MaxYAsInt),
                Positive = vertexIndices.Count(v => v.MaxYAsInt > 0),
                Negative = vertexIndices.Count(v => v.MaxYAsInt < 0),
                LargeValues = vertexIndices.Count(v => Math.Abs(v.MaxYAsInt) > 100000)
            };

            ConsoleLogger.WriteLine($"  BoundsMaxX (as vertex indices):");
            ConsoleLogger.WriteLine($"    Range: {maxXStats.Min} to {maxXStats.Max} (avg: {maxXStats.Average:F1})");
            ConsoleLogger.WriteLine($"    Distribution: {maxXStats.Positive} positive, {maxXStats.Negative} negative, {maxXStats.LargeValues} large values");
            
            ConsoleLogger.WriteLine($"  BoundsMaxY (as vertex indices):");
            ConsoleLogger.WriteLine($"    Range: {maxYStats.Min} to {maxYStats.Max} (avg: {maxYStats.Average:F1})");
            ConsoleLogger.WriteLine($"    Distribution: {maxYStats.Positive} positive, {maxYStats.Negative} negative, {maxYStats.LargeValues} large values");

            // Save vertex index analysis
            var csvPath = Path.Combine(_outputDirectory, "cross_tile_vertex_indices.csv");
            var csvContent = "Id,BoundsMaxX,BoundsMaxY,BoundsMaxZ,GroupKey,MaxXAsInt,MaxYAsInt,MaxXAsUInt,MaxYAsUInt\n" +
                string.Join("\n", vertexIndices.Select(v => 
                    $"{v.Id},{v.BoundsMaxX},{v.BoundsMaxY},{v.BoundsMaxZ},{v.GroupKey},{v.MaxXAsInt},{v.MaxYAsInt},{v.MaxXAsUInt},{v.MaxYAsUInt}"));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Cross-tile vertex analysis saved to: {csvPath}");
        }

        /// <summary>
        /// Test 4: Analyze BoundsMinX/Y/Z as direction vectors or normalized coordinates
        /// </summary>
        private async Task AnalyzeDirectionVectors(List<Pm4Surface> surfaces)
        {
            ConsoleLogger.WriteLine("[BOUNDS DECODER] TEST 4: Analyzing direction vectors/normalized coordinates...");

            var directionVectors = surfaces.Select(s => new
            {
                Id = s.Id,
                MinX = s.BoundsMinX,
                MinY = s.BoundsMinY,
                MinZ = s.BoundsMinZ,
                Magnitude = Math.Sqrt(s.BoundsMinX * s.BoundsMinX + s.BoundsMinY * s.BoundsMinY + s.BoundsMinZ * s.BoundsMinZ),
                IsNormalized = Math.Abs(Math.Sqrt(s.BoundsMinX * s.BoundsMinX + s.BoundsMinY * s.BoundsMinY + s.BoundsMinZ * s.BoundsMinZ) - 1.0) < 0.01,
                GroupKey = s.GroupKey
            }).ToList();

            var normalizedCount = directionVectors.Count(v => v.IsNormalized);
            var avgMagnitude = directionVectors.Average(v => v.Magnitude);
            var magnitudeStats = new
            {
                Min = directionVectors.Min(v => v.Magnitude),
                Max = directionVectors.Max(v => v.Magnitude),
                NearUnit = directionVectors.Count(v => Math.Abs(v.Magnitude - 1.0) < 0.1)
            };

            ConsoleLogger.WriteLine($"  Direction vector analysis:");
            ConsoleLogger.WriteLine($"    Normalized vectors (magnitude ≈ 1): {normalizedCount}/{surfaces.Count} ({normalizedCount * 100.0 / surfaces.Count:F1}%)");
            ConsoleLogger.WriteLine($"    Average magnitude: {avgMagnitude:F6}");
            ConsoleLogger.WriteLine($"    Magnitude range: {magnitudeStats.Min:F6} to {magnitudeStats.Max:F6}");
            ConsoleLogger.WriteLine($"    Near unit vectors (±0.1): {magnitudeStats.NearUnit}");

            // Save direction vector analysis
            var csvPath = Path.Combine(_outputDirectory, "direction_vectors_analysis.csv");
            var csvContent = "Id,MinX,MinY,MinZ,Magnitude,IsNormalized,GroupKey\n" +
                string.Join("\n", directionVectors.Select(v => 
                    $"{v.Id},{v.MinX},{v.MinY},{v.MinZ},{v.Magnitude},{v.IsNormalized},{v.GroupKey}"));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Direction vector analysis saved to: {csvPath}");
        }

        /// <summary>
        /// Test 5: Test linkage-based organization using decoded bounds data
        /// </summary>
        private async Task TestLinkageBasedOrganization(List<Pm4Surface> surfaces)
        {
            ConsoleLogger.WriteLine("[BOUNDS DECODER] TEST 5: Testing linkage-based organization...");

            // Group surfaces by decoded linkage data
            var linkageGroups = surfaces
                .GroupBy(s => new { TileRef = s.BoundsMaxZ, GroupKey = s.GroupKey })
                .Select(g => new
                {
                    TileRefId = g.Key.TileRef,
                    GroupKey = g.Key.GroupKey,
                    SurfaceCount = g.Count(),
                    TotalTriangles = g.Sum(s => s.IndexCount),
                    AvgTriangles = g.Average(s => s.IndexCount),
                    Surfaces = g.ToList()
                })
                .OrderByDescending(g => g.SurfaceCount)
                .ToList();

            ConsoleLogger.WriteLine($"  Found {linkageGroups.Count} linkage-based groups:");
            foreach (var group in linkageGroups.Take(10))
            {
                ConsoleLogger.WriteLine($"    TileRef: {group.TileRefId:X8}, GroupKey: {group.GroupKey}, Surfaces: {group.SurfaceCount}, Triangles: {group.TotalTriangles}");
            }

            // Analyze group size distribution
            var groupSizes = linkageGroups.Select(g => g.SurfaceCount).ToList();
            var sizeStats = new
            {
                Min = groupSizes.Min(),
                Max = groupSizes.Max(),
                Average = groupSizes.Average(),
                SingleSurface = linkageGroups.Count(g => g.SurfaceCount == 1),
                SmallGroups = linkageGroups.Count(g => g.SurfaceCount >= 2 && g.SurfaceCount <= 10),
                LargeGroups = linkageGroups.Count(g => g.SurfaceCount > 10)
            };

            ConsoleLogger.WriteLine($"  Group size distribution:");
            ConsoleLogger.WriteLine($"    Range: {sizeStats.Min} to {sizeStats.Max} surfaces (avg: {sizeStats.Average:F1})");
            ConsoleLogger.WriteLine($"    Single surface: {sizeStats.SingleSurface}, Small groups (2-10): {sizeStats.SmallGroups}, Large groups (>10): {sizeStats.LargeGroups}");

            // Save linkage-based organization analysis
            var csvPath = Path.Combine(_outputDirectory, "linkage_based_organization.csv");
            var csvContent = "TileRefId,TileRefHex,GroupKey,SurfaceCount,TotalTriangles,AvgTriangles\n" +
                string.Join("\n", linkageGroups.Select(g => 
                    $"{g.TileRefId},{g.TileRefId:X8},{g.GroupKey},{g.SurfaceCount},{g.TotalTriangles},{g.AvgTriangles:F2}"));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Linkage-based organization analysis saved to: {csvPath}");
        }
    }
}
