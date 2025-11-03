using System.Text.Json;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Fast JSON-based hierarchical container exporter for large PM4 datasets.
    /// Bypasses SQLite performance bottlenecks by exporting directly to JSON.
    /// </summary>
    public class Pm4HierarchicalJsonExporter
    {
        public class HierarchicalContainer
        {
            public float ContainerX { get; set; }
            public float ContainerY { get; set; }
            public float ContainerZ { get; set; }
            public int SurfaceCount { get; set; }
            public int TotalTriangles { get; set; }
            public int RelatedLinkCount { get; set; }
            public int RelatedPlacementCount { get; set; }
            public int CompletenessScore { get; set; }
            public string ObjectType { get; set; } = "";
            public bool IsCompleteObject { get; set; }
            public List<HierarchicalMember> Members { get; set; } = new();
            
            // Actual spatial bounds (not encoded)
            public float BoundsMinX { get; set; }
            public float BoundsMinY { get; set; }
            public float BoundsMinZ { get; set; }
            public float BoundsMaxX { get; set; }
            public float BoundsMaxY { get; set; }
            public float BoundsMaxZ { get; set; }
        }
        
        public class HierarchicalMember
        {
            public string MemberType { get; set; } = ""; // "Surface", "Link", "Placement"
            public int MemberId { get; set; }
            public int TriangleContribution { get; set; }
            
            // Additional member details for analysis
            public Dictionary<string, object> Properties { get; set; } = new();
        }
        
        public class HierarchicalExportResult
        {
            public string SourceFile { get; set; } = "";
            public DateTime ExportTime { get; set; }
            public int TotalContainers { get; set; }
            public int CompleteObjects { get; set; }
            public int LargeObjects { get; set; }
            public double AverageTrianglesPerContainer { get; set; }
            public List<HierarchicalContainer> Containers { get; set; } = new();
        }
        
        /// <summary>
        /// Fast hierarchical container export that processes data in-memory without database bottlenecks.
        /// </summary>
        public async Task<HierarchicalExportResult> ExportAsync(
            string sourceFileName,
            List<Pm4Surface> surfaces,
            List<Pm4Link> links,
            List<Pm4Placement> placements,
            List<Pm4Vertex> vertices,
            string outputPath)
        {
            ConsoleLogger.WriteLine($"[JSON HIERARCHICAL EXPORT] Processing {surfaces.Count} surfaces, {links.Count} links, {placements.Count} placements");
            
            // Group surfaces by hierarchical container identifiers (BoundsCenterX/Y/Z)
            var containerGroups = surfaces
                .GroupBy(s => new { 
                    ContainerX = s.BoundsCenterX, 
                    ContainerY = s.BoundsCenterY, 
                    ContainerZ = s.BoundsCenterZ 
                })
                .ToList();
            
            ConsoleLogger.WriteLine($"[JSON HIERARCHICAL EXPORT] Found {containerGroups.Count} unique container combinations");
            
            var containers = new List<HierarchicalContainer>();
            var processedCount = 0;
            
            foreach (var group in containerGroups)
            {
                var groupSurfaces = group.ToList();
                
                // Find related links (fast in-memory lookup)
                var relatedLinks = links.Where(l => 
                    groupSurfaces.Any(s => 
                        l.ParentIndex == (uint)s.BoundsCenterX ||
                        l.ParentIndex == (uint)s.BoundsCenterY ||
                        l.ParentIndex == (uint)s.BoundsCenterZ ||
                        l.MspiFirstIndex == s.MsviFirstIndex
                    )
                ).ToList();
                
                // Find related placements (fast in-memory lookup)
                var relatedPlacements = placements.Where(p => 
                    groupSurfaces.Any(s => 
                        p.Unknown4 == (uint)s.BoundsCenterX ||
                        p.Unknown4 == (uint)s.BoundsCenterY ||
                        p.Unknown4 == (uint)s.BoundsCenterZ ||
                        p.Unknown6 == (uint)s.BoundsCenterX ||
                        p.Unknown6 == (uint)s.BoundsCenterY ||
                        p.Unknown6 == (uint)s.BoundsCenterZ
                    )
                ).ToList();
                
                // Calculate object metrics
                var totalTriangles = groupSurfaces.Sum(s => s.IndexCount);
                var completenessScore = (groupSurfaces.Count > 0 ? 1 : 0) + 
                                       (relatedLinks.Count > 0 ? 1 : 0) + 
                                       (relatedPlacements.Count > 0 ? 1 : 0);
                
                // Calculate actual spatial bounds from vertices (fast range lookup)
                var minIndex = groupSurfaces.Min(s => s.MsviFirstIndex);
                var maxIndex = groupSurfaces.Max(s => s.MsviFirstIndex + s.IndexCount);
                
                var relevantVertices = vertices.Where(v => 
                    v.GlobalIndex >= minIndex && v.GlobalIndex < maxIndex &&
                    groupSurfaces.Any(s => v.GlobalIndex >= s.MsviFirstIndex && 
                                          v.GlobalIndex < s.MsviFirstIndex + s.IndexCount)
                ).ToList();
                
                var actualBounds = CalculateActualSpatialBounds(relevantVertices);
                
                // Classify object type and completeness
                var objectType = ClassifyObjectType(totalTriangles, completenessScore);
                var isCompleteObject = IsCompleteObject(totalTriangles, completenessScore);
                
                // Create hierarchical container
                var container = new HierarchicalContainer
                {
                    ContainerX = group.Key.ContainerX,
                    ContainerY = group.Key.ContainerY,
                    ContainerZ = group.Key.ContainerZ,
                    SurfaceCount = groupSurfaces.Count,
                    TotalTriangles = totalTriangles,
                    RelatedLinkCount = relatedLinks.Count,
                    RelatedPlacementCount = relatedPlacements.Count,
                    CompletenessScore = completenessScore,
                    ObjectType = objectType,
                    IsCompleteObject = isCompleteObject,
                    BoundsMinX = actualBounds.min.X,
                    BoundsMinY = actualBounds.min.Y,
                    BoundsMinZ = actualBounds.min.Z,
                    BoundsMaxX = actualBounds.max.X,
                    BoundsMaxY = actualBounds.max.Y,
                    BoundsMaxZ = actualBounds.max.Z
                };
                
                // Add members with details
                foreach (var surface in groupSurfaces)
                {
                    container.Members.Add(new HierarchicalMember
                    {
                        MemberType = "Surface",
                        MemberId = surface.Id,
                        TriangleContribution = surface.IndexCount,
                        Properties = new Dictionary<string, object>
                        {
                            ["GroupKey"] = surface.GroupKey,
                            ["MsviFirstIndex"] = surface.MsviFirstIndex,
                            ["IndexCount"] = surface.IndexCount,
                            ["BoundsCenterX"] = surface.BoundsCenterX,
                            ["BoundsCenterY"] = surface.BoundsCenterY,
                            ["BoundsCenterZ"] = surface.BoundsCenterZ
                        }
                    });
                }
                
                foreach (var link in relatedLinks)
                {
                    container.Members.Add(new HierarchicalMember
                    {
                        MemberType = "Link",
                        MemberId = link.Id,
                        TriangleContribution = 0,
                        Properties = new Dictionary<string, object>
                        {
                            ["ParentIndex"] = link.ParentIndex,
                            ["MspiFirstIndex"] = link.MspiFirstIndex
                        }
                    });
                }
                
                foreach (var placement in relatedPlacements)
                {
                    container.Members.Add(new HierarchicalMember
                    {
                        MemberType = "Placement",
                        MemberId = placement.Id,
                        TriangleContribution = 0,
                        Properties = new Dictionary<string, object>
                        {
                            ["Unknown4"] = placement.Unknown4,
                            ["Unknown6"] = placement.Unknown6,
                            ["X"] = placement.PositionX,
                            ["Y"] = placement.PositionY,
                            ["Z"] = placement.PositionZ
                        }
                    });
                }
                
                containers.Add(container);
                processedCount++;
                
                // Progress reporting every 10,000 containers
                if (processedCount % 10000 == 0)
                {
                    ConsoleLogger.WriteLine($"[JSON HIERARCHICAL EXPORT] Processed {processedCount}/{containerGroups.Count} containers...");
                }
            }
            
            // Create export result
            var result = new HierarchicalExportResult
            {
                SourceFile = sourceFileName,
                ExportTime = DateTime.UtcNow,
                TotalContainers = containers.Count,
                CompleteObjects = containers.Count(c => c.IsCompleteObject),
                LargeObjects = containers.Count(c => c.TotalTriangles > 10000),
                AverageTrianglesPerContainer = containers.Any() ? containers.Average(c => c.TotalTriangles) : 0,
                Containers = containers
            };
            
            // Export to JSON with optimized settings
            var jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            
            var jsonContent = JsonSerializer.Serialize(result, jsonOptions);
            await File.WriteAllTextAsync(outputPath, jsonContent);
            
            // Log summary statistics
            ConsoleLogger.WriteLine($"[JSON HIERARCHICAL EXPORT] Export complete:");
            ConsoleLogger.WriteLine($"  Total containers: {result.TotalContainers}");
            ConsoleLogger.WriteLine($"  Complete objects: {result.CompleteObjects}");
            ConsoleLogger.WriteLine($"  Large objects (>10K triangles): {result.LargeObjects}");
            ConsoleLogger.WriteLine($"  Average triangles per container: {result.AverageTrianglesPerContainer:F1}");
            ConsoleLogger.WriteLine($"  JSON file: {outputPath}");
            
            return result;
        }
        
        private (System.Numerics.Vector3 min, System.Numerics.Vector3 max) CalculateActualSpatialBounds(List<Pm4Vertex> vertices)
        {
            if (!vertices.Any())
            {
                return (System.Numerics.Vector3.Zero, System.Numerics.Vector3.Zero);
            }
            
            var min = new System.Numerics.Vector3(
                vertices.Min(v => v.X),
                vertices.Min(v => v.Y),
                vertices.Min(v => v.Z)
            );
            
            var max = new System.Numerics.Vector3(
                vertices.Max(v => v.X),
                vertices.Max(v => v.Y),
                vertices.Max(v => v.Z)
            );
            
            return (min, max);
        }
        
        private string ClassifyObjectType(int triangleCount, int completenessScore)
        {
            if (completenessScore >= 3 && triangleCount >= 38000)
                return "building";
            else if (completenessScore >= 2 && triangleCount >= 10000)
                return "structure";
            else if (completenessScore >= 2 && triangleCount >= 1000)
                return "object";
            else if (triangleCount >= 100)
                return "detail";
            else
                return "fragment";
        }
        
        private bool IsCompleteObject(int triangleCount, int completenessScore)
        {
            return completenessScore >= 2 && triangleCount >= 1000;
        }
    }
}
