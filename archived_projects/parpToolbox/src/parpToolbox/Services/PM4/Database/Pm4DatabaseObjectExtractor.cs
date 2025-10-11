using Microsoft.EntityFrameworkCore;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;
using System.Numerics;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Extracts coherent building objects from PM4 database using SQL queries and relational analysis.
    /// This leverages the database-first approach to handle complex hierarchical relationships.
    /// </summary>
    public class Pm4DatabaseObjectExtractor
    {
        private readonly string _databasePath;
        
        public Pm4DatabaseObjectExtractor(string databasePath)
        {
            _databasePath = databasePath;
        }
        
        /// <summary>
        /// Extracts building objects from the database using multiple analysis strategies.
        /// </summary>
        public async Task<List<ExtractedBuilding>> ExtractBuildingsAsync(int pm4FileId)
        {
            using var context = new Pm4DatabaseContext(_databasePath);
            
            ConsoleLogger.WriteLine($"Extracting buildings from PM4 file ID: {pm4FileId}");
            
            var buildings = new List<ExtractedBuilding>();
            
            // Strategy 1: Spatial clustering of surfaces
            var spatialBuildings = await ExtractBuildingsFromSpatialGroupsAsync(context, pm4FileId);
            buildings.AddRange(spatialBuildings);
            
            // Strategy 2: Hierarchical analysis using MPRL->MSLK relationships
            var hierarchicalBuildings = await ExtractBuildingsFromHierarchyAsync(context, pm4FileId);
            buildings.AddRange(hierarchicalBuildings);
            
            // Strategy 3: Hybrid approach combining both methods
            var hybridBuildings = await ExtractBuildingsFromHybridAnalysisAsync(context, pm4FileId);
            buildings.AddRange(hybridBuildings);
            
            ConsoleLogger.WriteLine($"Extracted {buildings.Count} total buildings using multiple strategies");
            
            return buildings;
        }
        
        /// <summary>
        /// Extracts buildings using spatial clustering of surfaces.
        /// </summary>
        private async Task<List<ExtractedBuilding>> ExtractBuildingsFromSpatialGroupsAsync(Pm4DatabaseContext context, int pm4FileId)
        {
            ConsoleLogger.WriteLine("Extracting buildings using spatial clustering...");
            
            var spatialGroups = await context.SurfaceGroups
                .Where(g => g.Pm4FileId == pm4FileId && g.ClusteringMethod == "Spatial")
                .Include(g => g.Members)
                    .ThenInclude(m => m.Surface)
                .ToListAsync();
            
            var buildings = new List<ExtractedBuilding>();
            
            foreach (var group in spatialGroups)
            {
                var building = await CreateBuildingFromSurfaceGroupAsync(context, group, "Spatial");
                if (building.VertexCount > 0)
                {
                    buildings.Add(building);
                }
            }
            
            ConsoleLogger.WriteLine($"Spatial clustering produced {buildings.Count} buildings");
            return buildings;
        }
        
        /// <summary>
        /// Extracts buildings using hierarchical MPRL->MSLK relationships.
        /// </summary>
        private async Task<List<ExtractedBuilding>> ExtractBuildingsFromHierarchyAsync(Pm4DatabaseContext context, int pm4FileId)
        {
            ConsoleLogger.WriteLine("Extracting buildings using hierarchical analysis...");
            
            // Get hierarchical object relationships
            var hierarchicalObjects = await context.GetHierarchicalObjectsAsync(pm4FileId);
            
            var buildings = new List<ExtractedBuilding>();
            int buildingIndex = 0;
            
            foreach (var (placementId, links) in hierarchicalObjects)
            {
                var building = await CreateBuildingFromHierarchicalLinksAsync(context, pm4FileId, placementId, links, buildingIndex++);
                if (building.VertexCount > 0)
                {
                    buildings.Add(building);
                }
            }
            
            ConsoleLogger.WriteLine($"Hierarchical analysis produced {buildings.Count} buildings");
            return buildings;
        }
        
        /// <summary>
        /// Extracts buildings using hybrid spatial + hierarchical analysis.
        /// </summary>
        private async Task<List<ExtractedBuilding>> ExtractBuildingsFromHybridAnalysisAsync(Pm4DatabaseContext context, int pm4FileId)
        {
            ConsoleLogger.WriteLine("Extracting buildings using hybrid analysis...");
            
            // Use SQL to find spatially coherent hierarchical objects
            var hybridQuery = @"
                SELECT 
                    sg.Id as SurfaceGroupId,
                    sg.GroupName,
                    sg.BoundsCenterX, sg.BoundsCenterY, sg.BoundsCenterZ,
                    COUNT(DISTINCT p.Id) as PlacementCount,
                    COUNT(DISTINCT l.Id) as LinkCount,
                    AVG(p.PositionX) as AvgPlacementX,
                    AVG(p.PositionY) as AvgPlacementY,
                    AVG(p.PositionZ) as AvgPlacementZ
                FROM Pm4SurfaceGroups sg
                JOIN Pm4SurfaceGroupMembers sgm ON sg.Id = sgm.SurfaceGroupId
                JOIN Pm4Surfaces s ON sgm.SurfaceId = s.Id
                LEFT JOIN Pm4Placements p ON ABS(p.PositionX - sg.BoundsCenterX) < 100 
                                          AND ABS(p.PositionY - sg.BoundsCenterY) < 100 
                                          AND ABS(p.PositionZ - sg.BoundsCenterZ) < 100
                LEFT JOIN Pm4Links l ON l.ParentIndex = p.Unknown4
                WHERE sg.Pm4FileId = {0}
                  AND sg.ClusteringMethod = 'Spatial'
                GROUP BY sg.Id, sg.GroupName, sg.BoundsCenterX, sg.BoundsCenterY, sg.BoundsCenterZ
                HAVING PlacementCount > 0 OR LinkCount > 0
                ORDER BY sg.SurfaceCount DESC";
            
            var hybridResults = await context.Database.SqlQueryRaw<HybridAnalysisResult>(hybridQuery, pm4FileId).ToListAsync();
            
            var buildings = new List<ExtractedBuilding>();
            
            foreach (var result in hybridResults)
            {
                var building = await CreateBuildingFromHybridAnalysisAsync(context, pm4FileId, result);
                if (building.VertexCount > 0)
                {
                    buildings.Add(building);
                }
            }
            
            ConsoleLogger.WriteLine($"Hybrid analysis produced {buildings.Count} buildings");
            return buildings;
        }
        
        /// <summary>
        /// Creates a building from a surface group.
        /// </summary>
        private async Task<ExtractedBuilding> CreateBuildingFromSurfaceGroupAsync(Pm4DatabaseContext context, Pm4SurfaceGroup group, string method)
        {
            var building = new ExtractedBuilding
            {
                Name = $"{method}_Building_{group.GroupIndex:D3}",
                ExtractionMethod = method,
                BoundsCenter = new Vector3(group.BoundsCenterX, group.BoundsCenterY, group.BoundsCenterZ),
                BoundsMin = new Vector3(group.BoundsMinX, group.BoundsMinY, group.BoundsMinZ),
                BoundsMax = new Vector3(group.BoundsMaxX, group.BoundsMaxY, group.BoundsMaxZ)
            };
            
            // Get all vertices for this surface group
            var vertexQuery = @"
                SELECT v.Id, v.GlobalIndex, v.X, v.Y, v.Z
                FROM Vertices v
                JOIN Surfaces s ON s.Pm4FileId = v.Pm4FileId
                JOIN SurfaceGroupMembers sgm ON sgm.SurfaceGroupId = {0}
                WHERE v.Pm4FileId = {1}
                ORDER BY v.GlobalIndex";
            
            var vertices = await context.Database.SqlQueryRaw<VertexResult>(vertexQuery, group.Id, group.Pm4FileId).ToListAsync();
            
            // Get all triangles for this group
            var triangleQuery = @"
                SELECT t.Id, t.GlobalIndex, t.VertexA, t.VertexB, t.VertexC
                FROM Triangles t
                WHERE t.Pm4FileId = {0}
                  AND EXISTS (
                      SELECT 1 FROM SurfaceGroupMembers sgm 
                      WHERE sgm.SurfaceGroupId = {1}
                  )
                ORDER BY t.GlobalIndex";
            
            var triangles = await context.Database.SqlQueryRaw<TriangleResult>(triangleQuery, group.Pm4FileId, group.Id).ToListAsync();
            
            building.Vertices = vertices.Select(v => new Vector3(v.X, v.Y, v.Z)).ToList();
            building.Triangles = triangles.Select(t => (t.VertexA, t.VertexB, t.VertexC)).ToList();
            
            ConsoleLogger.WriteLine($"Created {method} building '{building.Name}': {building.VertexCount} vertices, {building.TriangleCount} triangles");
            
            return building;
        }
        
        /// <summary>
        /// Creates a building from hierarchical MSLK links.
        /// </summary>
        private async Task<ExtractedBuilding> CreateBuildingFromHierarchicalLinksAsync(Pm4DatabaseContext context, int pm4FileId, uint placementId, List<Pm4Link> links, int buildingIndex)
        {
            var building = new ExtractedBuilding
            {
                Name = $"Hierarchical_Building_{buildingIndex:D3}_Placement_{placementId}",
                ExtractionMethod = "Hierarchical",
                PlacementId = placementId
            };
            
            // Get vertices referenced by these MSLK links through MSPI indices
            var vertexIndices = new HashSet<int>();
            
            foreach (var link in links)
            {
                if (link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
                {
                    // Get vertex indices from MSPI range
                    var indicesQuery = @"
                        SELECT GlobalIndex 
                        FROM Pm4Vertices 
                        WHERE Pm4FileId = {0} 
                          AND GlobalIndex >= {1} 
                          AND GlobalIndex < {2}";
                    
                    var indices = await context.Database.SqlQueryRaw<int>(
                        indicesQuery, 
                        pm4FileId, 
                        link.MspiFirstIndex, 
                        link.MspiFirstIndex + link.MspiIndexCount
                    ).ToListAsync();
                    
                    foreach (var index in indices)
                    {
                        vertexIndices.Add(index);
                    }
                }
            }
            
            if (vertexIndices.Count > 0)
            {
                // Get actual vertices
                var vertices = await context.Vertices
                    .Where(v => v.Pm4FileId == pm4FileId && vertexIndices.Contains(v.GlobalIndex))
                    .OrderBy(v => v.GlobalIndex)
                    .ToListAsync();
                
                building.Vertices = vertices.Select(v => new Vector3(v.X, v.Y, v.Z)).ToList();
                
                // Create triangles from vertex indices (triangle fan approach)
                var triangles = new List<(int, int, int)>();
                var vertexList = vertices.Select(v => v.GlobalIndex).ToList();
                
                for (int i = 1; i < vertexList.Count - 1; i++)
                {
                    triangles.Add((vertexList[0], vertexList[i], vertexList[i + 1]));
                }
                
                building.Triangles = triangles;
                
                // Calculate bounds
                if (building.Vertices.Count > 0)
                {
                    building.BoundsMin = new Vector3(
                        building.Vertices.Min(v => v.X),
                        building.Vertices.Min(v => v.Y),
                        building.Vertices.Min(v => v.Z)
                    );
                    
                    building.BoundsMax = new Vector3(
                        building.Vertices.Max(v => v.X),
                        building.Vertices.Max(v => v.Y),
                        building.Vertices.Max(v => v.Z)
                    );
                    
                    building.BoundsCenter = (building.BoundsMin + building.BoundsMax) * 0.5f;
                }
            }
            
            ConsoleLogger.WriteLine($"Created hierarchical building '{building.Name}': {building.VertexCount} vertices, {building.TriangleCount} triangles");
            
            return building;
        }
        
        /// <summary>
        /// Creates a building from hybrid analysis results.
        /// </summary>
        private async Task<ExtractedBuilding> CreateBuildingFromHybridAnalysisAsync(Pm4DatabaseContext context, int pm4FileId, HybridAnalysisResult result)
        {
            var building = new ExtractedBuilding
            {
                Name = $"Hybrid_{result.GroupName}",
                ExtractionMethod = "Hybrid",
                BoundsCenter = new Vector3(result.BoundsCenterX, result.BoundsCenterY, result.BoundsCenterZ)
            };
            
            // Combine spatial and hierarchical data
            var surfaceGroup = await context.SurfaceGroups
                .Include(g => g.Members)
                    .ThenInclude(m => m.Surface)
                .FirstOrDefaultAsync(g => g.Id == result.SurfaceGroupId);
            
            if (surfaceGroup != null)
            {
                building = await CreateBuildingFromSurfaceGroupAsync(context, surfaceGroup, "Hybrid");
                building.Name = $"Hybrid_{result.GroupName}";
                building.ExtractionMethod = "Hybrid";
            }
            
            return building;
        }
        
        /// <summary>
        /// Exports extracted buildings to OBJ files.
        /// </summary>
        public async Task ExportBuildingsToOBJAsync(List<ExtractedBuilding> buildings, string outputDirectory)
        {
            Directory.CreateDirectory(outputDirectory);
            
            ConsoleLogger.WriteLine($"Exporting {buildings.Count} buildings to OBJ files...");
            
            foreach (var building in buildings)
            {
                if (building.VertexCount == 0) continue;
                
                var objFileName = $"{building.Name}_Vertices{building.VertexCount}_Faces{building.TriangleCount}.obj";
                var objPath = Path.Combine(outputDirectory, objFileName);
                
                await ExportBuildingToOBJAsync(building, objPath);
                
                ConsoleLogger.WriteLine($"Exported '{building.Name}': {building.VertexCount} vertices, {building.TriangleCount} faces");
            }
            
            ConsoleLogger.WriteLine($"Completed OBJ export to {outputDirectory}");
        }
        
        /// <summary>
        /// Exports a single building to an OBJ file.
        /// </summary>
        private async Task ExportBuildingToOBJAsync(ExtractedBuilding building, string outputPath)
        {
            using var writer = new StreamWriter(outputPath);
            
            // Write header
            await writer.WriteLineAsync($"# {building.Name}");
            await writer.WriteLineAsync($"# Extraction Method: {building.ExtractionMethod}");
            await writer.WriteLineAsync($"# Vertices: {building.VertexCount}, Triangles: {building.TriangleCount}");
            await writer.WriteLineAsync($"# Generated: {DateTime.Now}");
            await writer.WriteLineAsync();
            
            // Write vertices
            foreach (var vertex in building.Vertices)
            {
                await writer.WriteLineAsync($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
            
            await writer.WriteLineAsync();
            
            // Write faces (1-indexed)
            foreach (var triangle in building.Triangles)
            {
                await writer.WriteLineAsync($"f {triangle.Item1 + 1} {triangle.Item2 + 1} {triangle.Item3 + 1}");
            }
        }
    }
    
    #region Result Classes
    
    /// <summary>
    /// Represents an extracted building with all its geometric data.
    /// </summary>
    public class ExtractedBuilding
    {
        public string Name { get; set; } = string.Empty;
        public string ExtractionMethod { get; set; } = string.Empty;
        public uint? PlacementId { get; set; }
        
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<(int, int, int)> Triangles { get; set; } = new List<(int, int, int)>();
        
        public Vector3 BoundsMin { get; set; }
        public Vector3 BoundsMax { get; set; }
        public Vector3 BoundsCenter { get; set; }
        
        public int VertexCount => Vertices.Count;
        public int TriangleCount => Triangles.Count;
    }
    
    /// <summary>
    /// Result class for hybrid analysis SQL queries.
    /// </summary>
    public class HybridAnalysisResult
    {
        public int SurfaceGroupId { get; set; }
        public string GroupName { get; set; } = string.Empty;
        public float BoundsCenterX { get; set; }
        public float BoundsCenterY { get; set; }
        public float BoundsCenterZ { get; set; }
        public int PlacementCount { get; set; }
        public int LinkCount { get; set; }
        public float AvgPlacementX { get; set; }
        public float AvgPlacementY { get; set; }
        public float AvgPlacementZ { get; set; }
    }
    
    /// <summary>
    /// Result class for vertex SQL queries.
    /// </summary>
    public class VertexResult
    {
        public int Id { get; set; }
        public int GlobalIndex { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }
    
    /// <summary>
    /// Result class for triangle SQL queries.
    /// </summary>
    public class TriangleResult
    {
        public int Id { get; set; }
        public int GlobalIndex { get; set; }
        public int VertexA { get; set; }
        public int VertexB { get; set; }
        public int VertexC { get; set; }
    }
    
    #endregion
}
