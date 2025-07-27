using Microsoft.EntityFrameworkCore;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;
using System.Numerics;
using System.Text.Json;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Exports PM4 scene data to an embedded SQLite database for complex hierarchical analysis.
    /// This treats PM4 as the specialized database format it actually is.
    /// </summary>
    public class Pm4DatabaseExporter
    {
        private readonly string _databasePath;
        
        public Pm4DatabaseExporter(string databasePath)
        {
            _databasePath = databasePath;
        }
        
        /// <summary>
        /// Exports a PM4 scene to the database with full hierarchical relationships.
        /// </summary>
        public async Task<int> ExportSceneAsync(Pm4Scene scene, string fileName, string filePath)
        {
            using var context = new Pm4DatabaseContext(_databasePath);
            await context.EnsureDatabaseCreatedAsync();
            
            ConsoleLogger.WriteLine($"Exporting PM4 scene '{fileName}' to database...");
            
            // Create PM4 file record
            var pm4File = new Pm4File
            {
                FileName = fileName,
                FilePath = filePath,
                ProcessedAt = DateTime.UtcNow,
                TotalVertices = scene.Vertices.Count,
                TotalTriangles = scene.Triangles.Count,
                TotalSurfaces = scene.Groups.Count,
                TotalLinks = scene.Links.Count,
                TotalPlacements = scene.Placements.Count
            };
            
            context.Files.Add(pm4File);
            await context.SaveChangesAsync();
            
            ConsoleLogger.WriteLine($"Created PM4 file record with ID: {pm4File.Id}");
            
            // Export all data in batches for performance
            await ExportVerticesAsync(context, pm4File.Id, scene.Vertices);
            await ExportTrianglesAsync(context, pm4File.Id, scene.Triangles);
            await ExportSurfacesAsync(context, pm4File.Id, scene.Groups, scene.Vertices);
            await ExportLinksAsync(context, pm4File.Id, scene.Links?.Cast<object>().ToList() ?? new List<object>());
            await ExportPlacementsAsync(context, pm4File.Id, scene.Placements?.Cast<object>().ToList() ?? new List<object>());
            await ExportSurfaceGroupsAsync(context, pm4File.Id, scene.Groups);
            
            ConsoleLogger.WriteLine($"Successfully exported PM4 scene to database. File ID: {pm4File.Id}");
            return pm4File.Id;
        }
        
        /// <summary>
        /// Exports vertices to the database with optimized batch processing for performance.
        /// </summary>
        private async Task ExportVerticesAsync(Pm4DatabaseContext context, int pm4FileId, List<Vector3> vertices)
        {
            ConsoleLogger.WriteLine($"Exporting {vertices.Count} vertices with optimized batching...");
            
            // Use larger batch size for better performance
            const int batchSize = 50000;
            for (int i = 0; i < vertices.Count; i += batchSize)
            {
                var batch = vertices
                    .Skip(i)
                    .Take(batchSize)
                    .Select((vertex, index) => new Pm4Vertex
                    {
                        Pm4FileId = pm4FileId,
                        GlobalIndex = i + index,
                        X = vertex.X,
                        Y = vertex.Y,
                        Z = vertex.Z,
                        ChunkType = "MSVT" // Default to MSVT, could be enhanced to detect MSPV
                    })
                    .ToList();
                
                context.Vertices.AddRange(batch);
                await context.SaveChangesAsync();
                
                if (i % (batchSize * 10) == 0)
                {
                    ConsoleLogger.WriteLine($"  Exported {Math.Min(i + batchSize, vertices.Count)} / {vertices.Count} vertices");
                }
            }
            
            ConsoleLogger.WriteLine($"Completed vertex export: {vertices.Count} vertices");
        }
        
        /// <summary>
        /// Exports triangles to the database with optimized batch processing for performance.
        /// </summary>
        private async Task ExportTrianglesAsync(Pm4DatabaseContext context, int pm4FileId, List<(int A, int B, int C)> triangles)
        {
            ConsoleLogger.WriteLine($"Exporting {triangles.Count} triangles with optimized batching...");
            
            // Use much larger batch size for better performance
            const int batchSize = 50000; // 50x larger batches
            
            // Disable change tracking for bulk inserts
            context.ChangeTracker.AutoDetectChangesEnabled = false;
            
            for (int i = 0; i < triangles.Count; i += batchSize)
            {
                var batch = new List<Pm4Triangle>(batchSize);
                var endIndex = Math.Min(i + batchSize, triangles.Count);
                
                for (int j = i; j < endIndex; j++)
                {
                    var triangle = triangles[j];
                    batch.Add(new Pm4Triangle
                    {
                        Pm4FileId = pm4FileId,
                        GlobalIndex = j,
                        VertexA = triangle.A,
                        VertexB = triangle.B,
                        VertexC = triangle.C
                    });
                }
                
                context.Triangles.AddRange(batch);
                await context.SaveChangesAsync();
                
                // Clear tracked entities to free memory
                context.ChangeTracker.Clear();
                
                ConsoleLogger.WriteLine($"  Exported {endIndex} / {triangles.Count} triangles ({(endIndex * 100.0 / triangles.Count):F1}%)");
            }
            
            // Re-enable change tracking
            context.ChangeTracker.AutoDetectChangesEnabled = true;
            
            ConsoleLogger.WriteLine($"Completed optimized triangle export: {triangles.Count} triangles");
        }
        
        /// <summary>
        /// Exports surfaces to the database with spatial bounds calculation.
        /// </summary>
        private async Task ExportSurfacesAsync(Pm4DatabaseContext context, int pm4FileId, List<SurfaceGroup> groups, List<Vector3> vertices)
        {
            ConsoleLogger.WriteLine($"Exporting {groups.Count} surface groups...");
            
            const int batchSize = 500;
            for (int i = 0; i < groups.Count; i += batchSize)
            {
                var batch = groups
                    .Skip(i)
                    .Take(batchSize)
                    .Select((group, index) =>
                    {
                        var bounds = CalculateSurfaceGroupBounds(group, vertices);
                        
                        return new Pm4Surface
                        {
                            Pm4FileId = pm4FileId,
                            GlobalIndex = i + index,
                            MsviFirstIndex = 0, // Would need to be calculated from actual MSUR data
                            IndexCount = group.Faces.Count * 3,
                            GroupKey = group.GroupKey,
                            RawFlags = group.RawFlags,
                            BoundsMinX = bounds.min.X,
                            BoundsMinY = bounds.min.Y,
                            BoundsMinZ = bounds.min.Z,
                            BoundsMaxX = bounds.max.X,
                            BoundsMaxY = bounds.max.Y,
                            BoundsMaxZ = bounds.max.Z,
                            BoundsCenterX = bounds.center.X,
                            BoundsCenterY = bounds.center.Y,
                            BoundsCenterZ = bounds.center.Z
                        };
                    })
                    .ToList();
                
                context.Surfaces.AddRange(batch);
                await context.SaveChangesAsync();
                
                if (i % (batchSize * 4) == 0)
                {
                    ConsoleLogger.WriteLine($"  Exported {Math.Min(i + batchSize, groups.Count)} / {groups.Count} surface groups");
                }
            }
            
            ConsoleLogger.WriteLine($"Completed surface export: {groups.Count} surface groups");
        }
        
        /// <summary>
        /// Exports MSLK links to the database with reflection-safe field extraction.
        /// </summary>
        private async Task ExportLinksAsync(Pm4DatabaseContext context, int pm4FileId, List<object> links)
        {
            ConsoleLogger.WriteLine($"Exporting {links.Count} MSLK links...");
            
            const int batchSize = 1000;
            for (int i = 0; i < links.Count; i += batchSize)
            {
                var batch = links
                    .Skip(i)
                    .Take(batchSize)
                    .Select((link, index) =>
                    {
                        var fields = ExtractMslkFields(link);
                        
                        return new Pm4Link
                        {
                            Pm4FileId = pm4FileId,
                            GlobalIndex = i + index,
                            ParentIndex = fields.ParentIndex,
                            MspiFirstIndex = fields.MspiFirstIndex,
                            MspiIndexCount = fields.MspiIndexCount,
                            ReferenceIndex = fields.ReferenceIndex,
                            RawFieldsJson = JsonSerializer.Serialize(fields.RawFields)
                        };
                    })
                    .ToList();
                
                context.Links.AddRange(batch);
                await context.SaveChangesAsync();
                
                if (i % (batchSize * 10) == 0)
                {
                    ConsoleLogger.WriteLine($"  Exported {Math.Min(i + batchSize, links.Count)} / {links.Count} MSLK links");
                }
            }
            
            ConsoleLogger.WriteLine($"Completed MSLK link export: {links.Count} links");
        }
        
        /// <summary>
        /// Exports MPRL placements to the database with reflection-safe field extraction.
        /// </summary>
        private async Task ExportPlacementsAsync(Pm4DatabaseContext context, int pm4FileId, List<object> placements)
        {
            ConsoleLogger.WriteLine($"Exporting {placements.Count} MPRL placements...");
            
            const int batchSize = 1000;
            for (int i = 0; i < placements.Count; i += batchSize)
            {
                var batch = placements
                    .Skip(i)
                    .Take(batchSize)
                    .Select((placement, index) =>
                    {
                        var fields = ExtractMprlFields(placement);
                        
                        return new Pm4Placement
                        {
                            Pm4FileId = pm4FileId,
                            GlobalIndex = i + index,
                            PositionX = fields.Position.X,
                            PositionY = fields.Position.Y,
                            PositionZ = fields.Position.Z,
                            Unknown4 = fields.Unknown4,
                            Unknown6 = fields.Unknown6,
                            RawFieldsJson = JsonSerializer.Serialize(fields.RawFields)
                        };
                    })
                    .ToList();
                
                context.Placements.AddRange(batch);
                await context.SaveChangesAsync();
                
                if (i % (batchSize * 10) == 0)
                {
                    ConsoleLogger.WriteLine($"  Exported {Math.Min(i + batchSize, placements.Count)} / {placements.Count} MPRL placements");
                }
            }
            
            ConsoleLogger.WriteLine($"Completed MPRL placement export: {placements.Count} placements");
        }
        
        /// <summary>
        /// Creates initial surface groups using spatial clustering and exports them.
        /// </summary>
        private async Task ExportSurfaceGroupsAsync(Pm4DatabaseContext context, int pm4FileId, List<SurfaceGroup> groups)
        {
            ConsoleLogger.WriteLine($"Creating spatial surface groups from {groups.Count} surfaces...");
            
            // Get all surfaces for this file
            var surfaces = await context.Surfaces
                .Where(s => s.Pm4FileId == pm4FileId)
                .ToListAsync();
            
            // Perform spatial clustering
            var spatialGroups = PerformSpatialClustering(surfaces);
            
            ConsoleLogger.WriteLine($"Created {spatialGroups.Count} spatial groups");
            
            // Export surface groups
            foreach (var (groupIndex, surfaceIds) in spatialGroups.Select((g, i) => (i, g)))
            {
                var groupSurfaces = surfaces.Where(s => surfaceIds.Contains(s.Id)).ToList();
                var bounds = CalculateGroupBounds(groupSurfaces);
                
                var surfaceGroup = new Pm4SurfaceGroup
                {
                    Pm4FileId = pm4FileId,
                    GroupIndex = groupIndex,
                    GroupName = $"SpatialGroup_{groupIndex:D3}",
                    ClusteringMethod = "Spatial",
                    BoundsMinX = bounds.min.X,
                    BoundsMinY = bounds.min.Y,
                    BoundsMinZ = bounds.min.Z,
                    BoundsMaxX = bounds.max.X,
                    BoundsMaxY = bounds.max.Y,
                    BoundsMaxZ = bounds.max.Z,
                    BoundsCenterX = bounds.center.X,
                    BoundsCenterY = bounds.center.Y,
                    BoundsCenterZ = bounds.center.Z,
                    SurfaceCount = surfaceIds.Count,
                    VertexCount = groupSurfaces.Sum(s => s.IndexCount),
                    TriangleCount = groupSurfaces.Sum(s => s.IndexCount / 3)
                };
                
                context.SurfaceGroups.Add(surfaceGroup);
                await context.SaveChangesAsync();
                
                // Add group members
                var members = surfaceIds.Select(surfaceId => new Pm4SurfaceGroupMember
                {
                    SurfaceGroupId = surfaceGroup.Id,
                    SurfaceId = surfaceId
                }).ToList();
                
                context.SurfaceGroupMembers.AddRange(members);
            }
            
            await context.SaveChangesAsync();
            ConsoleLogger.WriteLine($"Completed surface group export: {spatialGroups.Count} groups");
        }
        
        #region Helper Methods
        
        /// <summary>
        /// Calculates spatial bounds for a surface group.
        /// </summary>
        private (Vector3 min, Vector3 max, Vector3 center) CalculateSurfaceGroupBounds(SurfaceGroup group, List<Vector3> vertices)
        {
            if (group.Faces.Count == 0)
                return (Vector3.Zero, Vector3.Zero, Vector3.Zero);
            
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            
            foreach (var face in group.Faces)
            {
                var faceVertices = new[] { face.A, face.B, face.C };
                foreach (var vertexIndex in faceVertices)
                {
                    if (vertexIndex >= 0 && vertexIndex < vertices.Count)
                    {
                        var vertex = vertices[vertexIndex];
                        min = Vector3.Min(min, vertex);
                        max = Vector3.Max(max, vertex);
                    }
                }
            }
            
            var center = (min + max) * 0.5f;
            return (min, max, center);
        }
        
        /// <summary>
        /// Extracts MSLK fields using reflection with error handling.
        /// </summary>
        private (uint ParentIndex, int MspiFirstIndex, int MspiIndexCount, uint ReferenceIndex, Dictionary<string, object> RawFields) ExtractMslkFields(object mslkEntry)
        {
            var rawFields = new Dictionary<string, object>();
            uint parentIndex = 0;
            int mspiFirstIndex = -1;
            int mspiIndexCount = 0;
            uint referenceIndex = 0;
            
            try
            {
                var type = mslkEntry.GetType();
                var fields = type.GetFields();
                
                for (int i = 0; i < fields.Length; i++)
                {
                    var field = fields[i];
                    var value = field.GetValue(mslkEntry);
                    rawFields[field.Name] = value ?? DBNull.Value;
                    
                    // Extract known fields by position (safer than name matching)
                    switch (i)
                    {
                        case 3: // 4th field - ParentIndex
                            if (value is uint ui) parentIndex = ui;
                            else if (value is int ii) parentIndex = (uint)ii;
                            break;
                        case 4: // 5th field - MspiFirstIndex
                            if (value is int mfi) mspiFirstIndex = mfi;
                            break;
                        case 5: // 6th field - MspiIndexCount
                            if (value is int mic) mspiIndexCount = mic;
                            break;
                        case 6: // 7th field - ReferenceIndex
                            if (value is uint ri) referenceIndex = ri;
                            else if (value is int rii) referenceIndex = (uint)rii;
                            break;
                    }
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error extracting MSLK fields: {ex.Message}");
            }
            
            return (parentIndex, mspiFirstIndex, mspiIndexCount, referenceIndex, rawFields);
        }
        
        /// <summary>
        /// Extracts MPRL fields using reflection with error handling.
        /// </summary>
        private (Vector3 Position, uint Unknown4, uint Unknown6, Dictionary<string, object> RawFields) ExtractMprlFields(object mprlEntry)
        {
            var rawFields = new Dictionary<string, object>();
            var position = Vector3.Zero;
            uint unknown4 = 0;
            uint unknown6 = 0;
            
            try
            {
                var type = mprlEntry.GetType();
                var fields = type.GetFields();
                
                foreach (var field in fields)
                {
                    var value = field.GetValue(mprlEntry);
                    rawFields[field.Name] = value ?? DBNull.Value;
                    
                    // Extract known fields by name
                    if (field.Name.Contains("Position") && value is Vector3 pos)
                    {
                        position = pos;
                    }
                    else if (field.Name.Contains("Unknown4") && value is uint u4)
                    {
                        unknown4 = u4;
                    }
                    else if (field.Name.Contains("Unknown6") && value is uint u6)
                    {
                        unknown6 = u6;
                    }
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error extracting MPRL fields: {ex.Message}");
            }
            
            return (position, unknown4, unknown6, rawFields);
        }
        
        /// <summary>
        /// Performs spatial clustering on surfaces to identify building groups.
        /// </summary>
        private List<List<int>> PerformSpatialClustering(List<Pm4Surface> surfaces, float proximityThreshold = 50.0f)
        {
            var groups = new List<List<int>>();
            var processed = new HashSet<int>();
            
            foreach (var surface in surfaces)
            {
                if (processed.Contains(surface.Id)) continue;
                
                var group = new List<int> { surface.Id };
                processed.Add(surface.Id);
                
                // Find nearby surfaces
                foreach (var otherSurface in surfaces)
                {
                    if (processed.Contains(otherSurface.Id)) continue;
                    
                    var distance = Vector3.Distance(
                        new Vector3(surface.BoundsCenterX, surface.BoundsCenterY, surface.BoundsCenterZ),
                        new Vector3(otherSurface.BoundsCenterX, otherSurface.BoundsCenterY, otherSurface.BoundsCenterZ)
                    );
                    
                    if (distance <= proximityThreshold)
                    {
                        group.Add(otherSurface.Id);
                        processed.Add(otherSurface.Id);
                    }
                }
                
                // Only include groups with reasonable size
                if (group.Count >= 3 && group.Count <= 200)
                {
                    groups.Add(group);
                }
            }
            
            return groups;
        }
        
        /// <summary>
        /// Calculates bounds for a group of surfaces.
        /// </summary>
        private (Vector3 min, Vector3 max, Vector3 center) CalculateGroupBounds(List<Pm4Surface> surfaces)
        {
            if (surfaces.Count == 0)
                return (Vector3.Zero, Vector3.Zero, Vector3.Zero);
            
            var min = new Vector3(
                surfaces.Min(s => s.BoundsMinX),
                surfaces.Min(s => s.BoundsMinY),
                surfaces.Min(s => s.BoundsMinZ)
            );
            
            var max = new Vector3(
                surfaces.Max(s => s.BoundsMaxX),
                surfaces.Max(s => s.BoundsMaxY),
                surfaces.Max(s => s.BoundsMaxZ)
            );
            
            var center = (min + max) * 0.5f;
            return (min, max, center);
        }
        
        #endregion
    }
}
