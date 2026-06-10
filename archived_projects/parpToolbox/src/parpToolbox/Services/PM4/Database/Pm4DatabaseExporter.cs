using Microsoft.EntityFrameworkCore;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Utils;
using System.Numerics;
using System.Reflection;
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
        public async Task<int> ExportSceneAsync(Pm4Scene scene, string fileName, string filePath, IReadOnlyDictionary<string, byte[]>? capturedRawData = null)
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
            await ExportTriangleVerticesAsync(context, pm4File.Id, scene.Triangles);
            await ExportSurfacesAsync(context, pm4File.Id, scene.Surfaces, scene.Vertices);
            await ExportLinksAsync(context, pm4File.Id, scene.Links?.Cast<object>().ToList() ?? new List<object>());
            await ExportPlacementsAsync(context, pm4File.Id, scene.Placements?.Cast<object>().ToList() ?? new List<object>());
            await ExportPropertiesAsync(context, pm4File.Id, scene.Properties?.Cast<object>().ToList() ?? new List<object>());
            await ExportSurfaceGroupsAsync(context, pm4File.Id, scene.Groups);
            
            // Export hierarchical container relationships as JSON (fast, bypasses SQLite bottlenecks)
            await ExportHierarchicalContainersJsonAsync(scene, fileName, filePath);
            
            // Export raw chunks for future-proofing
            await ExportRawChunksAsync(context, pm4File.Id, capturedRawData?.ToDictionary(kvp => kvp.Key, kvp => kvp.Value) ?? new Dictionary<string, byte[]>());
            
            ConsoleLogger.WriteLine($"Successfully exported PM4 scene to database. File ID: {pm4File.Id}");
            return pm4File.Id;
        }
        
        /// <summary>
        /// Exports vertices to the database with optimized batch processing and detailed progress tracking.
        /// </summary>
        private async Task ExportVerticesAsync(Pm4DatabaseContext context, int pm4FileId, List<Vector3> vertices)
        {
            ConsoleLogger.WriteLine($"Exporting {vertices.Count:N0} vertices with optimized batching...");
            
            var startTime = DateTime.Now;
            const int batchSize = 50000;
            
            // Disable change tracking for bulk inserts
            context.ChangeTracker.AutoDetectChangesEnabled = false;
            
            for (int i = 0; i < vertices.Count; i += batchSize)
            {
                var batchStartTime = DateTime.Now;
                var endIndex = Math.Min(i + batchSize, vertices.Count);
                var actualBatchSize = endIndex - i;
                
                var batch = new List<Pm4Vertex>(actualBatchSize);
                for (int j = i; j < endIndex; j++)
                {
                    var vertex = vertices[j];
                    batch.Add(new Pm4Vertex
                    {
                        Pm4FileId = pm4FileId,
                        GlobalIndex = j,
                        X = vertex.X,
                        Y = vertex.Y,
                        Z = vertex.Z,
                        ChunkType = "MSVT" // Default to MSVT, could be enhanced to detect MSPV
                    });
                }
                
                context.Vertices.AddRange(batch);
                await context.SaveChangesAsync();
                context.ChangeTracker.Clear();
                
                var batchTime = DateTime.Now - batchStartTime;
                var totalTime = DateTime.Now - startTime;
                var progress = (double)endIndex / vertices.Count;
                var eta = progress > 0 ? TimeSpan.FromMilliseconds(totalTime.TotalMilliseconds / progress - totalTime.TotalMilliseconds) : TimeSpan.Zero;
                
                ConsoleLogger.WriteLine($"  Vertices: {endIndex:N0}/{vertices.Count:N0} ({progress:P1}) | Batch: {batchTime.TotalSeconds:F1}s | Total: {totalTime:mm\\:ss} | ETA: {eta:mm\\:ss}");
            }
            
            // Re-enable change tracking
            context.ChangeTracker.AutoDetectChangesEnabled = true;
            
            var finalTime = DateTime.Now - startTime;
            ConsoleLogger.WriteLine($"Completed vertex export: {vertices.Count:N0} vertices in {finalTime:mm\\:ss}");
        }
        
        /// <summary>
        /// Exports triangles to the database with optimized batch processing and detailed progress tracking.
        /// </summary>
        private async Task ExportTrianglesAsync(Pm4DatabaseContext context, int pm4FileId, List<(int A, int B, int C)> triangles)
        {
            ConsoleLogger.WriteLine($"Exporting {triangles.Count:N0} triangles with optimized batching...");
            
            var startTime = DateTime.Now;
            const int batchSize = 50000;
            
            // Disable change tracking for bulk inserts
            context.ChangeTracker.AutoDetectChangesEnabled = false;
            
            for (int i = 0; i < triangles.Count; i += batchSize)
            {
                var batchStartTime = DateTime.Now;
                var endIndex = Math.Min(i + batchSize, triangles.Count);
                var actualBatchSize = endIndex - i;
                
                var batch = new List<Pm4Triangle>(actualBatchSize);
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
                context.ChangeTracker.Clear();
                
                var batchTime = DateTime.Now - batchStartTime;
                var totalTime = DateTime.Now - startTime;
                var progress = (double)endIndex / triangles.Count;
                var eta = progress > 0 ? TimeSpan.FromMilliseconds(totalTime.TotalMilliseconds / progress - totalTime.TotalMilliseconds) : TimeSpan.Zero;
                
                ConsoleLogger.WriteLine($"  Triangles: {endIndex:N0}/{triangles.Count:N0} ({progress:P1}) | Batch: {batchTime.TotalSeconds:F1}s | Total: {totalTime:mm\\:ss} | ETA: {eta:mm\\:ss}");
            }
            
            // Re-enable change tracking
            context.ChangeTracker.AutoDetectChangesEnabled = true;
            
            var finalTime = DateTime.Now - startTime;
            ConsoleLogger.WriteLine($"Completed triangle export: {triangles.Count:N0} triangles in {finalTime:mm\\:ss}");
        }
        
        /// <summary>
        /// Exports triangle-vertex relationships to the database for relational triangle reconstruction.
        /// </summary>
        private async Task ExportTriangleVerticesAsync(Pm4DatabaseContext context, int pm4FileId, List<(int A, int B, int C)> triangles)
        {
            ConsoleLogger.WriteLine($"[CHUNK PROCESSING] Starting Triangle-Vertex relationship export: {triangles.Count:N0} triangles");
            
            var startTime = DateTime.Now;
            const int batchSize = 50000;
            
            // Disable change tracking for performance
            context.ChangeTracker.AutoDetectChangesEnabled = false;
            
            for (int i = 0; i < triangles.Count; i += batchSize)
            {
                var batchStartTime = DateTime.Now;
                var endIndex = Math.Min(i + batchSize, triangles.Count);
                
                var batch = new List<Pm4TriangleVertex>();
                
                // Get actual Triangle and Vertex IDs from database by GlobalIndex
                var triangleIds = await context.Triangles
                    .Where(t => t.Pm4FileId == pm4FileId && t.GlobalIndex >= i && t.GlobalIndex < endIndex)
                    .OrderBy(t => t.GlobalIndex)
                    .Select(t => new { t.Id, t.GlobalIndex })
                    .ToListAsync();
                    
                var vertexIds = await context.Vertices
                    .Where(v => v.Pm4FileId == pm4FileId)
                    .Select(v => new { v.Id, v.GlobalIndex })
                    .ToListAsync();
                    
                // Create lookup dictionary for fast vertex ID resolution
                var vertexIdLookup = vertexIds.ToDictionary(v => v.GlobalIndex, v => v.Id);
                
                for (int j = i; j < endIndex; j++)
                {
                    var triangle = triangles[j];
                    var triangleRecord = triangleIds.FirstOrDefault(t => t.GlobalIndex == j);
                    
                    if (triangleRecord == null)
                    {
                        ConsoleLogger.WriteLine($"[ERROR] Triangle with GlobalIndex {j} not found in database");
                        continue;
                    }
                    
                    // Validate vertex indices exist
                    if (!vertexIdLookup.ContainsKey(triangle.A) || 
                        !vertexIdLookup.ContainsKey(triangle.B) || 
                        !vertexIdLookup.ContainsKey(triangle.C))
                    {
                        ConsoleLogger.WriteLine($"[ERROR] Invalid vertex indices for triangle {j}: A={triangle.A}, B={triangle.B}, C={triangle.C}");
                        continue;
                    }
                    
                    // Create triangle-vertex relationships for each of the 3 vertices
                    // VertexPosition: 0=A, 1=B, 2=C
                    batch.Add(new Pm4TriangleVertex
                    {
                        TriangleId = triangleRecord.Id, // Use actual Triangle database ID
                        VertexId = vertexIdLookup[triangle.A], // Use actual Vertex database ID
                        VertexPosition = 0 // A vertex
                    });
                    
                    batch.Add(new Pm4TriangleVertex
                    {
                        TriangleId = triangleRecord.Id,
                        VertexId = vertexIdLookup[triangle.B],
                        VertexPosition = 1 // B vertex
                    });
                    
                    batch.Add(new Pm4TriangleVertex
                    {
                        TriangleId = triangleRecord.Id,
                        VertexId = vertexIdLookup[triangle.C],
                        VertexPosition = 2 // C vertex
                    });
                }
                
                context.TriangleVertices.AddRange(batch);
                await context.SaveChangesAsync();
                context.ChangeTracker.Clear();
                
                var batchTime = DateTime.Now - batchStartTime;
                var totalTime = DateTime.Now - startTime;
                var progress = (double)endIndex / triangles.Count;
                var eta = progress > 0 ? TimeSpan.FromMilliseconds(totalTime.TotalMilliseconds / progress - totalTime.TotalMilliseconds) : TimeSpan.Zero;
                
                ConsoleLogger.WriteLine($"  TriangleVertices: {endIndex:N0}/{triangles.Count:N0} ({progress:P1}) | Batch: {batchTime.TotalSeconds:F1}s | Total: {totalTime:mm\\:ss} | ETA: {eta:mm\\:ss}");
            }
            
            // Re-enable change tracking
            context.ChangeTracker.AutoDetectChangesEnabled = true;
            
            var finalTime = DateTime.Now - startTime;
            var totalRelationships = triangles.Count * 3; // 3 relationships per triangle
            ConsoleLogger.WriteLine($"Completed triangle-vertex export: {totalRelationships:N0} relationships in {finalTime:mm\\:ss}");
        }
        
        /// <summary>
        /// Exports surfaces to the database with proper MSUR Entry field extraction.
        /// </summary>
        private async Task ExportSurfacesAsync(Pm4DatabaseContext context, int pm4FileId, List<MsurChunk.Entry> surfaces, List<Vector3> vertices)
        {
            ConsoleLogger.WriteLine($"[CHUNK PROCESSING] Starting MSUR surface export: {surfaces.Count:N0} entries");
            
            // Debug: Log first surface structure if available
            if (surfaces.Count > 0)
            {
                LogChunkStructure("MSUR", surfaces[0]);
            }
            
            const int batchSize = 1000;
            
            for (int i = 0; i < surfaces.Count; i += batchSize)
            {
                var batchStartTime = DateTime.Now;
                var endIndex = Math.Min(i + batchSize, surfaces.Count);
                
                var batch = new List<Pm4Surface>();
                for (int j = i; j < endIndex; j++)
                {
                    var surface = surfaces[j];
                    
                    // Direct extraction from raw MSUR.Entry - no detection logic needed!
                    batch.Add(new Pm4Surface
                    {
                        Pm4FileId = pm4FileId,
                        GlobalIndex = j,
                        MsviFirstIndex = (int)surface.MsviFirstIndex,
                        IndexCount = (ushort)surface.IndexCount,
                        GroupKey = surface.SurfaceGroupKey,
                        RawFlags = (ushort)surface.SurfaceAttributeMask,
                        // Direct MSUR.Entry field mapping - no interpretation needed
                        BoundsMinX = surface.Nx,
                        BoundsMinY = surface.Ny, 
                        BoundsMinZ = surface.Nz,
                        BoundsMaxX = surface.Height,
                        BoundsMaxY = surface.MdosIndex,
                        BoundsMaxZ = surface.CompositeKey,
                        BoundsCenterX = surface.MsviFirstIndex,
                        BoundsCenterY = surface.IndexCount,
                        BoundsCenterZ = surface.SurfaceGroupKey
                    });
                }
                
                context.Surfaces.AddRange(batch);
                await context.SaveChangesAsync();
                
                var batchTime = DateTime.Now - batchStartTime;
                var progress = (double)endIndex / surfaces.Count;
                
                ConsoleLogger.WriteLine($"  Surfaces: {endIndex:N0}/{surfaces.Count:N0} ({progress:P1}) | Batch: {batchTime.TotalSeconds:F1}s");
            }
            
            ConsoleLogger.WriteLine($"Completed MSUR surface export: {surfaces.Count} surface entries");
        }
        
        /// <summary>
        /// Exports MSLK links to the database with optimized batch processing and minimal serialization.
        /// </summary>
        private async Task ExportLinksAsync(Pm4DatabaseContext context, int pm4FileId, List<object> links)
        {
            ConsoleLogger.WriteLine($"[CHUNK PROCESSING] Starting MSLK link export: {links.Count:N0} entries with optimized batching...");
            
            var startTime = DateTime.Now;
            
            // Debug: Log first link structure
            if (links.Count > 0)
            {
                LogChunkStructure("MSLK", links[0]);
            }
            
            // Use same large batch size as vertices/triangles for performance
            const int batchSize = 50000;
            
            // Disable change tracking for bulk inserts
            context.ChangeTracker.AutoDetectChangesEnabled = false;
            
            for (int i = 0; i < links.Count; i += batchSize)
            {
                var batchStartTime = DateTime.Now;
                var endIndex = Math.Min(i + batchSize, links.Count);
                
                var batch = new List<Pm4Link>();
                for (int j = i; j < endIndex; j++)
                {
                    var link = links[j];
                    
                    // Check if this is an MslkEntry or if we need to extract entries from chunk
                    if (link.GetType().Name.Contains("Entry"))
                    {
                        // Direct MslkEntry - extract fields properly
                        var fields = ExtractMslkEntryFields(link);
                        
                        batch.Add(new Pm4Link
                        {
                            Pm4FileId = pm4FileId,
                            GlobalIndex = j,
                            ParentIndex = fields.ParentIndex,
                            MspiFirstIndex = fields.MspiFirstIndex,
                            MspiIndexCount = fields.MspiIndexCount,
                            ReferenceIndex = fields.ReferenceIndex,
                            RawFieldsJson = System.Text.Json.JsonSerializer.Serialize(fields.RawFields)
                        });
                    }
                    else
                    {
                        // Fallback to old extraction method
                        var fields = ExtractMslkFields(link);
                        
                        batch.Add(new Pm4Link
                        {
                            Pm4FileId = pm4FileId,
                            GlobalIndex = j,
                            ParentIndex = fields.ParentIndex,
                            MspiFirstIndex = fields.MspiFirstIndex,
                            MspiIndexCount = fields.MspiIndexCount,
                            ReferenceIndex = fields.ReferenceIndex,
                            RawFieldsJson = System.Text.Json.JsonSerializer.Serialize(fields.RawFields)
                        });
                    }
                }
                
                context.Links.AddRange(batch);
                await context.SaveChangesAsync();
                
                // Clear tracked entities to prevent memory buildup
                context.ChangeTracker.Clear();
                
                // Progress tracking with timing
                var batchTime = DateTime.Now - batchStartTime;
                var totalTime = DateTime.Now - startTime;
                var progress = (double)endIndex / links.Count;
                var eta = progress > 0 ? TimeSpan.FromMilliseconds(totalTime.TotalMilliseconds / progress - totalTime.TotalMilliseconds) : TimeSpan.Zero;
                
                ConsoleLogger.WriteLine($"  Links: {endIndex:N0}/{links.Count:N0} ({progress:P1}) | Batch: {batchTime.TotalSeconds:F1}s | Total: {totalTime:mm\\:ss} | ETA: {eta:mm\\:ss}");
            }
            
            // Re-enable change tracking
            context.ChangeTracker.AutoDetectChangesEnabled = true;
            
            var finalTime = DateTime.Now - startTime;
            ConsoleLogger.WriteLine($"Completed MSLK link export: {links.Count:N0} links in {finalTime:mm\\:ss}");
        }
        
        /// <summary>
        /// Exports MPRL placements to the database with reflection-safe field extraction and detailed logging.
        /// </summary>
        private async Task ExportPlacementsAsync(Pm4DatabaseContext context, int pm4FileId, List<object> placements)
        {
            ConsoleLogger.WriteLine($"[CHUNK PROCESSING] Starting MPRL placement export: {placements.Count:N0} entries");
            
            // Debug: Log first placement structure
            if (placements.Count > 0)
            {
                LogChunkStructure("MPRL", placements[0]);
            }
            
            const int batchSize = 1000;
            
            for (int i = 0; i < placements.Count; i += batchSize)
            {
                var batch = new List<Pm4Placement>();
                
                for (int j = i; j < Math.Min(i + batchSize, placements.Count); j++)
                {
                    var placement = placements[j];
                    
                    try
                    {
                        // Check if this is a proper MPRL Entry record
                        if (placement.GetType().Name.Contains("Entry"))
                        {
                            // Extract fields from MPRL Entry record
                            var fields = ExtractMprlEntryFields(placement);
                            
                            var pm4Placement = new Pm4Placement
                            {
                                Pm4FileId = pm4FileId,
                                GlobalIndex = j,
                                PositionX = fields.Position.X,
                                PositionY = fields.Position.Y,
                                PositionZ = fields.Position.Z,
                                Unknown4 = fields.Unknown4,
                                Unknown6 = fields.Unknown6,
                                RawFieldsJson = System.Text.Json.JsonSerializer.Serialize(fields.RawFields)
                            };
                            
                            batch.Add(pm4Placement);
                        }
                        else
                        {
                            // Fallback to old extraction method
                            var fields = ExtractMprlFields(placement);
                            
                            var pm4Placement = new Pm4Placement
                            {
                                Pm4FileId = pm4FileId,
                                GlobalIndex = j,
                                PositionX = fields.Position.X,
                                PositionY = fields.Position.Y,
                                PositionZ = fields.Position.Z,
                                Unknown4 = fields.Unknown4,
                                Unknown6 = fields.Unknown6,
                                RawFieldsJson = System.Text.Json.JsonSerializer.Serialize(fields.RawFields)
                            };
                            
                            batch.Add(pm4Placement);
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[MPRL EXPORT] Error processing placement {j}: {ex.Message}");
                    }
                }
                
                context.Placements.AddRange(batch);
                await context.SaveChangesAsync();
                
                if (i % (batchSize * 10) == 0 || i + batchSize >= placements.Count)
                {
                    ConsoleLogger.WriteLine($"  Exported {Math.Min(i + batchSize, placements.Count)} / {placements.Count} MPRL placements");
                }
            }
            
            ConsoleLogger.WriteLine($"Completed MPRL placement export: {placements.Count} placements");
        }
        
        /// <summary>
        /// Creates initial surface groups using spatial clustering and exports them.
        /// DISABLED: Spatial clustering O(N²) algorithm causes freeze on large datasets (518K+ surfaces).
        /// </summary>
        private async Task ExportSurfaceGroupsAsync(Pm4DatabaseContext context, int pm4FileId, List<SurfaceGroup> groups)
        {
            ConsoleLogger.WriteLine($"[SPATIAL CLUSTERING] Skipping spatial clustering for {groups.Count} surfaces (algorithm disabled due to O(N²) performance issue)");
            
            // DISABLED: Spatial clustering causes freeze on large datasets
            // var surfaces = await context.Surfaces.Where(s => s.Pm4FileId == pm4FileId).ToListAsync();
            // var spatialGroups = PerformSpatialClustering(surfaces);
            
            // Create minimal placeholder groups for database integrity
            var spatialGroups = new List<List<int>>();
            
            ConsoleLogger.WriteLine($"[SPATIAL CLUSTERING] Skipped clustering, created {spatialGroups.Count} groups (spatial clustering disabled)");
            
            // DISABLED: Export surface groups step completely disabled
            // Since spatialGroups is empty, this foreach won't execute anyway
            // foreach (var (groupIndex, surfaceIds) in spatialGroups.Select((g, i) => (i, g))) { ... }
            
            // Skip all surface group processing since spatial clustering is disabled
            ConsoleLogger.WriteLine($"Completed surface group export: {spatialGroups.Count} groups");
        }
        
        /// <summary>
        /// Exports raw chunk data for future-proofing.
        /// </summary>
        private async Task ExportRawChunksAsync(Pm4DatabaseContext context, int fileId, Dictionary<string, byte[]> capturedRawData)
        {
            if (capturedRawData == null || !capturedRawData.Any())
            {
                ConsoleLogger.WriteLine("[CHUNK STORAGE] No raw chunks captured - raw data capture may be disabled");
                return;
            }
            
            ConsoleLogger.WriteLine($"[CHUNK STORAGE] Exporting {capturedRawData.Count} raw chunks for future-proofing...");
            
            var rawChunks = new List<Pm4RawChunk>();
            var offset = 0;
            
            foreach (var (chunkType, rawData) in capturedRawData)
            {
                var rawChunk = new Pm4RawChunk
                {
                    Pm4FileId = fileId,
                    ChunkType = chunkType,
                    ChunkOffset = offset,
                    ChunkSize = rawData.Length,
                    RawData = rawData,
                    ParsedAt = DateTime.UtcNow,
                    ParserVersion = "1.0.0-raw-capture",
                    ParsingNotes = $"Raw chunk captured during export, size: {rawData.Length} bytes"
                };
                
                rawChunks.Add(rawChunk);
                offset += rawData.Length;
            }
            
            context.RawChunks.AddRange(rawChunks);
            await context.SaveChangesAsync();
            
            ConsoleLogger.WriteLine($"[CHUNK STORAGE] Exported {rawChunks.Count} raw chunks totaling {offset:N0} bytes");
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
        /// Extracts MSLK Entry fields using reflection with detailed property access for proper field extraction.
        /// </summary>
        private (uint ParentIndex, int MspiFirstIndex, int MspiIndexCount, uint ReferenceIndex, Dictionary<string, object> RawFields) ExtractMslkEntryFields(object mslkEntry)
        {
            var rawFields = new Dictionary<string, object>();
            uint parentIndex = 0;
            int mspiFirstIndex = -1;
            int mspiIndexCount = 0;
            uint referenceIndex = 0;
            
            try
            {
                var type = mslkEntry.GetType();
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] MSLK Entry type: {type.Name}");
                
                // Extract using property reflection (more reliable than field position)
                var properties = type.GetProperties();
                var fields = type.GetFields();
                
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] MSLK Entry has {properties.Length} properties and {fields.Length} fields");
                
                // Try properties first (MslkEntry uses properties)
                foreach (var prop in properties)
                {
                    try
                    {
                        var value = prop.GetValue(mslkEntry);
                        rawFields[prop.Name] = value ?? DBNull.Value;
                        
                        // Extract known fields by name pattern
                        switch (prop.Name)
                        {
                            case "ParentIndex":
                            case "Unknown_0x04":
                                if (value is uint ui) parentIndex = ui;
                                else if (value is int ii) parentIndex = (uint)ii;
                                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE]   {prop.Name}: {value}");
                                break;
                            case "MspiFirstIndex":
                                if (value is int mfi) mspiFirstIndex = mfi;
                                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE]   {prop.Name}: {value}");
                                break;
                            case "MspiIndexCount":
                                if (value is int mic) mspiIndexCount = mic;
                                else if (value is byte mb) mspiIndexCount = mb;
                                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE]   {prop.Name}: {value}");
                                break;
                            case "ReferenceIndex":
                            case "Unknown_0x10":
                                if (value is uint ri) referenceIndex = ri;
                                else if (value is ushort rs) referenceIndex = rs;
                                else if (value is int rii) referenceIndex = (uint)rii;
                                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE]   {prop.Name}: {value}");
                                break;
                            default:
                                // Log other interesting fields
                                if (prop.Name.StartsWith("Unknown"))
                                {
                                    ConsoleLogger.WriteLine($"[CHUNK STRUCTURE]   {prop.Name}: {value}");
                                }
                                break;
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] Error reading property {prop.Name}: {ex.Message}");
                    }
                }
                
                // Also try fields as backup
                foreach (var field in fields)
                {
                    try
                    {
                        var value = field.GetValue(mslkEntry);
                        if (!rawFields.ContainsKey(field.Name))
                        {
                            rawFields[field.Name] = value ?? DBNull.Value;
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] Error reading field {field.Name}: {ex.Message}");
                    }
                }
                
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] MSLK Entry extracted: ParentIndex={parentIndex}, FirstIndex={mspiFirstIndex}, IndexCount={mspiIndexCount}, ReferenceIndex={referenceIndex}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error extracting MSLK Entry fields: {ex.Message}");
            }
            
            return (parentIndex, mspiFirstIndex, mspiIndexCount, referenceIndex, rawFields);
        }
        
        /// <summary>
        /// Checks if the surface group contains MSUR Entry data that can be extracted.
        /// </summary>
        private bool HasMsurEntryData(object surfaceGroup)
        {
            if (surfaceGroup == null) return false;
            
            var type = surfaceGroup.GetType();
            
            // Check if it's an Entry type or has Entry-like properties
            if (type.Name.Contains("Entry")) return true;
            
            // Check for MSUR Entry properties
            var properties = type.GetProperties();
            var hasNormalProps = properties.Any(p => p.Name.Equals("Nx", StringComparison.OrdinalIgnoreCase) ||
                                                     p.Name.Equals("Ny", StringComparison.OrdinalIgnoreCase) ||
                                                     p.Name.Equals("Nz", StringComparison.OrdinalIgnoreCase));
            var hasIndexProps = properties.Any(p => p.Name.Contains("MsviFirstIndex") ||
                                                    p.Name.Contains("IndexCount"));
            
            return hasNormalProps && hasIndexProps;
        }
        
        /// <summary>
        /// Extracts MSUR Entry fields from a surface object using reflection.
        /// Returns normal vectors, height, indices, and other MSUR-specific data.
        /// </summary>
        private (uint MsviFirstIndex, byte IndexCount, byte GroupKey, uint RawFlags, float Nx, float Ny, float Nz, float Height, uint MdosIndex, uint PackedParams, Dictionary<string, object> RawFields) ExtractMsurEntryFields(object msurEntry)
        {
            var rawFields = new Dictionary<string, object>();
            
            // Default values
            uint msviFirstIndex = 0;
            byte indexCount = 0;
            byte groupKey = 0;
            uint rawFlags = 0;
            float nx = 0.0f, ny = 0.0f, nz = 0.0f;
            float height = 0.0f;
            uint mdosIndex = 0;
            uint packedParams = 0;
            
            try
            {
                var type = msurEntry.GetType();
                var properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
                
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] MSUR Entry type: {type.Name}, Properties: {properties.Length}");
                
                foreach (var prop in properties)
                {
                    try
                    {
                        var value = prop.GetValue(msurEntry);
                        rawFields[prop.Name] = value ?? "null";
                        
                        // Extract known MSUR Entry fields
                        switch (prop.Name)
                        {
                            case "MsviFirstIndex":
                                if (value is uint uintVal) msviFirstIndex = uintVal;
                                break;
                            case "IndexCount":
                                if (value is byte byteVal) indexCount = byteVal;
                                break;
                            case "FlagsOrUnknown_0x00":
                            case "SurfaceGroupKey":
                                if (value is byte groupVal) groupKey = groupVal;
                                break;
                            case "Unknown_0x02":
                                if (value is byte flagVal) rawFlags = flagVal;
                                break;
                            case "Nx":
                                if (value is float nxVal) nx = nxVal;
                                break;
                            case "Ny":
                                if (value is float nyVal) ny = nyVal;
                                break;
                            case "Nz":
                                if (value is float nzVal) nz = nzVal;
                                break;
                            case "Height":
                                if (value is float heightVal) height = heightVal;
                                break;
                            case "MdosIndex":
                                if (value is uint mdosVal) mdosIndex = mdosVal;
                                break;
                            case "PackedParams":
                                if (value is uint packedVal) packedParams = packedVal;
                                break;
                        }
                        
                        ConsoleLogger.WriteLine($"[CHUNK STRUCTURE]   {prop.Name} = {value ?? "null"} ({prop.PropertyType.Name})");
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] Error reading property {prop.Name}: {ex.Message}");
                        rawFields[prop.Name] = $"ERROR: {ex.Message}";
                    }
                }
                
                // Log extracted values
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] Extracted MSUR fields: Normals=({nx:F3},{ny:F3},{nz:F3}), Height={height:F3}, MsviFirstIndex={msviFirstIndex}, IndexCount={indexCount}, GroupKey={groupKey}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] Error extracting MSUR fields: {ex.Message}");
                rawFields["extraction_error"] = ex.Message;
            }
            
            return (msviFirstIndex, indexCount, groupKey, rawFlags, nx, ny, nz, height, mdosIndex, packedParams, rawFields);
        }
        
        /// <summary>
        /// Logs the structure of a chunk for debugging.
        /// </summary>
        private void LogChunkStructure(string chunkType, object chunkEntry)
        {
            try
            {
                var type = chunkEntry.GetType();
                var fields = type.GetFields();
                
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] {chunkType} has {fields.Length} fields:");
                
                foreach (var field in fields)
                {
                    var value = field.GetValue(chunkEntry);
                    var valueStr = value?.ToString() ?? "null";
                    
                    // Truncate long values
                    if (valueStr.Length > 50)
                        valueStr = valueStr.Substring(0, 47) + "...";
                    
                    ConsoleLogger.WriteLine($"  {field.Name} ({field.FieldType.Name}): {valueStr}");
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"[CHUNK STRUCTURE] Error logging {chunkType} structure: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Extracts MPRL fields using reflection with improved field detection and error handling.
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
                
                ConsoleLogger.WriteLine($"[MPRL DECODE] Processing entry with {fields.Length} fields");
                
                foreach (var field in fields.Select((f, i) => new { Field = f, Index = i }))
                {
                    var value = field.Field.GetValue(mprlEntry);
                    rawFields[field.Field.Name] = value ?? DBNull.Value;
                    
                    // Log field for debugging
                    var valueStr = value?.ToString() ?? "null";
                    if (valueStr.Length > 30) valueStr = valueStr.Substring(0, 27) + "...";
                    ConsoleLogger.WriteLine($"  Field[{field.Index}] {field.Field.Name}: {valueStr}");
                    
                    // Extract position data - try multiple field name patterns
                    if (value is Vector3 pos && (field.Field.Name.Contains("Position") || field.Field.Name.Contains("Pos") || field.Index == 0))
                    {
                        position = pos;
                        ConsoleLogger.WriteLine($"[MPRL DECODE] Found position: {pos}");
                    }
                    // Try extracting position from individual X,Y,Z fields
                    else if (field.Field.Name.Contains("X") && value is float x)
                    {
                        position = new Vector3(x, position.Y, position.Z);
                    }
                    else if (field.Field.Name.Contains("Y") && value is float y)
                    {
                        position = new Vector3(position.X, y, position.Z);
                    }
                    else if (field.Field.Name.Contains("Z") && value is float z)
                    {
                        position = new Vector3(position.X, position.Y, z);
                    }
                    // Extract Unknown4 and Unknown6 fields
                    else if (field.Field.Name.Contains("Unknown4") && value is uint u4)
                    {
                        unknown4 = u4;
                    }
                    else if (field.Field.Name.Contains("Unknown6") && value is uint u6)
                    {
                        unknown6 = u6;
                    }
                }
                
                ConsoleLogger.WriteLine($"[MPRL DECODE] Final position: {position}, Unknown4: {unknown4}, Unknown6: {unknown6}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"[MPRL DECODE] Error extracting MPRL fields: {ex.Message}");
            }
            
            return (position, unknown4, unknown6, rawFields);
        }
        
        /// <summary>
        /// Extracts fields from MPRL Entry records using direct property/field access.
        /// </summary>
        private (Vector3 Position, uint Unknown4, uint Unknown6, Dictionary<string, object> RawFields) ExtractMprlEntryFields(object mprlEntry)
        {
            var rawFields = new Dictionary<string, object>();
            var position = Vector3.Zero;
            uint unknown4 = 0;
            uint unknown6 = 0;
            
            try
            {
                var type = mprlEntry.GetType();
                ConsoleLogger.WriteLine($"[MPRL ENTRY] Processing {type.Name} entry");
                
                // Try to access Entry record properties directly
                var properties = type.GetProperties();
                var fields = type.GetFields();
                
                ConsoleLogger.WriteLine($"[MPRL ENTRY] Found {properties.Length} properties, {fields.Length} fields");
                
                // Extract from properties (record types use properties)
                foreach (var prop in properties)
                {
                    try
                    {
                        var value = prop.GetValue(mprlEntry);
                        rawFields[prop.Name] = value ?? DBNull.Value;
                        
                        var valueStr = value?.ToString() ?? "null";
                        if (valueStr.Length > 30) valueStr = valueStr.Substring(0, 27) + "...";
                        ConsoleLogger.WriteLine($"  Property {prop.Name}: {valueStr}");
                        
                        // Extract position from Vector3 property
                        if (prop.Name == "Position" && value is Vector3 pos)
                        {
                            position = pos;
                            ConsoleLogger.WriteLine($"[MPRL ENTRY] Found position: {pos}");
                        }
                        // Extract Unknown4 field (likely ushort Unknown4)
                        else if (prop.Name == "Unknown4" && value != null)
                        {
                            unknown4 = Convert.ToUInt32(value);
                            ConsoleLogger.WriteLine($"[MPRL ENTRY] Found Unknown4: {unknown4}");
                        }
                        // Extract Unknown6 field (likely ushort Unknown6)
                        else if (prop.Name == "Unknown6" && value != null)
                        {
                            unknown6 = Convert.ToUInt32(value);
                            ConsoleLogger.WriteLine($"[MPRL ENTRY] Found Unknown6: {unknown6}");
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[MPRL ENTRY] Error accessing property {prop.Name}: {ex.Message}");
                    }
                }
                
                // Also check fields if properties didn't work
                foreach (var field in fields)
                {
                    try
                    {
                        var value = field.GetValue(mprlEntry);
                        rawFields[field.Name] = value ?? DBNull.Value;
                        
                        var valueStr = value?.ToString() ?? "null";
                        if (valueStr.Length > 30) valueStr = valueStr.Substring(0, 27) + "...";
                        ConsoleLogger.WriteLine($"  Field {field.Name}: {valueStr}");
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[MPRL ENTRY] Error accessing field {field.Name}: {ex.Message}");
                    }
                }
                
                ConsoleLogger.WriteLine($"[MPRL ENTRY] Final position: {position}, Unknown4: {unknown4}, Unknown6: {unknown6}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"[MPRL ENTRY] Error extracting MPRL Entry fields: {ex.Message}");
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
        
        /// <summary>
        /// Exports hierarchical container relationships decoded from BoundsCenterX/Y/Z fields.
        /// This replaces spatial clustering with proper hierarchical object assembly.
        /// </summary>
        private async Task ExportHierarchicalContainersAsync(Pm4DatabaseContext context, int pm4FileId)
        {
            ConsoleLogger.WriteLine("[HIERARCHICAL EXPORT] Decoding hierarchical container relationships...");
            
            // Load surfaces for hierarchical analysis (data already in memory/context)
            var surfaces = await context.Surfaces.Where(s => s.Pm4FileId == pm4FileId).ToListAsync();
            var links = await context.Links.Where(l => l.Pm4FileId == pm4FileId).ToListAsync();
            var placements = await context.Placements.Where(p => p.Pm4FileId == pm4FileId).ToListAsync();
            
            if (!surfaces.Any())
            {
                ConsoleLogger.WriteLine("[HIERARCHICAL EXPORT] No surfaces found for hierarchical decoding.");
                return;
            }
            
            ConsoleLogger.WriteLine($"[HIERARCHICAL EXPORT] Processing {surfaces.Count} surfaces, {links.Count} links, {placements.Count} placements");
            
            // Group surfaces by hierarchical container identifiers (BoundsCenterX/Y/Z)
            var containerGroups = surfaces
                .GroupBy(s => new { 
                    ContainerX = s.BoundsCenterX, 
                    ContainerY = s.BoundsCenterY, 
                    ContainerZ = s.BoundsCenterZ 
                })
                .ToList();
            
            ConsoleLogger.WriteLine($"[HIERARCHICAL EXPORT] Found {containerGroups.Count} unique container combinations");
            
            var containersToInsert = new List<Pm4HierarchicalContainer>();
            var membersToInsert = new List<Pm4HierarchicalContainerMember>();
            
            foreach (var group in containerGroups)
            {
                var groupSurfaces = group.ToList();
                
                // Find related links for this container group
                var relatedLinks = links.Where(l => 
                    groupSurfaces.Any(s => 
                        l.ParentIndex == (uint)s.BoundsCenterX ||
                        l.ParentIndex == (uint)s.BoundsCenterY ||
                        l.ParentIndex == (uint)s.BoundsCenterZ ||
                        l.MspiFirstIndex == s.MsviFirstIndex
                    )
                ).ToList();
                
                // Find related placements for this container group
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
                
                // Calculate actual spatial bounds from vertices (simplified query to avoid EF translation issues)
                var minIndex = groupSurfaces.Min(s => s.MsviFirstIndex);
                var maxIndex = groupSurfaces.Max(s => s.MsviFirstIndex + s.IndexCount);
                
                var vertices = await context.Vertices
                    .Where(v => v.Pm4FileId == pm4FileId && 
                                v.GlobalIndex >= minIndex && 
                                v.GlobalIndex < maxIndex)
                    .ToListAsync();
                
                // Filter vertices to exact surface ranges on client side
                var relevantVertices = vertices.Where(v => 
                    groupSurfaces.Any(s => v.GlobalIndex >= s.MsviFirstIndex && 
                                          v.GlobalIndex < s.MsviFirstIndex + s.IndexCount)
                ).ToList();
                
                var actualBounds = CalculateActualSpatialBounds(relevantVertices);
                
                // Classify object type based on size and completeness
                var objectType = ClassifyObjectType(totalTriangles, completenessScore);
                var isCompleteObject = IsCompleteObject(totalTriangles, completenessScore);
                
                // Create hierarchical container
                var container = new Pm4HierarchicalContainer
                {
                    Pm4FileId = pm4FileId,
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
                
                containersToInsert.Add(container);
                
                // Create member records for surfaces
                foreach (var surface in groupSurfaces)
                {
                    membersToInsert.Add(new Pm4HierarchicalContainerMember
                    {
                        MemberType = "Surface",
                        MemberId = surface.Id,
                        TriangleContribution = surface.IndexCount
                    });
                }
                
                // Create member records for links
                foreach (var link in relatedLinks)
                {
                    membersToInsert.Add(new Pm4HierarchicalContainerMember
                    {
                        MemberType = "Link",
                        MemberId = link.Id,
                        TriangleContribution = 0
                    });
                }
                
                // Create member records for placements
                foreach (var placement in relatedPlacements)
                {
                    membersToInsert.Add(new Pm4HierarchicalContainerMember
                    {
                        MemberType = "Placement",
                        MemberId = placement.Id,
                        TriangleContribution = 0
                    });
                }
            }
            
            // Batch insert containers
            ConsoleLogger.WriteLine($"[HIERARCHICAL EXPORT] Inserting {containersToInsert.Count} hierarchical containers...");
            context.HierarchicalContainers.AddRange(containersToInsert);
            await context.SaveChangesAsync();
            
            // Update member container IDs and batch insert
            var memberIndex = 0;
            foreach (var container in containersToInsert)
            {
                var memberCount = container.SurfaceCount + container.RelatedLinkCount + container.RelatedPlacementCount;
                for (int i = 0; i < memberCount && memberIndex < membersToInsert.Count; i++, memberIndex++)
                {
                    membersToInsert[memberIndex].ContainerId = container.Id;
                }
            }
            
            ConsoleLogger.WriteLine($"[HIERARCHICAL EXPORT] Inserting {membersToInsert.Count} container members...");
            context.HierarchicalContainerMembers.AddRange(membersToInsert);
            await context.SaveChangesAsync();
            
            // Log summary statistics
            var completeObjects = containersToInsert.Count(c => c.IsCompleteObject);
            var largeObjects = containersToInsert.Count(c => c.TotalTriangles > 10000);
            var averageTriangles = containersToInsert.Any() ? containersToInsert.Average(c => c.TotalTriangles) : 0;
            
            ConsoleLogger.WriteLine($"[HIERARCHICAL EXPORT] Hierarchical container export complete:");
            ConsoleLogger.WriteLine($"  Total containers: {containersToInsert.Count}");
            ConsoleLogger.WriteLine($"  Complete objects: {completeObjects}");
            ConsoleLogger.WriteLine($"  Large objects (>10K triangles): {largeObjects}");
            ConsoleLogger.WriteLine($"  Average triangles per container: {averageTriangles:F1}");
        }
        
        /// <summary>
        /// Calculates actual spatial bounds from vertex data (not encoded bounds).
        /// </summary>
        private (Vector3 min, Vector3 max) CalculateActualSpatialBounds(List<Pm4Vertex> vertices)
        {
            if (!vertices.Any())
            {
                return (Vector3.Zero, Vector3.Zero);
            }
            
            var min = new Vector3(
                vertices.Min(v => v.X),
                vertices.Min(v => v.Y),
                vertices.Min(v => v.Z)
            );
            
            var max = new Vector3(
                vertices.Max(v => v.X),
                vertices.Max(v => v.Y),
                vertices.Max(v => v.Z)
            );
            
            return (min, max);
        }
        
        /// <summary>
        /// Classifies object type based on triangle count and completeness.
        /// </summary>
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
        
        /// <summary>
        /// Determines if an object is considered complete based on size and relationships.
        /// </summary>
        private bool IsCompleteObject(int triangleCount, int completenessScore)
        {
            return completenessScore >= 2 && triangleCount >= 1000;
        }
        
        /// <summary>
        /// Fast JSON-based hierarchical container export that bypasses SQLite performance bottlenecks.
        /// </summary>
        private async Task ExportHierarchicalContainersJsonAsync(Pm4Scene scene, string fileName, string filePath)
        {
            ConsoleLogger.WriteLine("[JSON HIERARCHICAL EXPORT] Starting fast JSON-based hierarchical container export...");
            
            // Convert scene data to database models for consistency using existing extraction logic
            var surfaces = scene.Surfaces.Select((s, index) => 
            {
                var extractedFields = ExtractMsurEntryFields(s);
                return new Pm4Surface
                {
                    Id = index + 1,
                    BoundsCenterX = (float)extractedFields.RawFields.GetValueOrDefault("BoundsCenterX", 0f),
                    BoundsCenterY = (float)extractedFields.RawFields.GetValueOrDefault("BoundsCenterY", 0f),
                    BoundsCenterZ = (float)extractedFields.RawFields.GetValueOrDefault("BoundsCenterZ", 0f),
                    MsviFirstIndex = (int)extractedFields.MsviFirstIndex,
                    IndexCount = extractedFields.IndexCount,
                    GroupKey = extractedFields.GroupKey
                };
            }).ToList();
            
            var links = scene.Links?.Select((l, index) => new Pm4Link
            {
                Id = index + 1,
                ParentIndex = (uint)(l.GetType().GetProperty("ParentIndex")?.GetValue(l) ?? 0u),
                MspiFirstIndex = (int)(l.GetType().GetProperty("MspiFirstIndex")?.GetValue(l) ?? 0u)
            }).ToList() ?? new List<Pm4Link>();
            
            var placements = scene.Placements?.Select((p, index) => new Pm4Placement
            {
                Id = index + 1,
                Unknown4 = Convert.ToUInt32(p.GetType().GetProperty("Unknown4")?.GetValue(p) ?? 0),
                Unknown6 = Convert.ToUInt32(p.GetType().GetProperty("Unknown6")?.GetValue(p) ?? 0),
                PositionX = Convert.ToSingle(p.GetType().GetProperty("X")?.GetValue(p) ?? 0f),
                PositionY = Convert.ToSingle(p.GetType().GetProperty("Y")?.GetValue(p) ?? 0f),
                PositionZ = Convert.ToSingle(p.GetType().GetProperty("Z")?.GetValue(p) ?? 0f)
            }).ToList() ?? new List<Pm4Placement>();
            
            var vertices = scene.Vertices.Select((v, index) => new Pm4Vertex
            {
                Id = index + 1,
                GlobalIndex = index,
                X = v.X,
                Y = v.Y,
                Z = v.Z
            }).ToList();
            
            // Create output file path
            var outputDir = Path.GetDirectoryName(filePath) ?? Path.GetDirectoryName(fileName) ?? "./";
            var outputFileName = Path.GetFileNameWithoutExtension(fileName) + "_hierarchical_containers.json";
            var outputPath = Path.Combine(outputDir, outputFileName);
            
            // Use fast JSON exporter
            var jsonExporter = new Pm4HierarchicalJsonExporter();
            var result = await jsonExporter.ExportAsync(
                fileName,
                surfaces,
                links,
                placements,
                vertices,
                outputPath
            );
            
            ConsoleLogger.WriteLine($"[JSON HIERARCHICAL EXPORT] Successfully exported {result.TotalContainers} hierarchical containers to: {outputPath}");
        }
        
        /// <summary>
        /// Exports MPRR properties to the database with optimized batch processing.
        /// MPRR contains the definitive building boundaries using sentinel values (Value1=65535).
        /// </summary>
        private async Task ExportPropertiesAsync(Pm4DatabaseContext context, int pm4FileId, List<object> properties)
        {
            ConsoleLogger.WriteLine($"[CHUNK PROCESSING] Starting MPRR property export: {properties.Count:N0} entries");
            
            if (properties.Count == 0)
            {
                ConsoleLogger.WriteLine("[MPRR EXPORT] No MPRR properties found in scene data");
                return;
            }
            
            // Debug: Log first property structure
            if (properties.Count > 0)
            {
                LogChunkStructure("MPRR", properties[0]);
            }
            
            var startTime = DateTime.Now;
            const int batchSize = 10000;
            int sentinelCount = 0;
            
            // Disable change tracking for performance
            context.ChangeTracker.AutoDetectChangesEnabled = false;
            
            for (int i = 0; i < properties.Count; i += batchSize)
            {
                var batchStartTime = DateTime.Now;
                var endIndex = Math.Min(i + batchSize, properties.Count);
                var batch = new List<Pm4Property>();
                
                for (int j = i; j < endIndex; j++)
                {
                    var property = properties[j];
                    
                    try
                    {
                        // Extract MPRR Entry fields using reflection
                        var value1 = (ushort)(property.GetType().GetProperty("Value1")?.GetValue(property) ?? 0);
                        var value2 = (ushort)(property.GetType().GetProperty("Value2")?.GetValue(property) ?? 0);
                        
                        bool isSentinel = value1 == 65535;
                        if (isSentinel) sentinelCount++;
                        
                        var pm4Property = new Pm4Property
                        {
                            Pm4FileId = pm4FileId,
                            GlobalIndex = j,
                            Value1 = value1,
                            Value2 = value2,
                            IsBoundarySentinel = isSentinel
                        };
                        
                        batch.Add(pm4Property);
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[MPRR EXPORT] Error processing property {j}: {ex.Message}");
                    }
                }
                
                context.Properties.AddRange(batch);
                await context.SaveChangesAsync();
                context.ChangeTracker.Clear();
                
                var batchTime = DateTime.Now - batchStartTime;
                var totalTime = DateTime.Now - startTime;
                var progress = (double)endIndex / properties.Count;
                var eta = progress > 0 ? TimeSpan.FromMilliseconds(totalTime.TotalMilliseconds / progress - totalTime.TotalMilliseconds) : TimeSpan.Zero;
                
                ConsoleLogger.WriteLine($"  Properties: {endIndex:N0}/{properties.Count:N0} ({progress:P1}) | Batch: {batchTime.TotalSeconds:F1}s | Total: {totalTime:mm\\:ss} | ETA: {eta:mm\\:ss}");
            }
            
            // Re-enable change tracking
            context.ChangeTracker.AutoDetectChangesEnabled = true;
            
            var finalTime = DateTime.Now - startTime;
            ConsoleLogger.WriteLine($"Completed MPRR property export: {properties.Count:N0} properties in {finalTime:mm\\:ss}");
            ConsoleLogger.WriteLine($"[MPRR ANALYSIS] Found {sentinelCount:N0} building boundary sentinels (Value1=65535)");
            ConsoleLogger.WriteLine($"[MPRR ANALYSIS] Estimated {sentinelCount + 1:N0} building objects from sentinel boundaries");
        }
        
        #endregion
    }
}
