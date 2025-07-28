using System.Text.Json;
using ParpToolbox.Utils;
using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Comprehensive JSON-based PM4 export pipeline that replaces SQLite database approach.
    /// Exports all PM4 data as structured JSON files for maximum performance and flexibility.
    /// </summary>
    public class Pm4JsonExportPipeline
    {
        public class Pm4JsonExport
        {
            public Pm4FileInfo FileInfo { get; set; } = new();
            public List<Pm4VertexData> Vertices { get; set; } = new();
            public List<Pm4TriangleData> Triangles { get; set; } = new();
            public List<Pm4SurfaceData> Surfaces { get; set; } = new();
            public List<Pm4LinkData> Links { get; set; } = new();
            public List<Pm4PlacementData> Placements { get; set; } = new();
            public List<Pm4HierarchicalContainerData> HierarchicalContainers { get; set; } = new();
            public Dictionary<string, byte[]> RawChunks { get; set; } = new();
            public Pm4ExportMetrics Metrics { get; set; } = new();
        }
        
        public class Pm4FileInfo
        {
            public string FileName { get; set; } = "";
            public string FilePath { get; set; } = "";
            public DateTime ProcessedAt { get; set; }
            public int TotalVertices { get; set; }
            public int TotalTriangles { get; set; }
            public int TotalSurfaces { get; set; }
            public int TotalLinks { get; set; }
            public int TotalPlacements { get; set; }
        }
        
        public class Pm4VertexData
        {
            public int Id { get; set; }
            public int GlobalIndex { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }
        }
        
        public class Pm4TriangleData
        {
            public int Id { get; set; }
            public int GlobalIndex { get; set; }
            public int VertexA { get; set; }
            public int VertexB { get; set; }
            public int VertexC { get; set; }
        }
        
        public class Pm4SurfaceData
        {
            public int Id { get; set; }
            public int GlobalIndex { get; set; }
            
            // REAL FIELDS ONLY (based on wowdev.wiki PD4.md official documentation)
            // struct { uint8_t _0x00; uint8_t _0x01; uint8_t _0x02; uint8_t _0x03; 
            //          float _0x04; float _0x08; float _0x0c; float _0x10; 
            //          uint32_t MSVI_first_index; uint32_t _0x18; uint32_t _0x1c; }
            
            public byte Flags { get; set; }                    // _0x00: flags (bitmask32)
            public byte IndexCount { get; set; }               // _0x01: count of indices in MSVI
            public byte Unknown02 { get; set; }                // _0x02: unknown
            public byte Padding03 { get; set; }                // _0x03: Always 0, padding
            
            public float UnknownFloat04 { get; set; }          // _0x04: Unknown float
            public float UnknownFloat08 { get; set; }          // _0x08: Unknown float  
            public float UnknownFloat0C { get; set; }          // _0x0c: Unknown float
            public float UnknownFloat10 { get; set; }          // _0x10: Unknown float
            
            public uint MsviFirstIndex { get; set; }           // MSVI_first_index: Vertex index start
            public uint Unknown18 { get; set; }                // _0x18: Unknown uint32
            public uint Unknown1C { get; set; }                // _0x1c: Unknown uint32
            
            // Raw properties extracted from chunk (for debugging/analysis)
            public Dictionary<string, object> RawProperties { get; set; } = new();
        }
        
        public class Pm4LinkData
        {
            public int Id { get; set; }
            public int GlobalIndex { get; set; }
            public uint ParentIndex { get; set; }
            public int MspiFirstIndex { get; set; }
            public Dictionary<string, object> Properties { get; set; } = new();
        }
        
        public class Pm4PlacementData
        {
            public int Id { get; set; }
            public int GlobalIndex { get; set; }
            public float PositionX { get; set; }
            public float PositionY { get; set; }
            public float PositionZ { get; set; }
            public uint Unknown4 { get; set; }
            public uint Unknown6 { get; set; }
            public Dictionary<string, object> Properties { get; set; } = new();
        }
        
        public class Pm4HierarchicalContainerData
        {
            public int Id { get; set; }
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
            
            // Actual spatial bounds
            public float ActualBoundsMinX { get; set; }
            public float ActualBoundsMinY { get; set; }
            public float ActualBoundsMinZ { get; set; }
            public float ActualBoundsMaxX { get; set; }
            public float ActualBoundsMaxY { get; set; }
            public float ActualBoundsMaxZ { get; set; }
            
            public List<Pm4ContainerMemberData> Members { get; set; } = new();
        }
        
        public class Pm4ContainerMemberData
        {
            public string MemberType { get; set; } = ""; // "Surface", "Link", "Placement"
            public int MemberId { get; set; }
            public int TriangleContribution { get; set; }
            public Dictionary<string, object> Properties { get; set; } = new();
        }
        
        public class Pm4ExportMetrics
        {
            public DateTime ExportStartTime { get; set; }
            public DateTime ExportEndTime { get; set; }
            public TimeSpan TotalExportTime { get; set; }
            public long MemoryUsageMB { get; set; }
            public int CompleteObjects { get; set; }
            public int LargeObjects { get; set; }
            public double AverageTrianglesPerContainer { get; set; }
            public Dictionary<string, object> AdditionalMetrics { get; set; } = new();
        }
        
        /// <summary>
        /// Exports complete PM4 scene as comprehensive JSON data.
        /// </summary>
        public async Task<string> ExportSceneAsync(
            Pm4Scene scene, 
            string fileName, 
            string filePath,
            IReadOnlyDictionary<string, byte[]>? capturedRawData = null,
            string? outputDirectory = null)
        {
            var startTime = DateTime.UtcNow;
            var memoryBefore = GC.GetTotalMemory(false);
            
            ConsoleLogger.WriteLine($"[JSON EXPORT] Starting comprehensive JSON export for: {fileName}");
            ConsoleLogger.WriteLine($"[JSON EXPORT] Scene data: {scene.Vertices.Count} vertices, {scene.Triangles.Count} triangles, {scene.Surfaces.Count} surfaces");
            
            // Create export object
            var export = new Pm4JsonExport
            {
                FileInfo = new Pm4FileInfo
                {
                    FileName = fileName,
                    FilePath = filePath,
                    ProcessedAt = startTime,
                    TotalVertices = scene.Vertices.Count,
                    TotalTriangles = scene.Triangles.Count,
                    TotalSurfaces = scene.Surfaces.Count,
                    TotalLinks = scene.Links?.Count ?? 0,
                    TotalPlacements = scene.Placements?.Count ?? 0
                }
            };
            
            // Export vertices
            export.Vertices = scene.Vertices.Select((v, index) => new Pm4VertexData
            {
                Id = index + 1,
                GlobalIndex = index,
                X = v.X,
                Y = v.Y,
                Z = v.Z
            }).ToList();
            
            // Export triangles
            export.Triangles = scene.Triangles.Select((t, index) => new Pm4TriangleData
            {
                Id = index + 1,
                GlobalIndex = index,
                VertexA = t.A,
                VertexB = t.B,
                VertexC = t.C
            }).ToList();
            
            ConsoleLogger.WriteLine($"[JSON EXPORT] Exported {export.Vertices.Count} vertices and {export.Triangles.Count} triangles");
            
            // Export surfaces with field extraction
            export.Surfaces = await ExportSurfacesAsync(scene.Surfaces?.Cast<object>().ToList() ?? new List<object>());
            ConsoleLogger.WriteLine($"[JSON EXPORT] Exported {export.Surfaces.Count} surfaces");
            
            // Export links
            export.Links = await ExportLinksAsync(scene.Links?.Cast<object>().ToList() ?? new List<object>());
            ConsoleLogger.WriteLine($"[JSON EXPORT] Exported {export.Links.Count} links");
            
            // Export placements
            export.Placements = await ExportPlacementsAsync(scene.Placements?.Cast<object>().ToList() ?? new List<object>());
            ConsoleLogger.WriteLine($"[JSON EXPORT] Exported {export.Placements.Count} placements");
            
            // Export hierarchical containers (fast in-memory processing)
            export.HierarchicalContainers = await ExportHierarchicalContainersAsync(
                export.Surfaces, export.Links, export.Placements, export.Vertices);
            ConsoleLogger.WriteLine($"[JSON EXPORT] Exported {export.HierarchicalContainers.Count} hierarchical containers");
            
            // Skip raw chunks from JSON export to prevent OutOfMemoryException
            // Raw binary data is not suitable for JSON serialization
            if (capturedRawData != null)
            {
                ConsoleLogger.WriteLine($"[JSON EXPORT] Skipping {capturedRawData.Count} raw chunks from JSON (binary data not suitable for JSON)");
                // export.RawChunks = capturedRawData.ToDictionary(kvp => kvp.Key, kvp => kvp.Value); // DISABLED
            }
            
            // Calculate metrics
            var endTime = DateTime.UtcNow;
            var memoryAfter = GC.GetTotalMemory(false);
            
            export.Metrics = new Pm4ExportMetrics
            {
                ExportStartTime = startTime,
                ExportEndTime = endTime,
                TotalExportTime = endTime - startTime,
                MemoryUsageMB = (memoryAfter - memoryBefore) / (1024 * 1024),
                CompleteObjects = export.HierarchicalContainers.Count(c => c.IsCompleteObject),
                LargeObjects = export.HierarchicalContainers.Count(c => c.TotalTriangles > 10000),
                AverageTrianglesPerContainer = export.HierarchicalContainers.Any() 
                    ? export.HierarchicalContainers.Average(c => c.TotalTriangles) : 0
            };
            
            // Determine output directory
            var outputDir = outputDirectory ?? Path.GetDirectoryName(filePath) ?? "./";
            var baseFileName = Path.GetFileNameWithoutExtension(fileName);
            
            // Export to separate JSON files with memory-optimized settings
            var jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,   // Make individual files readable
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
                Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
            };
            
            ConsoleLogger.WriteLine($"[JSON EXPORT] Exporting to split JSON files in: {outputDir}");
            
            // Export each data type to separate files for easier analysis
            var exportTasks = new List<Task<string>>();
            
            // 1. Vertices
            exportTasks.Add(ExportDataTypeAsync(export.Vertices, Path.Combine(outputDir, $"{baseFileName}_vertices.json"), jsonOptions, "vertices"));
            
            // 2. Triangles  
            exportTasks.Add(ExportDataTypeAsync(export.Triangles, Path.Combine(outputDir, $"{baseFileName}_triangles.json"), jsonOptions, "triangles"));
            
            // 3. Surfaces (MOST IMPORTANT for chunk validation)
            exportTasks.Add(ExportDataTypeAsync(export.Surfaces, Path.Combine(outputDir, $"{baseFileName}_surfaces.json"), jsonOptions, "surfaces"));
            
            // 4. Links
            exportTasks.Add(ExportDataTypeAsync(export.Links, Path.Combine(outputDir, $"{baseFileName}_links.json"), jsonOptions, "links"));
            
            // 5. Placements
            exportTasks.Add(ExportDataTypeAsync(export.Placements, Path.Combine(outputDir, $"{baseFileName}_placements.json"), jsonOptions, "placements"));
            
            // 6. Hierarchical Containers (if any)
            if (export.HierarchicalContainers.Any())
            {
                exportTasks.Add(ExportDataTypeAsync(export.HierarchicalContainers, Path.Combine(outputDir, $"{baseFileName}_containers.json"), jsonOptions, "containers"));
            }
            
            // 7. File Info and Metrics
            var summary = new { export.FileInfo, export.Metrics };
            exportTasks.Add(ExportDataTypeAsync(summary, Path.Combine(outputDir, $"{baseFileName}_summary.json"), jsonOptions, "summary"));
            
            // Wait for all exports to complete
            var exportedFiles = await Task.WhenAll(exportTasks);
            var outputPath = outputDir; // Return directory instead of single file
            
            // Log completion summary
            ConsoleLogger.WriteLine($"[JSON EXPORT] Export complete!");
            ConsoleLogger.WriteLine($"  File: {outputPath}");
            ConsoleLogger.WriteLine($"  Duration: {export.Metrics.TotalExportTime:hh\\:mm\\:ss}");
            ConsoleLogger.WriteLine($"  Memory used: {export.Metrics.MemoryUsageMB} MB");
            ConsoleLogger.WriteLine($"  Complete objects: {export.Metrics.CompleteObjects}");
            ConsoleLogger.WriteLine($"  Large objects: {export.Metrics.LargeObjects}");
            ConsoleLogger.WriteLine($"  Avg triangles/container: {export.Metrics.AverageTrianglesPerContainer:F1}");
            
            return outputPath;
        }
        
        /// <summary>
        /// Helper method to export a specific data type to its own JSON file
        /// </summary>
        private async Task<string> ExportDataTypeAsync<T>(T data, string filePath, JsonSerializerOptions jsonOptions, string dataTypeName)
        {
            try
            {
                ConsoleLogger.WriteLine($"[JSON EXPORT] Writing {dataTypeName} to: {Path.GetFileName(filePath)}");
                
                var jsonContent = JsonSerializer.Serialize(data, jsonOptions);
                await File.WriteAllTextAsync(filePath, jsonContent);
                
                var fileInfo = new FileInfo(filePath);
                ConsoleLogger.WriteLine($"[JSON EXPORT] {dataTypeName} exported: {fileInfo.Length / 1024 / 1024:F1} MB");
                
                return filePath;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"[JSON EXPORT] Error exporting {dataTypeName}: {ex.Message}");
                throw;
            }
        }
        
        private Task<List<Pm4SurfaceData>> ExportSurfacesAsync(List<object> surfaces)
        {
            return Task.FromResult(surfaces.Select((s, index) =>
            {
                var surfaceData = new Pm4SurfaceData
                {
                    Id = index + 1,
                    GlobalIndex = index
                };
                
                // Extract ONLY real fields that exist in official MSUR documentation
                // Official structure: { uint8_t _0x00; uint8_t _0x01; uint8_t _0x02; uint8_t _0x03;
                //                       float _0x04; float _0x08; float _0x0c; float _0x10;
                //                       uint32_t MSVI_first_index; uint32_t _0x18; uint32_t _0x1c; }
                
                var type = s.GetType();
                var properties = type.GetProperties();
                
                foreach (var prop in properties)
                {
                    try
                    {
                        var value = prop.GetValue(s);
                        
                        // Map only to REAL fields using EXACT property names from MSUR Entry objects
                        switch (prop.Name)
                        {
                            // Real field: uint8_t _0x00 (flags) - CORRECT property name
                            case "FlagsOrUnknown_0x00":
                            case "SurfaceGroupKey":  // Alias for FlagsOrUnknown_0x00
                                surfaceData.Flags = Convert.ToByte(value ?? 0);
                                break;
                                
                            // Real field: uint8_t _0x01 (count of indices in MSVI)
                            case "IndexCount":
                                surfaceData.IndexCount = Convert.ToByte(value ?? 0);
                                break;
                                
                            // Real field: uint8_t _0x02 (unknown)
                            case "Unknown_0x02":
                                surfaceData.Unknown02 = Convert.ToByte(value ?? 0);
                                break;
                                
                            // Real field: uint8_t _0x03 (padding)
                            case "Padding_0x03":
                                surfaceData.Padding03 = Convert.ToByte(value ?? 0);
                                break;
                                
                            // Real field: float _0x04 (Normal X)
                            case "Nx":
                                surfaceData.UnknownFloat04 = Convert.ToSingle(value ?? 0f);
                                break;
                                
                            // Real field: float _0x08 (Normal Y)
                            case "Ny":
                                surfaceData.UnknownFloat08 = Convert.ToSingle(value ?? 0f);
                                break;
                                
                            // Real field: float _0x0c (Normal Z)
                            case "Nz":
                                surfaceData.UnknownFloat0C = Convert.ToSingle(value ?? 0f);
                                break;
                                
                            // Real field: float _0x10 (Height/Plane D)
                            case "Height":
                                surfaceData.UnknownFloat10 = Convert.ToSingle(value ?? 0f);
                                break;
                                
                            // Real field: uint32_t MSVI_first_index (vertex index start)
                            case "MsviFirstIndex":
                                surfaceData.MsviFirstIndex = Convert.ToUInt32(value ?? 0u);
                                break;
                                
                            // Real field: uint32_t _0x18 (MDOS index)
                            case "MdosIndex":
                                surfaceData.Unknown18 = Convert.ToUInt32(value ?? 0u);
                                break;
                                
                            // Real field: uint32_t _0x1c (Packed params)
                            case "PackedParams":
                                surfaceData.Unknown1C = Convert.ToUInt32(value ?? 0u);
                                break;
                                
                            // IGNORE FABRICATED FIELDS - DO NOT EXPORT
                            case "BoundsCenterX":
                            case "BoundsCenterY":
                            case "BoundsCenterZ":
                            case "BoundsMinX":
                            case "BoundsMinY":
                            case "BoundsMinZ":
                            case "BoundsMaxX":
                            case "BoundsMaxY":
                            case "BoundsMaxZ":
                                // FABRICATED - DO NOT EXPORT
                                ConsoleLogger.WriteLine($"[JSON EXPORT] IGNORING fabricated field: {prop.Name}");
                                break;
                                
                            default:
                                // Store unrecognized properties for debugging
                                if (value != null)
                                {
                                    surfaceData.RawProperties[prop.Name] = value;
                                }
                                break;
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[JSON EXPORT] Warning: Failed to extract surface property {prop.Name}: {ex.Message}");
                    }
                }
                
                return surfaceData;
            }).ToList());
        }
        
        private Task<List<Pm4LinkData>> ExportLinksAsync(List<object> links)
        {
            return Task.FromResult(links.Select((l, index) =>
            {
                var linkData = new Pm4LinkData
                {
                    Id = index + 1,
                    GlobalIndex = index
                };
                
                // Extract link properties safely
                var type = l.GetType();
                var properties = type.GetProperties();
                
                foreach (var prop in properties)
                {
                    try
                    {
                        var value = prop.GetValue(l);
                        switch (prop.Name)
                        {
                            case "ParentIndex":
                                linkData.ParentIndex = Convert.ToUInt32(value ?? 0u);
                                break;
                            case "MspiFirstIndex":
                                linkData.MspiFirstIndex = Convert.ToInt32(value ?? 0);
                                break;
                            default:
                                if (value != null)
                                {
                                    linkData.Properties[prop.Name] = value;
                                }
                                break;
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[JSON EXPORT] Warning: Failed to extract link property {prop.Name}: {ex.Message}");
                    }
                }
                
                return linkData;
            }).ToList());
        }
        
        private Task<List<Pm4PlacementData>> ExportPlacementsAsync(List<object> placements)
        {
            return Task.FromResult(placements.Select((p, index) =>
            {
                var placementData = new Pm4PlacementData
                {
                    Id = index + 1,
                    GlobalIndex = index
                };
                
                // Extract placement properties safely
                var type = p.GetType();
                var properties = type.GetProperties();
                
                foreach (var prop in properties)
                {
                    try
                    {
                        var value = prop.GetValue(p);
                        switch (prop.Name)
                        {
                            case "X":
                                placementData.PositionX = Convert.ToSingle(value ?? 0f);
                                break;
                            case "Y":
                                placementData.PositionY = Convert.ToSingle(value ?? 0f);
                                break;
                            case "Z":
                                placementData.PositionZ = Convert.ToSingle(value ?? 0f);
                                break;
                            case "Unknown4":
                                placementData.Unknown4 = Convert.ToUInt32(value ?? 0u);
                                break;
                            case "Unknown6":
                                placementData.Unknown6 = Convert.ToUInt32(value ?? 0u);
                                break;
                            default:
                                if (value != null)
                                {
                                    placementData.Properties[prop.Name] = value;
                                }
                                break;
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"[JSON EXPORT] Warning: Failed to extract placement property {prop.Name}: {ex.Message}");
                    }
                }
                
                return placementData;
            }).ToList());
        }
        
        private Task<List<Pm4HierarchicalContainerData>> ExportHierarchicalContainersAsync(
            List<Pm4SurfaceData> surfaces,
            List<Pm4LinkData> links, 
            List<Pm4PlacementData> placements,
            List<Pm4VertexData> vertices)
        {
            ConsoleLogger.WriteLine($"[JSON EXPORT] Processing hierarchical containers from {surfaces.Count} surfaces...");
            
            ConsoleLogger.WriteLine($"[JSON EXPORT] CRITICAL DISCOVERY: BoundsCenterX/Y/Z fields are not present in raw PM4 data!");
            ConsoleLogger.WriteLine($"[JSON EXPORT] Redesigning grouping logic using ONLY validated, real fields from raw chunks.");
            
            // Real-field-based hierarchical grouping (no fabricated bounds fields)
            // Flags = Surface type/category (real field: _0x00 from official documentation)
            // MsviFirstIndex ranges = Vertex index ranges (geometric proximity)
            var containerGroups = surfaces
                .GroupBy(s => new { 
                    s.Flags,  // Real field: _0x00 (surface type/category flags)
                    VertexRange = s.MsviFirstIndex / 1000  // Geometric proximity grouping
                })
                .Where(g => g.Count() >= 1)  // At least 1 surface per group
                .ToList();
            
            ConsoleLogger.WriteLine($"[JSON EXPORT] Found {containerGroups.Count} surface groups using real fields (Flags + VertexRange)");
            
            // Diagnostic: Check group distribution
            var groupSizes = containerGroups.Select(g => g.Count()).OrderByDescending(x => x).ToList();
            ConsoleLogger.WriteLine($"[JSON EXPORT] Group size distribution: Largest={groupSizes.FirstOrDefault()}, Smallest={groupSizes.LastOrDefault()}, Average={groupSizes.Average():F1}");
            
            // EMERGENCY BYPASS: If we still have a single massive group, skip hierarchical processing
            if (containerGroups.Count == 1 && containerGroups[0].Count() > 100000)
            {
                ConsoleLogger.WriteLine($"[JSON EXPORT] EMERGENCY BYPASS: Still detecting single massive group ({containerGroups[0].Count()} surfaces).");
                ConsoleLogger.WriteLine($"[JSON EXPORT] This indicates all surfaces have identical GroupKey and VertexRange values.");
                ConsoleLogger.WriteLine($"[JSON EXPORT] SKIPPING hierarchical container processing to prevent hang.");
                ConsoleLogger.WriteLine($"[JSON EXPORT] PackedParams analysis needed to find proper grouping fields.");
                
                // Return empty containers list to complete export without hang
                return Task.FromResult(new List<Pm4HierarchicalContainerData>());
            }
            
            // Limit processing to prevent hangs (process top groups by triangle count)
            var maxGroups = 10000;  // Reasonable limit
            var processableGroups = containerGroups
                .OrderByDescending(g => g.Sum(s => s.IndexCount))  // Order by triangle count
                .Take(maxGroups)
                .ToList();
                
            if (processableGroups.Count < containerGroups.Count)
            {
                ConsoleLogger.WriteLine($"[JSON EXPORT] Processing top {processableGroups.Count} groups (by triangle count) to prevent performance issues.");
            }
            
            var containers = new List<Pm4HierarchicalContainerData>();
            var processedCount = 0;
            
            foreach (var group in processableGroups)
            {
                var groupSurfaces = group.ToList();
                
                // Find related links using REAL fields only
                var relatedLinks = links.Where(l => 
                    groupSurfaces.Any(s => 
                        l.MspiFirstIndex == s.MsviFirstIndex ||  // Direct index match
                        (l.MspiFirstIndex >= s.MsviFirstIndex && l.MspiFirstIndex < s.MsviFirstIndex + s.IndexCount)  // Within surface range
                    )
                ).ToList();
                
                // Find related placements using REAL fields only  
                var relatedPlacements = placements.Where(p => 
                    groupSurfaces.Any(s => 
                        p.Unknown4 == (uint)s.Flags ||  // Group type match (using real field)
                        (p.Unknown4 >= s.MsviFirstIndex && p.Unknown4 < s.MsviFirstIndex + s.IndexCount)  // Index range match
                    )
                ).ToList();
                
                // Calculate metrics
                var totalTriangles = groupSurfaces.Sum(s => s.IndexCount);
                var completenessScore = (groupSurfaces.Count > 0 ? 1 : 0) + 
                                       (relatedLinks.Count > 0 ? 1 : 0) + 
                                       (relatedPlacements.Count > 0 ? 1 : 0);
                
                // Calculate actual spatial bounds from vertices
                var minIndex = groupSurfaces.Min(s => s.MsviFirstIndex);
                var maxIndex = groupSurfaces.Max(s => s.MsviFirstIndex + s.IndexCount);
                
                var relevantVertices = vertices.Where(v => 
                    v.GlobalIndex >= minIndex && v.GlobalIndex < maxIndex &&
                    groupSurfaces.Any(s => v.GlobalIndex >= s.MsviFirstIndex && 
                                          v.GlobalIndex < s.MsviFirstIndex + s.IndexCount)
                ).ToList();
                
                var actualBounds = CalculateActualBounds(relevantVertices);
                
                // Create container using REAL field identifiers
                var container = new Pm4HierarchicalContainerData
                {
                    Id = processedCount + 1,
                    ContainerX = group.Key.Flags,  // Use Flags as primary identifier (real field)
                    ContainerY = group.Key.VertexRange,  // Use VertexRange as secondary identifier  
                    ContainerZ = 0,  // No third identifier needed
                    SurfaceCount = groupSurfaces.Count,
                    TotalTriangles = totalTriangles,
                    RelatedLinkCount = relatedLinks.Count,
                    RelatedPlacementCount = relatedPlacements.Count,
                    CompletenessScore = completenessScore,
                    ObjectType = ClassifyObjectType(totalTriangles, completenessScore),
                    IsCompleteObject = IsCompleteObject(totalTriangles, completenessScore),
                    ActualBoundsMinX = actualBounds.min.X,
                    ActualBoundsMinY = actualBounds.min.Y,
                    ActualBoundsMinZ = actualBounds.min.Z,
                    ActualBoundsMaxX = actualBounds.max.X,
                    ActualBoundsMaxY = actualBounds.max.Y,
                    ActualBoundsMaxZ = actualBounds.max.Z
                };
                
                // Add members
                foreach (var surface in groupSurfaces)
                {
                    container.Members.Add(new Pm4ContainerMemberData
                    {
                        MemberType = "Surface",
                        MemberId = surface.Id,
                        TriangleContribution = surface.IndexCount,
                        Properties = surface.RawProperties  // Updated to use RawProperties
                    });
                }
                
                foreach (var link in relatedLinks)
                {
                    container.Members.Add(new Pm4ContainerMemberData
                    {
                        MemberType = "Link",
                        MemberId = link.Id,
                        TriangleContribution = 0,
                        Properties = link.Properties
                    });
                }
                
                foreach (var placement in relatedPlacements)
                {
                    container.Members.Add(new Pm4ContainerMemberData
                    {
                        MemberType = "Placement",
                        MemberId = placement.Id,
                        TriangleContribution = 0,
                        Properties = placement.Properties
                    });
                }
                
                containers.Add(container);
                processedCount++;
                
                // Progress reporting
                if (processedCount % 1000 == 0)
                {
                    ConsoleLogger.WriteLine($"[JSON EXPORT] Processed {processedCount}/{processableGroups.Count} containers...");
                }
            }
            
            return Task.FromResult(containers);
        }
        
        private (Vector3 min, Vector3 max) CalculateActualBounds(List<Pm4VertexData> vertices)
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
