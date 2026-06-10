using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Services.Coordinate;
using System.Globalization;
using ParpToolbox.Utils;
using ParpToolbox.Formats.P4;
using ParpToolbox.Services.WMO;
using ParpToolbox.Services.PM4;
using WoWFormatLib.FileProviders;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command: mscn-wmo-compare <pm4File> <wmoFile> [--tolerance <t>]
    /// Extracts MSCN anchors from PM4 file and compares with WMO geometry to validate spatial correlation.
    /// </summary>
    internal static class MscnWmoComparisonCommand
    {
        public static async Task<int> Run(string[] args)
        {
            if (args.Length < 3)
            {
                ConsoleLogger.WriteLine("Usage: mscn-wmo-compare <pm4File> <wmoFile> [--tolerance <t>] [--multi-tile]");
                ConsoleLogger.WriteLine("Example: mscn-wmo-compare development_15_37.pm4 StormwindHarbor.wmo --tolerance 5.0 --multi-tile");
                ConsoleLogger.WriteLine("  --multi-tile: Automatically load MSCN data from adjacent tiles for comprehensive comparison");
                return 1;
            }

            string pm4FilePath = args[1];
            string wmoFilePath = args[2];
            float tolerance = 5.0f; // Default spatial tolerance
            string? groupFilter = null; // Optional group filter
            bool multiTile = false; // Multi-tile aggregation flag

            // Parse optional arguments
            for (int i = 3; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--tolerance":
                        if (i + 1 < args.Length && float.TryParse(args[i + 1], out float t))
                        {
                            tolerance = t;
                            i++; // Skip next argument
                        }
                        break;
                    case "--group-filter":
                        if (i + 1 < args.Length)
                        {
                            groupFilter = args[i + 1];
                            i++; // Skip next argument
                        }
                        break;
                    case "--multi-tile":
                        multiTile = true;
                        break;
                }
            }

            try
            {
                ConsoleLogger.WriteLine("=== MSCN-WMO Geometry Comparison ===");
                ConsoleLogger.WriteLine($"PM4 File: {pm4FilePath}");
                ConsoleLogger.WriteLine($"WMO File: {wmoFilePath}");
                ConsoleLogger.WriteLine($"Tolerance: {tolerance:F1} units");
                ConsoleLogger.WriteLine();

                // Extract MSCN anchor points from PM4 file(s)
                List<Vector3> mscnVertices;
                if (multiTile)
                {
                    mscnVertices = ExtractMscnAnchorsFromNeighbourTiles(pm4FilePath);
                    ConsoleLogger.WriteLine($"Extracted {mscnVertices.Count} MSCN anchor points from multi-tile aggregation");
                }
                else
                {
                    mscnVertices = ExtractMscnAnchorsFromPm4(pm4FilePath);
                    ConsoleLogger.WriteLine($"Extracted {mscnVertices.Count} MSCN anchor points from single PM4");
                }
                
                // Debug: Show first 10 MSCN vertices
                ConsoleLogger.WriteLine("First 10 MSCN vertices:");
                for (int i = 0; i < Math.Min(10, mscnVertices.Count); i++)
                {
                    var v = mscnVertices[i];
                    ConsoleLogger.WriteLine($"  MSCN[{i}]: ({v.X:F2}, {v.Y:F2}, {v.Z:F2})");
                }

                // Load WMO vertices
                var wmoVertices = LoadWmoVertices(wmoFilePath, groupFilter);
                ConsoleLogger.WriteLine($"Loaded {wmoVertices.Count} WMO vertices");
                
                // Debug: Show first 10 WMO vertices
                ConsoleLogger.WriteLine("First 10 WMO vertices:");
                for (int i = 0; i < Math.Min(10, wmoVertices.Count); i++)
                {
                    var v = wmoVertices[i];
                    ConsoleLogger.WriteLine($"  WMO[{i}]: ({v.X:F2}, {v.Y:F2}, {v.Z:F2})");
                }

                if (mscnVertices.Count == 0 || wmoVertices.Count == 0)
                {
                    ConsoleLogger.WriteLine("Error: No vertices found in one or both files");
                    return 1;
                }

                // Perform object-level comparison analysis
                var correlationResult = AnalyzeObjectCorrelation(mscnVertices, wmoVertices, tolerance, groupFilter);
                
                // Generate comparison report and export visualization objects
                var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"session_{DateTime.Now:yyyyMMdd_HHmmss}");
                Directory.CreateDirectory(outputDir);
                
                // Implement MSCN-to-local coordinate normalization
                ConsoleLogger.WriteLine($"Implementing MSCN-to-local coordinate normalization...");
                var normalizationResults = await PerformMscnNormalizationComparison(mscnVertices, wmoVertices, tolerance, groupFilter, outputDir);
                
                // Export visualization objects for analysis
                await ExportVisualizationObjects(mscnVertices, wmoVertices, outputDir, groupFilter);
                
                // Generate normalization report
                var reportPath = Path.Combine(outputDir, $"mscn_wmo_comparison_{DateTime.Now:yyyyMMdd_HHmmss}.txt");
                await GenerateNormalizationReport(normalizationResults, mscnVertices, wmoVertices, tolerance, reportPath);

                ConsoleLogger.WriteLine($"Comparison complete. Report saved to: {reportPath}");
                return 0;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during comparison: {ex.Message}");
                return 1;
            }
        }

        private static List<Vector3> ExtractMscnAnchorsFromMultipleTiles_Legacy(string pm4FilePath)
        {
            var vertices = new List<Vector3>();
            var rawVertices = new List<Vector3>(); // For raw data export
            var normalizedVertices = new List<Vector3>(); // For normalized data export
            
            try
            {
                // Get directory containing the input PM4 file
                string directoryPath = Path.GetDirectoryName(pm4FilePath) ?? ".";                
                ConsoleLogger.WriteLine($"Loading PM4 tiles from directory: {directoryPath}");
                
                // Use the global tile loader to load all tiles in the directory
                var globalScene = Pm4GlobalTileLoader.LoadRegion(directoryPath, "*.pm4");
                
                ConsoleLogger.WriteLine($"Loaded {globalScene.TotalLoadedTiles} tiles with {globalScene.GlobalVertices.Count} total vertices");
                
                // Convert global scene to standard scene for compatibility
                var unifiedScene = Pm4GlobalTileLoader.ToStandardScene(globalScene);
                
                // Extract MSCN anchors from the unified scene using the same approach as single tile
                // Find MSCN chunks in ExtraChunks
                var mscnChunks = unifiedScene.ExtraChunks
                    .Where(chunk => chunk.GetType().Name.Contains("Mscn"))
                    .ToList();
                
                ConsoleLogger.WriteLine($"Found {mscnChunks.Count} MSCN chunks across all tiles");
                
                // Process the MSCN chunks using the same reflection-based approach as single tile
                int totalVertexCount = 0;
                foreach (var chunk in mscnChunks)
                {
                    ConsoleLogger.WriteLine($"Processing MSCN chunk: {chunk.GetType().Name}");
                    
                    // Use reflection to get vertices from MSCN chunk
                    var verticesProperty = chunk.GetType().GetProperty("Vertices");
                    ConsoleLogger.WriteLine($"Vertices property found: {verticesProperty != null}");
                    
                    if (verticesProperty != null)
                    {
                        var chunkVertices = verticesProperty.GetValue(chunk);
                        
                        if (chunkVertices != null)
                        {
                            // Try to iterate through the collection using reflection
                            var enumerableInterface = chunkVertices.GetType().GetInterface("IEnumerable");
                            if (enumerableInterface != null)
                            {
                                var vertexCount = 0;
                                foreach (var vertex in (System.Collections.IEnumerable)chunkVertices)
                                {
                                    if (vertex != null)
                                    {
                                        // Get vertex properties using reflection
                                        var vertexType = vertex.GetType();
                                        object? xMember = vertexType.GetProperty("X") ?? (object?)vertexType.GetField("X");
                                        object? yMember = vertexType.GetProperty("Y") ?? (object?)vertexType.GetField("Y");
                                        object? zMember = vertexType.GetProperty("Z") ?? (object?)vertexType.GetField("Z");
                                        
                                        if (xMember != null && yMember != null && zMember != null)
                                        {
                                            var xVal = Convert.ToSingle(GetMemberValue(xMember, vertex));
                                            var yVal = Convert.ToSingle(GetMemberValue(yMember, vertex));
                                            var zVal = Convert.ToSingle(GetMemberValue(zMember, vertex));
                                            
                                            // Store raw vertex values for export
                                            var rawVertex = new Vector3(xVal, yVal, zVal);
                                            rawVertices.Add(rawVertex);
                                            
                                            // Create normalized vertices (divide X and Y by 533.33 as suggested)
                                            const float normalizationFactor = 533.33f;
                                            var normalizedVertex = new Vector3(
                                                xVal / normalizationFactor,
                                                yVal / normalizationFactor,
                                                zVal
                                            );
                                            normalizedVertices.Add(normalizedVertex);
                                            
                                            // Apply coordinate transformation using unified service
                                            var transformedVertex = CoordinateTransformationService.ApplyMscnTransformation(xVal, yVal, zVal);
                                            
                                            // Make coordinates relative by subtracting tile offset (17066 for world coordinates)
                                            transformedVertex.X -= 17066.0f;
                                            transformedVertex.Y -= 17066.0f;
                                            
                                            vertices.Add(transformedVertex);
                                            vertexCount++;
                                            totalVertexCount++;
                                        }
                                    }
                                }
                                ConsoleLogger.WriteLine($"Processed {vertexCount} vertices from this chunk");
                            }
                        }
                    }
                }
                
                // Export raw and normalized vertices to OBJ files for visual inspection
                var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"session_{DateTime.Now:yyyyMMdd_HHmmss}");
                Directory.CreateDirectory(outputDir);
                
                // Export raw MSCN vertices
                var rawObjPath = Path.Combine(outputDir, "mscn_raw_multi_tile.obj");
                ExportVerticesAsObj(rawVertices, rawObjPath, "Raw Multi-Tile MSCN Vertices (Server/World Coordinates)").GetAwaiter().GetResult();
                ConsoleLogger.WriteLine($"Exported raw multi-tile MSCN vertices to {rawObjPath}");
                
                // Export normalized MSCN vertices
                var normalizedObjPath = Path.Combine(outputDir, "mscn_normalized_multi_tile.obj");
                ExportVerticesAsObj(normalizedVertices, normalizedObjPath, "Normalized Multi-Tile MSCN Vertices (Divided by 533.33)").GetAwaiter().GetResult();
                ConsoleLogger.WriteLine($"Exported normalized multi-tile MSCN vertices to {normalizedObjPath}");
                
                ConsoleLogger.WriteLine($"Extracted {vertices.Count} MSCN anchor vertices from multiple tiles");
                ConsoleLogger.WriteLine($"Raw vertices: {rawVertices.Count}, Normalized vertices: {normalizedVertices.Count}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error loading PM4 tiles: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            
            return vertices;
        }
        
        public static List<Vector3> ExtractMscnAnchorsFromPm4(string pm4FilePath)
        {
            var vertices = new List<Vector3>();
            var rawVertices = new List<Vector3>(); // Store raw vertices for export
            var normalizedVertices = new List<Vector3>(); // Store normalized vertices for export
            
            try
            {
                ConsoleLogger.WriteLine($"Loading PM4 file: {pm4FilePath}");
                var pm4Adapter = new Pm4Adapter();
                var scene = pm4Adapter.Load(pm4FilePath);
                
                // Find MSCN chunks in ExtraChunks
                var mscnChunks = scene.ExtraChunks
                    .Where(chunk => chunk.GetType().Name.Contains("Mscn"))
                    .ToList();
                
                ConsoleLogger.WriteLine($"Found {mscnChunks.Count} MSCN chunks");
                
                foreach (var chunk in mscnChunks)
                {
                    ConsoleLogger.WriteLine($"Processing MSCN chunk: {chunk.GetType().Name}");
                    
                    // Use reflection to get vertices from MSCN chunk
                    var verticesProperty = chunk.GetType().GetProperty("Vertices");
                    ConsoleLogger.WriteLine($"Vertices property found: {verticesProperty != null}");
                    
                    if (verticesProperty != null)
                    {
                        var chunkVertices = verticesProperty.GetValue(chunk);
                        ConsoleLogger.WriteLine($"Vertices value type: {chunkVertices?.GetType().Name ?? "null"}");
                        
                        if (chunkVertices != null)
                        {
                            // Try to iterate through the collection using reflection
                            var enumerableInterface = chunkVertices.GetType().GetInterface("IEnumerable");
                            if (enumerableInterface != null)
                            {
                                var vertexCount = 0;
                                foreach (var vertex in (System.Collections.IEnumerable)chunkVertices)
                                {
                                    if (vertex != null)
                                    {
                                        // Debug: log the vertex type and available members
                                        var vertexType = vertex.GetType();
                                        if (vertexCount == 0) // Only log for first vertex
                                        {
                                            ConsoleLogger.WriteLine($"Vertex type: {vertexType.FullName}");
                                            var props = vertexType.GetProperties().Select(p => p.Name);
                                            var fields = vertexType.GetFields().Select(f => f.Name);
                                            ConsoleLogger.WriteLine($"Properties: {string.Join(", ", props)}");
                                            ConsoleLogger.WriteLine($"Fields: {string.Join(", ", fields)}");
                                        }
                                        
                                        object? xMember = vertexType.GetProperty("X") ?? (object?)vertexType.GetField("X");
                                        object? yMember = vertexType.GetProperty("Y") ?? (object?)vertexType.GetField("Y");
                                        object? zMember = vertexType.GetProperty("Z") ?? (object?)vertexType.GetField("Z");
                                        
                                        if (xMember != null && yMember != null && zMember != null)
                                        {
                                            var xVal = Convert.ToSingle(GetMemberValue(xMember, vertex));
                                            var yVal = Convert.ToSingle(GetMemberValue(yMember, vertex));
                                            var zVal = Convert.ToSingle(GetMemberValue(zMember, vertex));
                                            
                                            // Store raw vertex values for export
                                            var rawVertex = new Vector3(xVal, yVal, zVal);
                                            rawVertices.Add(rawVertex);
                                            
                                            // Create normalized vertices (divide X and Y by 533.33 as suggested)
                                            const float normalizationFactor = 533.33f;
                                            var normalizedVertex = new Vector3(
                                                xVal / normalizationFactor,
                                                yVal / normalizationFactor,
                                                zVal
                                            );
                                            normalizedVertices.Add(normalizedVertex);
                                            
                                            // Log raw MSCN values before ANY transformation
                                            if (vertexCount < 10)
                                            {
                                                ConsoleLogger.WriteLine($"Raw MSCN[{vertexCount}]: X={xVal:F2}, Y={yVal:F2}, Z={zVal:F2}");
                                                ConsoleLogger.WriteLine($"Normalized MSCN[{vertexCount}]: X={normalizedVertex.X:F2}, Y={normalizedVertex.Y:F2}, Z={normalizedVertex.Z:F2}");
                                            }
                                            
                                            // Apply coordinate transformation using unified service
                                            var transformedVertex = CoordinateTransformationService.ApplyMscnTransformation(xVal, yVal, zVal);
                                            
                                            // Log values after transformation but before tile offset subtraction
                                            if (vertexCount < 10)
                                            {
                                                ConsoleLogger.WriteLine($"Post-transform MSCN[{vertexCount}]: ({transformedVertex.X:F2}, {transformedVertex.Y:F2}, {transformedVertex.Z:F2})");
                                            }
                                            
                                            // Make coordinates relative by subtracting tile offset (17066 for world coordinates)
                                            transformedVertex.X -= 17066.0f;
                                            transformedVertex.Y -= 17066.0f;
                                            
                                            // No longer scaling MSCN vertices - using them directly
                                            if (vertexCount < 10)
                                            {
                                                ConsoleLogger.WriteLine($"MSCN[{vertexCount}] (unscaled): ({transformedVertex.X:F2}, {transformedVertex.Y:F2}, {transformedVertex.Z:F2})");
                                            }
                                            
                                            vertices.Add(transformedVertex);
                                            vertexCount++;
                                        }
                                        else
                                        {
                                            ConsoleLogger.WriteLine($"Could not find X, Y, Z members in vertex type {vertexType.Name}");
                                        }
                                    }
                                }
                                ConsoleLogger.WriteLine($"Processed {vertexCount} vertices from this chunk");
                                
                                // Export raw and normalized vertices to OBJ files for visual inspection
                                var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"session_{DateTime.Now:yyyyMMdd_HHmmss}");
                                Directory.CreateDirectory(outputDir);
                                
                                // Export raw MSCN vertices
                                var rawObjPath = Path.Combine(outputDir, "mscn_raw.obj");
                                ExportVerticesAsObj(rawVertices, rawObjPath, "Raw MSCN Vertices (Server/World Coordinates)").GetAwaiter().GetResult();
                                ConsoleLogger.WriteLine($"Exported raw MSCN vertices to {rawObjPath}");
                                
                                // Export normalized MSCN vertices
                                var normalizedObjPath = Path.Combine(outputDir, "mscn_normalized.obj");
                                ExportVerticesAsObj(normalizedVertices, normalizedObjPath, "Normalized MSCN Vertices (Divided by 533.33)").GetAwaiter().GetResult();
                                ConsoleLogger.WriteLine($"Exported normalized MSCN vertices to {normalizedObjPath}");
                            }
                            else
                            {
                                // Try to access as a collection
                                var countProperty = chunkVertices.GetType().GetProperty("Count");
                                if (countProperty != null)
                                {
                                    var count = countProperty.GetValue(chunkVertices);
                                    ConsoleLogger.WriteLine($"Collection has {count} items but could not enumerate");
                                }
                            }
                        }
                    }
                }
                
                ConsoleLogger.WriteLine($"Extracted {vertices.Count} MSCN anchor vertices");
                ConsoleLogger.WriteLine($"Raw vertices: {rawVertices.Count}, Normalized vertices: {normalizedVertices.Count}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error loading PM4 file: {ex.Message}");
            }
            
            return vertices;
        }
        
        private static List<Vector3> ExtractMscnAnchorsFromMultipleTiles(string basePm4FilePath)
        {
            var allVertices = new List<Vector3>();
            var rawVertices = new List<Vector3>();
            var normalizedVertices = new List<Vector3>();
            
            try
            {
                // Get directory containing the input PM4 file
                string directoryPath = Path.GetDirectoryName(basePm4FilePath) ?? ".";
                ConsoleLogger.WriteLine($"Loading PM4 tiles from directory: {directoryPath}");
                
                // METHOD 1: Use GlobalTileLoader to load all tiles in the directory
                ConsoleLogger.WriteLine("Using GlobalTileLoader to load linked tiles...");
                var globalScene = Pm4GlobalTileLoader.LoadRegion(directoryPath, "*.pm4");
                
                ConsoleLogger.WriteLine($"Loaded {globalScene.TotalLoadedTiles} tiles with {globalScene.GlobalVertices.Count} total vertices");
                
                // Convert global scene to standard scene for compatibility
                var unifiedScene = Pm4GlobalTileLoader.ToStandardScene(globalScene);
                
                // Extract MSCN anchors from the unified scene
                var mscnChunks = unifiedScene.ExtraChunks
                    .Where(chunk => chunk.GetType().Name.Contains("Mscn"))
                    .ToList();
                
                ConsoleLogger.WriteLine($"Found {mscnChunks.Count} MSCN chunks across all tiles");
                
                // Process the MSCN chunks
                int totalVertexCount = 0;
                foreach (var chunk in mscnChunks)
                {
                    ConsoleLogger.WriteLine($"Processing MSCN chunk: {chunk.GetType().Name}");
                    
                    // Use reflection to get vertices from MSCN chunk
                    var verticesProperty = chunk.GetType().GetProperty("Vertices");
                    if (verticesProperty != null)
                    {
                        var chunkVertices = verticesProperty.GetValue(chunk);
                        
                        if (chunkVertices != null)
                        {
                            // Try to iterate through the collection using reflection
                            var enumerableInterface = chunkVertices.GetType().GetInterface("IEnumerable");
                            if (enumerableInterface != null)
                            {
                                var vertexCount = 0;
                                foreach (var vertex in (System.Collections.IEnumerable)chunkVertices)
                                {
                                    if (vertex != null)
                                    {
                                        // Get vertex properties using reflection
                                        var vertexType = vertex.GetType();
                                        object? xMember = vertexType.GetProperty("X") ?? (object?)vertexType.GetField("X");
                                        object? yMember = vertexType.GetProperty("Y") ?? (object?)vertexType.GetField("Y");
                                        object? zMember = vertexType.GetProperty("Z") ?? (object?)vertexType.GetField("Z");
                                        
                                        if (xMember != null && yMember != null && zMember != null)
                                        {
                                            var xVal = Convert.ToSingle(GetMemberValue(xMember, vertex));
                                            var yVal = Convert.ToSingle(GetMemberValue(yMember, vertex));
                                            var zVal = Convert.ToSingle(GetMemberValue(zMember, vertex));
                                            
                                            // Store raw vertex values for export
                                            var rawVertex = new Vector3(xVal, yVal, zVal);
                                            rawVertices.Add(rawVertex);
                                            
                                            // Create normalized vertices (divide X and Y by 533.33 as suggested)
                                            const float normalizationFactor = 533.33f;
                                            var normalizedVertex = new Vector3(
                                                xVal / normalizationFactor,
                                                yVal / normalizationFactor,
                                                zVal
                                            );
                                            normalizedVertices.Add(normalizedVertex);
                                            
                                            // Apply coordinate transformation using unified service
                                            var transformedVertex = CoordinateTransformationService.ApplyMscnTransformation(xVal, yVal, zVal);
                                            
                                            // Make coordinates relative by subtracting tile offset
                                            transformedVertex.X -= 17066.0f;
                                            transformedVertex.Y -= 17066.0f;
                                            
                                            allVertices.Add(transformedVertex);
                                            vertexCount++;
                                            totalVertexCount++;
                                        }
                                    }
                                }
                                ConsoleLogger.WriteLine($"Processed {vertexCount} vertices from this chunk");
                            }
                        }
                    }
                }
                
                // Export raw and normalized vertices to OBJ files for visual inspection
                var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"session_{DateTime.Now:yyyyMMdd_HHmmss}");
                Directory.CreateDirectory(outputDir);
                
                // Export raw MSCN vertices
                var rawObjPath = Path.Combine(outputDir, "mscn_raw_multi_tile.obj");
                ExportVerticesAsObj(rawVertices, rawObjPath, "Raw Multi-Tile MSCN Vertices (Server/World Coordinates)").GetAwaiter().GetResult();
                ConsoleLogger.WriteLine($"Exported raw multi-tile MSCN vertices to {rawObjPath}");
                
                // Export normalized MSCN vertices
                var normalizedObjPath = Path.Combine(outputDir, "mscn_normalized_multi_tile.obj");
                ExportVerticesAsObj(normalizedVertices, normalizedObjPath, "Normalized Multi-Tile MSCN Vertices (Divided by 533.33)").GetAwaiter().GetResult();
                ConsoleLogger.WriteLine($"Exported normalized multi-tile MSCN vertices to {normalizedObjPath}");
                
                ConsoleLogger.WriteLine($"Extracted {allVertices.Count} MSCN anchor vertices from multiple tiles");
                ConsoleLogger.WriteLine($"Raw vertices: {rawVertices.Count}, Normalized vertices: {normalizedVertices.Count}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error loading PM4 tiles: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
                
                // Fallback to the legacy method if GlobalTileLoader fails
                ConsoleLogger.WriteLine("Falling back to manual tile loading method...");
                
                // Extract tile coordinates from filename (e.g., development_15_37.pm4 -> 15, 37)
                var fileName = Path.GetFileNameWithoutExtension(basePm4FilePath);
                var parts = fileName.Split('_');
                
                if (parts.Length < 3 || !int.TryParse(parts[^2], out int baseTileX) || !int.TryParse(parts[^1], out int baseTileY))
                {
                    ConsoleLogger.WriteLine($"Warning: Could not parse tile coordinates from {fileName}, falling back to single tile");
                    return ExtractMscnAnchorsFromPm4(basePm4FilePath);
                }
                
                ConsoleLogger.WriteLine($"Base tile coordinates: {baseTileX}_{baseTileY}");
                
                // Define adjacent tile offsets (3x3 grid around base tile)
                var tileOffsets = new (int dx, int dy)[] 
                {
                    (-1, -1), (0, -1), (1, -1),
                    (-1,  0), (0,  0), (1,  0),
                    (-1,  1), (0,  1), (1,  1)
                };
                
                var baseDir = Path.GetDirectoryName(basePm4FilePath) ?? ".";
                var filePrefix = string.Join("_", parts.Take(parts.Length - 2));
                
                foreach (var (dx, dy) in tileOffsets)
                {
                    var tileX = baseTileX + dx;
                    var tileY = baseTileY + dy;
                    var tileFileName = $"{filePrefix}_{tileX:D2}_{tileY:D2}.pm4";
                    var tileFilePath = Path.Combine(baseDir, tileFileName);
                    
                    if (File.Exists(tileFilePath))
                    {
                        ConsoleLogger.WriteLine($"Loading MSCN data from tile {tileX:D2}_{tileY:D2}...");
                        try
                        {
                            var tileVertices = ExtractMscnAnchorsFromPm4(tileFilePath);
                            allVertices.AddRange(tileVertices);
                            ConsoleLogger.WriteLine($"  Added {tileVertices.Count} vertices from {tileFileName}");
                        }
                        catch (Exception innerEx)
                        {
                            ConsoleLogger.WriteLine($"  Warning: Failed to load {tileFileName}: {innerEx.Message}");
                        }
                    }
                    else
                    {
                        ConsoleLogger.WriteLine($"  Tile {tileX:D2}_{tileY:D2} not found: {tileFileName}");
                    }
                }
            }
            
            ConsoleLogger.WriteLine($"Multi-tile aggregation complete: {allVertices.Count} total MSCN vertices");
            return allVertices;
        }
        
        private static async Task<List<(string name, SpatialCorrelationResult result, List<Vector3> transformedVertices)>> TestCoordinateTransformations(
            List<Vector3> originalMscnVertices, 
            List<Vector3> wmoVertices, 
            float tolerance, 
            string? groupFilter, 
            string outputDir)
        {
            var transformationResults = new List<(string name, SpatialCorrelationResult result, List<Vector3> transformedVertices)>();
            
            // Define common coordinate system transformations
            var transformations = new Dictionary<string, Func<Vector3, Vector3>>
            {
                ["Original"] = v => v,
                ["Z-Flip"] = v => new Vector3(v.X, v.Y, -v.Z),
                ["180° Rotation (XY)"] = v => new Vector3(-v.X, -v.Y, v.Z),
                ["90° CW Rotation (XY)"] = v => new Vector3(v.Y, -v.X, v.Z),
                ["90° CCW Rotation (XY)"] = v => new Vector3(-v.Y, v.X, v.Z),
                ["270° CW Rotation (XY)"] = v => new Vector3(-v.Y, v.X, v.Z), // Same as 90° CCW
                ["45° CW Rotation (XY)"] = v => RotateXY(v, -45f),
                ["135° CW Rotation (XY)"] = v => RotateXY(v, -135f),
                ["225° CW Rotation (XY)"] = v => RotateXY(v, -225f),
                ["315° CW Rotation (XY)"] = v => RotateXY(v, -315f),
                ["X-Y Swap"] = v => new Vector3(v.Y, v.X, v.Z),
                ["X-Z Swap"] = v => new Vector3(v.Z, v.Y, v.X),
                ["Y-Z Swap"] = v => new Vector3(v.X, v.Z, v.Y),
                ["Mirror X"] = v => new Vector3(-v.X, v.Y, v.Z),
                ["Mirror Y"] = v => new Vector3(v.X, -v.Y, v.Z)
            };
            
            ConsoleLogger.WriteLine($"Testing {transformations.Count} coordinate transformations...");
            
            foreach (var (transformName, transform) in transformations)
            {
                ConsoleLogger.WriteLine($"  Testing: {transformName}");
                
                // Apply transformation
                var transformedVertices = originalMscnVertices.Select(transform).ToList();
                
                // Test correlation
                var result = AnalyzeObjectCorrelation(transformedVertices, wmoVertices, tolerance, groupFilter);
                
                // Store result
                transformationResults.Add((transformName, result, transformedVertices));
                
                ConsoleLogger.WriteLine($"    Result: {result.Matches.Count} matches ({result.MatchPercentage:F1}%)");
                
                // Export transformed vertices for visualization if they have good correlation
                if (result.MatchPercentage > 0.1f) // Only export if there are some matches
                {
                    var transformedObjPath = Path.Combine(outputDir, $"mscn_anchors_{transformName.Replace("°", "deg").Replace(" ", "_").Replace("(", "").Replace(")", "")}.obj");
                    await ExportVerticesAsObj(transformedVertices, transformedObjPath, $"MSCN Anchors ({transformName})");
                }
            }
            
            // Sort by match percentage (best first)
            transformationResults.Sort((a, b) => b.result.MatchPercentage.CompareTo(a.result.MatchPercentage));
            
            ConsoleLogger.WriteLine();
            ConsoleLogger.WriteLine("=== TRANSFORMATION RESULTS (Best to Worst) ===");
            foreach (var (name, result, _) in transformationResults.Take(5))
            {
                ConsoleLogger.WriteLine($"  {name}: {result.Matches.Count} matches ({result.MatchPercentage:F1}%)");
            }
            
            return transformationResults;
        }
        
        private static async Task<(SpatialCorrelationResult originalResult, SpatialCorrelationResult filteredResult, SpatialCorrelationResult rotatedFilteredResult, List<Vector3> filteredMscnVertices, List<Vector3> rotatedFilteredMscnVertices)> PerformSpatialFilteredComparison(
            List<Vector3> mscnVertices, 
            List<Vector3> wmoVertices, 
            float tolerance, 
            string? groupFilter, 
            string outputDir)
        {
            ConsoleLogger.WriteLine($"Performing spatial filtering analysis...");
            
            // Calculate WMO bounds to determine filtering region
            var wmoBounds = CalculateBounds(wmoVertices);
            ConsoleLogger.WriteLine($"WMO bounds: Center=({wmoBounds.Center.X:F2}, {wmoBounds.Center.Y:F2}, {wmoBounds.Center.Z:F2}), Size=({wmoBounds.Size.X:F2}, {wmoBounds.Size.Y:F2}, {wmoBounds.Size.Z:F2})");
            
            // Expand WMO bounds by 2x to create filtering region (account for placement offset)
            var filterRadius = Math.Max(wmoBounds.Size.X, wmoBounds.Size.Y) * 1.5f;
            ConsoleLogger.WriteLine($"Using spatial filter radius: {filterRadius:F2} units");
            
            // Test different potential WMO world positions to find best match
            var testPositions = new List<(string name, Vector3 offset)>
            {
                ("WMO Center", wmoBounds.Center),
                ("Origin", Vector3.Zero),
                ("MSCN Center", CalculateBounds(mscnVertices).Center),
                ("Offset +100,+100", wmoBounds.Center + new Vector3(100, 100, 0)),
                ("Offset -100,-100", wmoBounds.Center + new Vector3(-100, -100, 0)),
                ("Offset +200,+200", wmoBounds.Center + new Vector3(200, 200, 0)),
                ("Offset -200,-200", wmoBounds.Center + new Vector3(-200, -200, 0))
            };
            
            var bestResult = (name: "None", originalResult: new SpatialCorrelationResult(), filteredResult: new SpatialCorrelationResult(), rotatedFilteredResult: new SpatialCorrelationResult(), filteredVertices: new List<Vector3>(), rotatedFilteredVertices: new List<Vector3>());
            
            foreach (var (positionName, testCenter) in testPositions)
            {
                ConsoleLogger.WriteLine($"\nTesting spatial filter around: {positionName} ({testCenter.X:F2}, {testCenter.Y:F2}, {testCenter.Z:F2})");
                
                // Filter MSCN vertices to region around test position
                var filteredMscnVertices = mscnVertices.Where(v => 
                    Vector3.Distance(v, testCenter) <= filterRadius
                ).ToList();
                
                ConsoleLogger.WriteLine($"  Filtered MSCN vertices: {filteredMscnVertices.Count} / {mscnVertices.Count} ({(float)filteredMscnVertices.Count / mscnVertices.Count * 100:F1}%)");
                
                if (filteredMscnVertices.Count < 10) // Skip if too few vertices
                {
                    ConsoleLogger.WriteLine($"  Skipping - too few filtered vertices");
                    continue;
                }
                
                // Test original filtered vertices
                var filteredResult = AnalyzeObjectCorrelation(filteredMscnVertices, wmoVertices, tolerance, groupFilter);
                ConsoleLogger.WriteLine($"  Original filtered correlation: {filteredResult.MatchPercentage:F1}% ({filteredResult.Matches.Count} matches)");
                
                // Test 180-degree rotated filtered vertices
                var rotatedFilteredVertices = filteredMscnVertices.Select(v => CoordinateTransformationService.Apply180DegreeRotation(v)).ToList();
                var rotatedFilteredResult = AnalyzeObjectCorrelation(rotatedFilteredVertices, wmoVertices, tolerance, groupFilter);
                ConsoleLogger.WriteLine($"  180° rotated filtered correlation: {rotatedFilteredResult.MatchPercentage:F1}% ({rotatedFilteredResult.Matches.Count} matches)");
                
                // Keep track of best result
                var bestPercentage = Math.Max(filteredResult.MatchPercentage, rotatedFilteredResult.MatchPercentage);
                if (bestPercentage > Math.Max(bestResult.filteredResult.MatchPercentage, bestResult.rotatedFilteredResult.MatchPercentage))
                {
                    var originalResult = AnalyzeObjectCorrelation(mscnVertices, wmoVertices, tolerance, groupFilter);
                    bestResult = (positionName, originalResult, filteredResult, rotatedFilteredResult, filteredMscnVertices, rotatedFilteredVertices);
                    ConsoleLogger.WriteLine($"  *** NEW BEST: {positionName} with {bestPercentage:F1}% correlation");
                }
                
                // Export filtered vertices for visualization if they show promise
                if (bestPercentage > 0.1f)
                {
                    var filteredObjPath = Path.Combine(outputDir, $"mscn_filtered_{positionName.Replace(" ", "_").Replace(",", "").Replace("+", "plus").Replace("-", "minus")}.obj");
                    await ExportVerticesAsObj(filteredMscnVertices, filteredObjPath, $"Filtered MSCN ({positionName})");
                    
                    if (rotatedFilteredResult.MatchPercentage > filteredResult.MatchPercentage)
                    {
                        var rotatedObjPath = Path.Combine(outputDir, $"mscn_filtered_rotated_{positionName.Replace(" ", "_").Replace(",", "").Replace("+", "plus").Replace("-", "minus")}.obj");
                        await ExportVerticesAsObj(rotatedFilteredVertices, rotatedObjPath, $"Filtered Rotated MSCN ({positionName})");
                    }
                }
            }
            
            ConsoleLogger.WriteLine();
            ConsoleLogger.WriteLine($"=== SPATIAL FILTERING RESULTS ===");
            ConsoleLogger.WriteLine($"Best position: {bestResult.name}");
            ConsoleLogger.WriteLine($"Original (all MSCN): {bestResult.originalResult.MatchPercentage:F1}% ({bestResult.originalResult.Matches.Count} matches)");
            ConsoleLogger.WriteLine($"Filtered: {bestResult.filteredResult.MatchPercentage:F1}% ({bestResult.filteredResult.Matches.Count} matches)");
            ConsoleLogger.WriteLine($"Filtered + 180° rotated: {bestResult.rotatedFilteredResult.MatchPercentage:F1}% ({bestResult.rotatedFilteredResult.Matches.Count} matches)");
            
            return (bestResult.originalResult, bestResult.filteredResult, bestResult.rotatedFilteredResult, bestResult.filteredVertices, bestResult.rotatedFilteredVertices);
        }
        
        private static async Task<(SpatialCorrelationResult originalResult, SpatialCorrelationResult normalizedResult, SpatialCorrelationResult rotatedNormalizedResult, List<Vector3> normalizedMscnVertices, List<Vector3> rotatedNormalizedVertices)> PerformMscnNormalizationComparison(
            List<Vector3> mscnVertices, 
            List<Vector3> wmoVertices, 
            float tolerance, 
            string? groupFilter, 
            string outputDir)
        {
            ConsoleLogger.WriteLine($"Performing MSCN-to-local coordinate normalization...");
            
            // Calculate bounds for both datasets
            var mscnBounds = CalculateBounds(mscnVertices);
            var wmoBounds = CalculateBounds(wmoVertices);
            
            ConsoleLogger.WriteLine($"MSCN bounds: Center=({mscnBounds.Center.X:F2}, {mscnBounds.Center.Y:F2}, {mscnBounds.Center.Z:F2}), Size=({mscnBounds.Size.X:F2}, {mscnBounds.Size.Y:F2}, {mscnBounds.Size.Z:F2})");
            ConsoleLogger.WriteLine($"WMO bounds: Center=({wmoBounds.Center.X:F2}, {wmoBounds.Center.Y:F2}, {wmoBounds.Center.Z:F2}), Size=({wmoBounds.Size.X:F2}, {wmoBounds.Size.Y:F2}, {wmoBounds.Size.Z:F2})");
            
            // Step 1: Center MSCN coordinates (translate to origin)
            var centeredMscnVertices = mscnVertices.Select(v => v - mscnBounds.Center).ToList();
            ConsoleLogger.WriteLine($"Step 1: Centered MSCN coordinates around origin");
            
            // Step 2: Scale MSCN coordinates to match WMO size
            var scaleFactorX = wmoBounds.Size.X / mscnBounds.Size.X;
            var scaleFactorY = wmoBounds.Size.Y / mscnBounds.Size.Y;
            var scaleFactorZ = wmoBounds.Size.Z / mscnBounds.Size.Z;
            
            // Use uniform scaling based on the average of X and Y (ignore Z for now as it might be different)
            var uniformScale = (scaleFactorX + scaleFactorY) / 2.0f;
            ConsoleLogger.WriteLine($"Scale factors: X={scaleFactorX:F4}, Y={scaleFactorY:F4}, Z={scaleFactorZ:F4}, Uniform={uniformScale:F4}");
            
            var scaledMscnVertices = centeredMscnVertices.Select(v => new Vector3(
                v.X * uniformScale,
                v.Y * uniformScale,
                v.Z * scaleFactorZ  // Use Z-specific scaling for height
            )).ToList();
            ConsoleLogger.WriteLine($"Step 2: Scaled MSCN coordinates to match WMO size");
            
            // Step 3: Translate to WMO center
            var normalizedMscnVertices = scaledMscnVertices.Select(v => v + wmoBounds.Center).ToList();
            ConsoleLogger.WriteLine($"Step 3: Translated MSCN coordinates to WMO center");
            
            // Verify normalized bounds
            var normalizedBounds = CalculateBounds(normalizedMscnVertices);
            ConsoleLogger.WriteLine($"Normalized MSCN bounds: Center=({normalizedBounds.Center.X:F2}, {normalizedBounds.Center.Y:F2}, {normalizedBounds.Center.Z:F2}), Size=({normalizedBounds.Size.X:F2}, {normalizedBounds.Size.Y:F2}, {normalizedBounds.Size.Z:F2})");
            
            // Test original correlation (should be 0%)
            var originalResult = AnalyzeObjectCorrelation(mscnVertices, wmoVertices, tolerance, groupFilter);
            ConsoleLogger.WriteLine($"Original correlation: {originalResult.MatchPercentage:F1}% ({originalResult.Matches.Count} matches)");
            
            // Test normalized correlation
            var normalizedResult = AnalyzeObjectCorrelation(normalizedMscnVertices, wmoVertices, tolerance, groupFilter);
            ConsoleLogger.WriteLine($"Normalized correlation: {normalizedResult.MatchPercentage:F1}% ({normalizedResult.Matches.Count} matches)");
            
            // Test normalized + 180-degree rotated correlation
            var rotatedNormalizedVertices = normalizedMscnVertices.Select(v => 
                CoordinateTransformationService.Apply180DegreeRotation(v, wmoBounds.Center)
            ).ToList();
            
            var rotatedNormalizedResult = AnalyzeObjectCorrelation(rotatedNormalizedVertices, wmoVertices, tolerance, groupFilter);
            ConsoleLogger.WriteLine($"Normalized + 180° rotated correlation: {rotatedNormalizedResult.MatchPercentage:F1}% ({rotatedNormalizedResult.Matches.Count} matches)");
            
            // Export normalized vertices for visualization
            var normalizedObjPath = Path.Combine(outputDir, "mscn_normalized.obj");
            await ExportVerticesAsObj(normalizedMscnVertices, normalizedObjPath, "Normalized MSCN Anchors");
            
            var rotatedNormalizedObjPath = Path.Combine(outputDir, "mscn_normalized_rotated180.obj");
            await ExportVerticesAsObj(rotatedNormalizedVertices, rotatedNormalizedObjPath, "Normalized + 180° Rotated MSCN Anchors");
            
            // Export match visualizations for debugging
            await ExportMatchVisualization(normalizedResult, normalizedMscnVertices, wmoVertices, outputDir, "normalized");
            await ExportMatchVisualization(rotatedNormalizedResult, rotatedNormalizedVertices, wmoVertices, outputDir, "rotated_normalized");
            
            // Debug: Test with larger tolerance to see if we get any matches
            ConsoleLogger.WriteLine($"\nDEBUG: Testing with larger tolerance values...");
            var debugTolerances = new float[] { 10.0f, 25.0f, 50.0f, 100.0f };
            foreach (var debugTolerance in debugTolerances)
            {
                var debugResult = AnalyzeObjectCorrelation(normalizedMscnVertices, wmoVertices, debugTolerance, groupFilter);
                ConsoleLogger.WriteLine($"  Tolerance {debugTolerance:F1}: {debugResult.Matches.Count} matches ({debugResult.MatchPercentage:F1}%)");
                
                if (debugResult.Matches.Count > 0)
                {
                    await ExportMatchVisualization(debugResult, normalizedMscnVertices, wmoVertices, outputDir, $"debug_tolerance_{debugTolerance:F0}");
                    break; // Stop at first tolerance that gives matches
                }
            }
            
            ConsoleLogger.WriteLine();
            ConsoleLogger.WriteLine($"=== NORMALIZATION RESULTS ===");
            ConsoleLogger.WriteLine($"Original: {originalResult.MatchPercentage:F1}% ({originalResult.Matches.Count} matches)");
            ConsoleLogger.WriteLine($"Normalized: {normalizedResult.MatchPercentage:F1}% ({normalizedResult.Matches.Count} matches)");
            ConsoleLogger.WriteLine($"Normalized + 180° rotated: {rotatedNormalizedResult.MatchPercentage:F1}% ({rotatedNormalizedResult.Matches.Count} matches)");
            
            ConsoleLogger.WriteLine($"Exported normalized visualizations:");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(normalizedObjPath)}");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(rotatedNormalizedObjPath)}");
            
            return (originalResult, normalizedResult, rotatedNormalizedResult, normalizedMscnVertices, rotatedNormalizedVertices);
        }
        
        private static async Task ExportMatchVisualization(
            SpatialCorrelationResult correlationResult,
            List<Vector3> mscnVertices,
            List<Vector3> wmoVertices,
            string outputDir,
            string suffix)
        {
            if (correlationResult.Matches.Count == 0)
            {
                ConsoleLogger.WriteLine($"No matches to export for {suffix}");
                return;
            }
            
            ConsoleLogger.WriteLine($"Exporting {correlationResult.Matches.Count} matches for {suffix}...");
            
            // Export matched MSCN vertices
            var matchedMscnPath = Path.Combine(outputDir, $"matched_mscn_{suffix}.obj");
            using (var writer = new StreamWriter(matchedMscnPath))
            {
                await writer.WriteLineAsync($"# Matched MSCN Vertices ({suffix})");
                await writer.WriteLineAsync($"# {correlationResult.Matches.Count} matches");
                await writer.WriteLineAsync();
                
                foreach (var match in correlationResult.Matches)
                {
                    var vertex = mscnVertices[match.MscnIndex];
                    await writer.WriteLineAsync($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }
            }
            
            // Export matched WMO vertices
            var matchedWmoPath = Path.Combine(outputDir, $"matched_wmo_{suffix}.obj");
            using (var writer = new StreamWriter(matchedWmoPath))
            {
                await writer.WriteLineAsync($"# Matched WMO Vertices ({suffix})");
                await writer.WriteLineAsync($"# {correlationResult.Matches.Count} matches");
                await writer.WriteLineAsync();
                
                foreach (var match in correlationResult.Matches)
                {
                    var vertex = wmoVertices[match.WmoIndex];
                    await writer.WriteLineAsync($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }
            }
            
            // Export match pairs as lines
            var matchPairsPath = Path.Combine(outputDir, $"match_pairs_{suffix}.obj");
            using (var writer = new StreamWriter(matchPairsPath))
            {
                await writer.WriteLineAsync($"# Match Pairs ({suffix})");
                await writer.WriteLineAsync($"# {correlationResult.Matches.Count} pairs, distance tolerance used");
                await writer.WriteLineAsync();
                
                int vertexIndex = 1;
                foreach (var match in correlationResult.Matches)
                {
                    var mscnVertex = mscnVertices[match.MscnIndex];
                    var wmoVertex = wmoVertices[match.WmoIndex];
                    
                    // Add both vertices
                    await writer.WriteLineAsync($"v {mscnVertex.X:F6} {mscnVertex.Y:F6} {mscnVertex.Z:F6}");
                    await writer.WriteLineAsync($"v {wmoVertex.X:F6} {wmoVertex.Y:F6} {wmoVertex.Z:F6}");
                    
                    // Add line connecting them
                    await writer.WriteLineAsync($"l {vertexIndex} {vertexIndex + 1}");
                    
                    vertexIndex += 2;
                }
            }
            
            ConsoleLogger.WriteLine($"Match visualization exported:");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(matchedMscnPath)} ({correlationResult.Matches.Count} MSCN vertices)");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(matchedWmoPath)} ({correlationResult.Matches.Count} WMO vertices)");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(matchPairsPath)} ({correlationResult.Matches.Count} connection lines)");
        }
        
        private static Vector3 RotateXY(Vector3 v, float angleDegrees)
        {
            var angleRadians = angleDegrees * (float)Math.PI / 180f;
            var cos = (float)Math.Cos(angleRadians);
            var sin = (float)Math.Sin(angleRadians);
            
            return new Vector3(
                v.X * cos - v.Y * sin,
                v.X * sin + v.Y * cos,
                v.Z
            );
        }
        
        public static List<Vector3> LoadWmoVertices(string wmoFilePath, string? groupFilter = null)
        {
            var vertices = new List<Vector3>();
            
            try
            {
                ConsoleLogger.WriteLine($"Loading WMO file: {wmoFilePath}");
                
                if (!File.Exists(wmoFilePath))
                {
                    ConsoleLogger.WriteLine($"WMO file not found: {wmoFilePath}");
                    return vertices;
                }
                
                // Configure FileProvider for local file access
                ConsoleLogger.WriteLine($"Configuring FileProvider for local file access...");
                var localProvider = new LocalFileProvider(Path.GetDirectoryName(wmoFilePath) ?? ".");
                FileProvider.SetProvider(localProvider, "local");
                FileProvider.SetDefaultBuild("local");
                
                // Use the existing WMO loader infrastructure
                ConsoleLogger.WriteLine($"Initializing WowToolsLocalWmoLoader...");
                var wmoLoader = new WowToolsLocalWmoLoader();
                
                ConsoleLogger.WriteLine($"Loading WMO with WowToolsLocalWmoLoader...");
                var (textures, groups) = wmoLoader.Load(wmoFilePath, includeFacades: false);
                
                ConsoleLogger.WriteLine($"Loaded WMO with {groups.Count} groups and {textures.Count} textures");
                
                // Apply group filter if specified
                var filteredGroups = groups;
                if (!string.IsNullOrEmpty(groupFilter))
                {
                    filteredGroups = groups.Where(g => g.Name.Contains(groupFilter, StringComparison.OrdinalIgnoreCase)).ToList();
                    ConsoleLogger.WriteLine($"Filtered to {filteredGroups.Count} groups matching '{groupFilter}'");
                }
                
                ConsoleLogger.WriteLine($"Processing {filteredGroups.Count} filtered groups");
                
                // Instead of scaling down WMO vertices, we'll keep them at original scale
                // and scale up MSCN vertices to match. This avoids precision issues from tiny values.
                var debugCount = 0;
                var totalProcessed = 0;
                foreach (var group in filteredGroups)
                {
                    ConsoleLogger.WriteLine($"Processing group {group.Name} with {group.Vertices.Count} vertices");
                    var groupVertexCount = 0;
                    foreach (var vertex in group.Vertices)
                    {
                        // Debug: Show first few original vertex values
                        if (debugCount < 10)
                        {
                            ConsoleLogger.WriteLine($"Original WMO[{debugCount}]: ({vertex.X:F2}, {vertex.Y:F2}, {vertex.Z:F2})");
                        }
                        
                        // Keep WMO vertices at original scale
                        // No scaling or transformation needed at this stage
                        vertices.Add(vertex);
                        
                        groupVertexCount++;
                        debugCount++;
                        totalProcessed++;
                        
                        // Safety check to prevent infinite loops
                        if (totalProcessed > 1000000)
                        {
                            ConsoleLogger.WriteLine("Safety break: Processed over 1 million vertices");
                            break;
                        }
                    }
                    
                    ConsoleLogger.WriteLine($"Loaded {groupVertexCount} vertices from group {group.Name}");
                    if (totalProcessed > 1000000) break;
                }
                
                ConsoleLogger.WriteLine($"Total WMO vertices loaded: {vertices.Count}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error loading WMO file: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            
            return vertices;
        }
        
        private static Dictionary<(int, int, int), List<int>> BuildSpatialHashGrid(List<Vector3> vertices, float cellSize)
        {
            var grid = new Dictionary<(int, int, int), List<int>>();
            
            for (int i = 0; i < vertices.Count; i++)
            {
                var vertex = vertices[i];
                var cellKey = (
                    (int)Math.Floor(vertex.X / cellSize),
                    (int)Math.Floor(vertex.Y / cellSize),
                    (int)Math.Floor(vertex.Z / cellSize)
                );
                
                if (!grid.ContainsKey(cellKey))
                {
                    grid[cellKey] = new List<int>();
                }
                grid[cellKey].Add(i);
            }
            
            return grid;
        }
        
        private static (float distance, int index) FindNearestWmoVertex(
            Vector3 mscnVertex, 
            List<Vector3> wmoVertices, 
            Dictionary<(int, int, int), List<int>> spatialGrid, 
            float tolerance)
        {
            float closestDistance = float.MaxValue;
            int closestIndex = -1;
            
            // Early exit check for empty vertex list
            if (wmoVertices.Count == 0) return (float.MaxValue, -1);
            
            // Calculate the cell coordinates for the MSCN vertex
            var centerCell = (
                (int)Math.Floor(mscnVertex.X / tolerance),
                (int)Math.Floor(mscnVertex.Y / tolerance),
                (int)Math.Floor(mscnVertex.Z / tolerance)
            );
            
            // First, check the center cell for an exact or very close match
            // This can save a lot of time if there's a good match in the center cell
            if (spatialGrid.TryGetValue(centerCell, out var centerVertices))
            {
                foreach (var vertexIndex in centerVertices)
                {
                    var distance = Vector3.Distance(mscnVertex, wmoVertices[vertexIndex]);
                    if (distance < closestDistance)
                    {
                        closestDistance = distance;
                        closestIndex = vertexIndex;
                        
                        // Early exit if we find a very close match
                        if (distance < tolerance * 0.1f)
                        {
                            return (distance, vertexIndex);
                        }
                    }
                }
            }
            
            // If we didn't find a good match in center cell, search neighboring cells
            // but skip the center cell since we already processed it
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        // Skip the center cell - we already processed it
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        
                        var cellKey = (centerCell.Item1 + dx, centerCell.Item2 + dy, centerCell.Item3 + dz);
                        
                        if (spatialGrid.TryGetValue(cellKey, out var vertexIndices))
                        {
                            foreach (var vertexIndex in vertexIndices)
                            {
                                var distance = Vector3.Distance(mscnVertex, wmoVertices[vertexIndex]);
                                if (distance < closestDistance)
                                {
                                    closestDistance = distance;
                                    closestIndex = vertexIndex;
                                    
                                    // Early exit if we find a very close match
                                    if (distance < tolerance * 0.1f)
                                    {
                                        return (distance, vertexIndex);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            return (closestDistance, closestIndex);
        }
        
        private static List<(Vector3 center, Vector3 size, int count)> CreateSpatialClusters(List<Vector3> vertices, float clusterRadius)
        {
            ConsoleLogger.WriteLine($"Creating spatial clusters from {vertices.Count} vertices with radius {clusterRadius}...");
            
            var clusters = new List<(Vector3 center, Vector3 size, int count)>();
            
            // Safety check for empty vertex list
            if (vertices.Count == 0)
            {
                ConsoleLogger.WriteLine("No vertices to cluster");
                return clusters;
            }
            
            // Use a more reasonable cell size for the spatial grid
            // Too large values cause poor performance, too small values may miss neighbors
            var cellSize = clusterRadius * 2.0f; 
            
            // Performance optimization: if there are too many vertices, use sampling
            const int maxVerticesToProcess = 50000;
            var verticesToProcess = vertices;
            if (vertices.Count > maxVerticesToProcess)
            {
                ConsoleLogger.WriteLine($"Sampling {maxVerticesToProcess} vertices from {vertices.Count} for clustering efficiency");
                
                // Sample vertices evenly
                var sampledVertices = new List<Vector3>(maxVerticesToProcess);
                var step = vertices.Count / maxVerticesToProcess;
                for (int i = 0; i < vertices.Count; i += step)
                {
                    sampledVertices.Add(vertices[i]);
                    if (sampledVertices.Count >= maxVerticesToProcess)
                        break;
                }
                verticesToProcess = sampledVertices;
                ConsoleLogger.WriteLine($"Sampled {verticesToProcess.Count} vertices for processing");
            }
            
            // Create DBSCAN-like clustering with spatial hash grid for efficiency
            var processed = new HashSet<int>(); // Use HashSet for faster lookups
            var spatialGrid = new Dictionary<(int, int, int), List<int>>();
            
            // Populate spatial grid
            ConsoleLogger.WriteLine("Building spatial hash grid...");
            var startTime = DateTime.Now;
            for (int i = 0; i < verticesToProcess.Count; i++)
            {
                var vertex = verticesToProcess[i];
                var cellKey = (
                    (int)Math.Floor(vertex.X / cellSize),
                    (int)Math.Floor(vertex.Y / cellSize),
                    (int)Math.Floor(vertex.Z / cellSize)
                );
                
                if (!spatialGrid.ContainsKey(cellKey))
                    spatialGrid[cellKey] = new List<int>();
                spatialGrid[cellKey].Add(i);
            }
            
            var gridBuildTime = (DateTime.Now - startTime).TotalSeconds;
            ConsoleLogger.WriteLine($"Created spatial grid with {spatialGrid.Count} cells in {gridBuildTime:F2} seconds");
            
            // Set a reasonable timeout to prevent hanging
            var timeout = TimeSpan.FromSeconds(30);
            var clusteringStartTime = DateTime.Now;
            
            // Process vertices using spatial grid with timeout protection
            ConsoleLogger.WriteLine("Finding clusters...");
            int processedCount = 0;
            int clusterCount = 0;
            
            for (int i = 0; i < verticesToProcess.Count; i++)
            {
                // Check for timeout
                if (DateTime.Now - clusteringStartTime > timeout)
                {
                    ConsoleLogger.WriteLine($"Clustering timed out after {timeout.TotalSeconds} seconds. Processed {processedCount}/{verticesToProcess.Count} vertices.");
                    break;
                }
                
                if (processed.Contains(i)) continue;
                
                var clusterVertices = new List<Vector3>();
                var queue = new Queue<int>();
                queue.Enqueue(i);
                processed.Add(i);
                processedCount++;
                
                // Use a maximum cluster size to prevent very large clusters
                const int maxClusterSize = 5000;
                
                // Find all vertices within cluster radius using spatial grid
                while (queue.Count > 0 && clusterVertices.Count < maxClusterSize)
                {
                    // Check for timeout periodically
                    if (clusterVertices.Count % 1000 == 0 && DateTime.Now - clusteringStartTime > timeout)
                    {
                        ConsoleLogger.WriteLine($"Clustering timed out while processing cluster {clusterCount}. Breaking early.");
                        break;
                    }
                    
                    var currentIndex = queue.Dequeue();
                    var currentVertex = verticesToProcess[currentIndex];
                    clusterVertices.Add(currentVertex);
                    
                    // Check neighboring cells
                    var centerCell = (
                        (int)Math.Floor(currentVertex.X / cellSize),
                        (int)Math.Floor(currentVertex.Y / cellSize),
                        (int)Math.Floor(currentVertex.Z / cellSize)
                    );
                    
                    // Check immediate neighborhood (instead of 3x3x3 which is expensive)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                var cellKey = (centerCell.Item1 + dx, centerCell.Item2 + dy, centerCell.Item3 + dz);
                                
                                if (spatialGrid.TryGetValue(cellKey, out var cellVertices))
                                {
                                    foreach (var vertexIndex in cellVertices)
                                    {
                                        if (!processed.Contains(vertexIndex) && 
                                            Vector3.Distance(currentVertex, verticesToProcess[vertexIndex]) <= clusterRadius)
                                        {
                                            processed.Add(vertexIndex);
                                            processedCount++;
                                            queue.Enqueue(vertexIndex);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Calculate cluster bounds
                if (clusterVertices.Count >= 5) // Ignore very small clusters (noise)
                {
                    var bounds = CalculateBounds(clusterVertices);
                    clusters.Add((bounds.Center, bounds.Size, clusterVertices.Count));
                    clusterCount++;
                    
                    // Log progress periodically
                    if (clusterCount % 10 == 0 || clusters.Count == 1)
                    {
                        ConsoleLogger.WriteLine($"Created {clusterCount} clusters so far, processed {processedCount}/{verticesToProcess.Count} vertices");
                    }
                }
            }
            
            var totalTime = (DateTime.Now - clusteringStartTime).TotalSeconds;
            ConsoleLogger.WriteLine($"Created {clusters.Count} spatial clusters in {totalTime:F2} seconds");
            return clusters;
        }
        


        /// <summary>
        /// Aggregates MSCN anchor vertices from the base PM4 tile and its immediate
        /// neighbours (3×3 grid). This method honours the user's expectation for the
        /// --multi-tile flag without resorting to loading every PM4 in the directory.
        /// </summary>
        private static List<Vector3> ExtractMscnAnchorsFromNeighbourTiles(string basePm4FilePath)
        {
            var allVertices = new List<Vector3>();

            // Derive tile coordinates from filename e.g. development_15_37.pm4 → 15,37
            var fileName = Path.GetFileNameWithoutExtension(basePm4FilePath);
            var parts = fileName.Split('_');
            if (parts.Length < 3 || !int.TryParse(parts[^2], out int baseTileX) || !int.TryParse(parts[^1], out int baseTileY))
            {
                ConsoleLogger.WriteLine($"Warning: Could not parse tile coordinates from {fileName}. Falling back to single-tile mode.");
                return ExtractMscnAnchorsFromPm4(basePm4FilePath);
            }

            var baseDir   = Path.GetDirectoryName(basePm4FilePath) ?? ".";
            var filePrefix = string.Join("_", parts.Take(parts.Length - 2));

            ConsoleLogger.WriteLine($"Loading MSCN anchors from base tile {baseTileX:D2}_{baseTileY:D2} and neighbours (3×3 grid)…");

            var offsets = new (int dx,int dy)[]
            {
                (-1,-1),(0,-1),(1,-1),
                (-1, 0),(0, 0),(1, 0),
                (-1, 1),(0, 1),(1, 1)
            };

            foreach (var (dx,dy) in offsets)
            {
                int tileX = baseTileX + dx;
                int tileY = baseTileY + dy;
                var tilePath = Path.Combine(baseDir, $"{filePrefix}_{tileX:D2}_{tileY:D2}.pm4");
                if (!File.Exists(tilePath))
                {
                    ConsoleLogger.WriteLine($"  Tile {tileX:D2}_{tileY:D2} not found – skipping");
                    continue;
                }

                try
                {
                    var verts = ExtractMscnAnchorsFromPm4(tilePath);
                    ConsoleLogger.WriteLine($"  Added {verts.Count} vertices from {Path.GetFileName(tilePath)}");
                    allVertices.AddRange(verts);
                }
                catch (Exception ex)
                {
                    ConsoleLogger.WriteLine($"  Failed to load {Path.GetFileName(tilePath)}: {ex.Message}");
                }
            }

            ConsoleLogger.WriteLine($"Neighbour-tile aggregation complete → {allVertices.Count} MSCN vertices");
            return allVertices;
        }

        private static object? GetMemberValue(object member, object instance)
        {
            return member switch
            {
                System.Reflection.PropertyInfo prop => prop.GetValue(instance),
                System.Reflection.FieldInfo field => field.GetValue(instance),
                _ => null
            };
        }

        private static List<Vector3> LoadObjVertices(string objPath)
        {
            var vertices = new List<Vector3>();
            
            if (!File.Exists(objPath))
            {
                ConsoleLogger.WriteLine($"Warning: File not found: {objPath}");
                return vertices;
            }

            foreach (var line in File.ReadAllLines(objPath))
            {
                if (line.StartsWith("v "))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 4 && 
                        float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float x) &&
                        float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float y) &&
                        float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
                    {
                        vertices.Add(new Vector3(x, y, z));
                    }
                }
            }

            return vertices;
        }

        public static SpatialCorrelationResult AnalyzeObjectCorrelation(
            List<Vector3> mscnVertices, 
            List<Vector3> wmoVertices, 
            float tolerance,
            string? groupFilter)
        {
            var result = new SpatialCorrelationResult();
            var matchedMscnIndices = new HashSet<int>();
            var matchedWmoIndices = new HashSet<int>();

            // Safety check for empty input
            if (mscnVertices.Count == 0 || wmoVertices.Count == 0)
            {
                ConsoleLogger.WriteLine("Warning: Empty vertex set detected. Skipping correlation.");
                
                // Update result with correct statistics for empty case
                result.TotalMatches = 0;
                result.TotalMscnVertices = mscnVertices.Count;
                result.TotalWmoVertices = wmoVertices.Count;
                result.MatchedMscnVertices = 0;
                result.MatchedWmoVertices = 0;
                result.MatchPercentage = 0;
                
                // Calculate spatial bounds (or empty bounds if no vertices)
                result.MscnBounds = mscnVertices.Count > 0 ? CalculateBounds(mscnVertices) : (Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero);
                result.WmoBounds = wmoVertices.Count > 0 ? CalculateBounds(wmoVertices) : (Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero);
                
                return result;
            }

            ConsoleLogger.WriteLine($"Analyzing object-level correlation between {mscnVertices.Count} MSCN and {wmoVertices.Count} WMO vertices...");
            
            // Set a timeout for the entire correlation process
            var timeout = TimeSpan.FromSeconds(60); // 1 minute timeout
            var correlationStartTime = DateTime.Now;
            
            // Calculate spatial bounds once
            var mscnBounds = CalculateBounds(mscnVertices);
            var wmoBounds = CalculateBounds(wmoVertices);
            
            ConsoleLogger.WriteLine($"MSCN bounds: Center=({mscnBounds.Center.X:F2}, {mscnBounds.Center.Y:F2}, {mscnBounds.Center.Z:F2}), Size=({mscnBounds.Size.X:F2}, {mscnBounds.Size.Y:F2}, {mscnBounds.Size.Z:F2})");
            ConsoleLogger.WriteLine($"WMO bounds: Center=({wmoBounds.Center.X:F2}, {wmoBounds.Center.Y:F2}, {wmoBounds.Center.Z:F2}), Size=({wmoBounds.Size.X:F2}, {wmoBounds.Size.Y:F2}, {wmoBounds.Size.Z:F2})");
            
            // Use a reasonable cluster radius - not too large
            float clusterRadius = Math.Min(tolerance * 2.0f, 10.0f); // Cap at 10 units maximum
            
            // Create MSCN object clusters using spatial clustering
            ConsoleLogger.WriteLine($"Creating MSCN anchor clusters with radius {clusterRadius}...");
            
            // Limit processing to manageable vertex count
            var maxVerticesToProcess = Math.Min(mscnVertices.Count, 10000); // Cap at 10k vertices
            List<Vector3> verticesToProcess;
            
            if (mscnVertices.Count > maxVerticesToProcess)
            {
                ConsoleLogger.WriteLine($"Large dataset detected. Sampling {maxVerticesToProcess} vertices from {mscnVertices.Count} total vertices.");
                verticesToProcess = new List<Vector3>(maxVerticesToProcess);
                var step = mscnVertices.Count / maxVerticesToProcess;
                for (int i = 0; i < mscnVertices.Count; i += step)
                {
                    verticesToProcess.Add(mscnVertices[i]);
                    if (verticesToProcess.Count >= maxVerticesToProcess)
                        break;
                }
            }
            else
            {
                verticesToProcess = mscnVertices;
            }
            
            var mscnClusters = CreateSpatialClusters(verticesToProcess, clusterRadius);
            ConsoleLogger.WriteLine($"Created {mscnClusters.Count} MSCN clusters");
            
            // Build spatial hash grid for efficient WMO vertex lookup once
            var wmoSpatialGrid = BuildSpatialHashGrid(wmoVertices, tolerance);
            
            // Cap number of clusters to process for performance
            var maxClustersToProcess = Math.Min(mscnClusters.Count, 100); // Cap at 100 clusters
            if (mscnClusters.Count > maxClustersToProcess)
            {
                ConsoleLogger.WriteLine($"Processing first {maxClustersToProcess} clusters out of {mscnClusters.Count} total");
                mscnClusters = mscnClusters.Take(maxClustersToProcess).ToList();
            }
            
            // Process clusters with timeout protection
            int objectMatches = 0;
            int processedClusters = 0;
            int totalMatchesFound = 0;
            
            foreach (var cluster in mscnClusters)
            {
                // Check for timeout
                if (DateTime.Now - correlationStartTime > timeout)
                {
                    ConsoleLogger.WriteLine($"Correlation timeout after {timeout.TotalSeconds} seconds. Processed {processedClusters}/{mscnClusters.Count} clusters.");
                    break;
                }
                
                processedClusters++;
                if (processedClusters % 10 == 0)
                {
                    ConsoleLogger.WriteLine($"Processing cluster {processedClusters}/{mscnClusters.Count}...");
                }
                
                var clusterCenter = cluster.center;
                var clusterSize = cluster.size;
                
                // Skip very large clusters (likely noise or poorly formed)
                if (clusterSize.Length() > tolerance * 20)
                {
                    ConsoleLogger.WriteLine($"Skipping oversized cluster at ({clusterCenter.X:F2}, {clusterCenter.Y:F2}, {clusterCenter.Z:F2}) with size {clusterSize.Length():F2}");
                    continue;
                }
                
                // Only check clusters that might reasonably overlap with WMO bounds
                var distanceToBounds = Math.Max(0, Vector3.Distance(clusterCenter, wmoBounds.Center) - 
                                             (clusterSize.Length() / 2 + wmoBounds.Size.Length() / 2));
                
                if (distanceToBounds > tolerance * 5) // Skip clusters that are too far from WMO bounds
                    continue;
                
                objectMatches++;
                
                // Find MSCN vertices efficiently using index lookup rather than distance check
                // Pre-compute cluster bounds
                var clusterBoundsMin = new Vector3(
                    clusterCenter.X - clusterSize.X/2,
                    clusterCenter.Y - clusterSize.Y/2,
                    clusterCenter.Z - clusterSize.Z/2
                );
                var clusterBoundsMax = new Vector3(
                    clusterCenter.X + clusterSize.X/2,
                    clusterCenter.Y + clusterSize.Y/2,
                    clusterCenter.Z + clusterSize.Z/2
                );
                
                // Add a small padding to ensure we catch all relevant vertices
                clusterBoundsMin -= new Vector3(tolerance);
                clusterBoundsMax += new Vector3(tolerance);
                
                // Build spatial hash for this specific cluster region (for MSCN vertices)
                var clusterMscnGrid = new Dictionary<(int, int, int), List<int>>();
                var cellSize = tolerance; // Use tolerance as cell size
                
                // Only process a limited batch of vertices per cluster
                int mscnProcessed = 0;
                int matchesInCluster = 0;
                int maxVerticesPerCluster = 500; // Limit vertices per cluster
                
                // Process original MSCN vertices (not the sampled ones) but use spatial filtering
                for (int mscnIndex = 0; mscnIndex < mscnVertices.Count && mscnProcessed < maxVerticesPerCluster; mscnIndex++)
                {
                    if (matchedMscnIndices.Contains(mscnIndex)) continue;
                    
                    var mscnVertex = mscnVertices[mscnIndex];
                    
                    // Skip if outside cluster bounds (quick reject)
                    if (mscnVertex.X < clusterBoundsMin.X || mscnVertex.X > clusterBoundsMax.X ||
                        mscnVertex.Y < clusterBoundsMin.Y || mscnVertex.Y > clusterBoundsMax.Y ||
                        mscnVertex.Z < clusterBoundsMin.Z || mscnVertex.Z > clusterBoundsMax.Z)
                        continue;
                    
                    mscnProcessed++;
                    
                    // Find nearest WMO vertex using spatial grid
                    var (nearestDistance, nearestWmoIndex) = FindNearestWmoVertex(mscnVertex, wmoVertices, wmoSpatialGrid, tolerance);
                    
                    // Accept match if within tolerance and not already matched
                    if (nearestDistance <= tolerance && nearestWmoIndex >= 0 && !matchedWmoIndices.Contains(nearestWmoIndex))
                    {
                        // Found a valid vertex match
                        matchedMscnIndices.Add(mscnIndex);
                        matchedWmoIndices.Add(nearestWmoIndex);
                        result.Matches.Add(new SpatialMatch 
                        { 
                            MscnIndex = mscnIndex, 
                            MscnVertex = mscnVertex,
                            WmoIndex = nearestWmoIndex, 
                            WmoVertex = wmoVertices[nearestWmoIndex],
                            Distance = nearestDistance 
                        });
                        
                        matchesInCluster++;
                        totalMatchesFound++;
                        
                        // Early exit if we found enough matches in this cluster
                        if (matchesInCluster >= 20) // Cap matches per cluster
                            break;
                    }
                }
                
                // Early exit check for overall matches
                if (totalMatchesFound > 1000) // Cap total matches
                {
                    ConsoleLogger.WriteLine($"Found {totalMatchesFound} matches, stopping early for performance");
                    break;
                }
            }
            
            var processingTime = (DateTime.Now - correlationStartTime).TotalSeconds;
            ConsoleLogger.WriteLine($"Correlation completed in {processingTime:F1} seconds");
            ConsoleLogger.WriteLine($"Processed {processedClusters} out of {mscnClusters.Count} clusters");
            ConsoleLogger.WriteLine($"Found {objectMatches} object-level matches out of {mscnClusters.Count} MSCN clusters");
            ConsoleLogger.WriteLine($"Found {result.Matches.Count} vertex-to-vertex matches within matched objects");
            
            // Update result with correct statistics
            result.TotalMatches = result.Matches.Count;
            result.TotalMscnVertices = mscnVertices.Count;
            result.TotalWmoVertices = wmoVertices.Count;
            result.MatchedMscnVertices = matchedMscnIndices.Count;
            result.MatchedWmoVertices = matchedWmoIndices.Count;
            result.MatchPercentage = result.TotalMscnVertices > 0 ? (float)result.MatchedMscnVertices / result.TotalMscnVertices * 100f : 0f;

            // Store spatial bounds
            result.MscnBounds = mscnBounds;
            result.WmoBounds = wmoBounds;

            return result;
        }

        private static (Vector3 Min, Vector3 Max, Vector3 Center, Vector3 Size) CalculateBounds(List<Vector3> vertices)
        {
            if (vertices.Count == 0)
                return (Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero);

            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);

            foreach (var vertex in vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            var center = (min + max) * 0.5f;
            var size = max - min;

            return (min, max, center, size);
        }

        private static async Task GenerateComparisonReport(
            SpatialCorrelationResult originalResult,
            SpatialCorrelationResult rotatedResult,
            List<Vector3> mscnVertices,
            List<Vector3> wmoVertices,
            float tolerance,
            string reportPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath) ?? ".");

            using var writer = new StreamWriter(reportPath);

            await writer.WriteLineAsync("=== MSCN-WMO Geometry Comparison Report ===");
            await writer.WriteLineAsync($"Generated: {DateTime.Now}");
            await writer.WriteLineAsync($"Tolerance: {tolerance:F1} units");
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== SUMMARY ===");
            await writer.WriteLineAsync($"MSCN Vertices: {originalResult.TotalMscnVertices:N0}");
            await writer.WriteLineAsync($"WMO Vertices: {originalResult.TotalWmoVertices:N0}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync("=== ORIGINAL COORDINATES CORRELATION ===");
            await writer.WriteLineAsync($"Spatial Matches: {originalResult.Matches.Count:N0}");
            await writer.WriteLineAsync($"MSCN Match Rate: {originalResult.MatchPercentage:F1}% ({originalResult.MatchedMscnVertices}/{originalResult.TotalMscnVertices})");
            await writer.WriteLineAsync($"WMO Coverage: {(float)originalResult.MatchedWmoVertices / originalResult.TotalWmoVertices * 100f:F1}% ({originalResult.MatchedWmoVertices}/{originalResult.TotalWmoVertices})");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync("=== 180° ROTATED COORDINATES CORRELATION ===");
            await writer.WriteLineAsync($"Spatial Matches: {rotatedResult.Matches.Count:N0}");
            await writer.WriteLineAsync($"MSCN Match Rate: {rotatedResult.MatchPercentage:F1}% ({rotatedResult.MatchedMscnVertices}/{rotatedResult.TotalMscnVertices})");
            await writer.WriteLineAsync($"WMO Coverage: {(float)rotatedResult.MatchedWmoVertices / rotatedResult.TotalWmoVertices * 100f:F1}% ({rotatedResult.MatchedWmoVertices}/{rotatedResult.TotalWmoVertices})");
            await writer.WriteLineAsync();
            
            // Determine which approach worked better
            var betterResult = rotatedResult.MatchPercentage > originalResult.MatchPercentage ? "180° ROTATED" : "ORIGINAL";
            var betterPercentage = Math.Max(rotatedResult.MatchPercentage, originalResult.MatchPercentage);
            await writer.WriteLineAsync($"=== BEST ALIGNMENT: {betterResult} ({betterPercentage:F1}% correlation) ===");
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== SPATIAL BOUNDS ===");
            await writer.WriteLineAsync($"MSCN Bounds: {originalResult.MscnBounds.Min:F2} to {originalResult.MscnBounds.Max:F2}");
            await writer.WriteLineAsync($"MSCN Center: {originalResult.MscnBounds.Center:F2}");
            await writer.WriteLineAsync($"MSCN Size: {originalResult.MscnBounds.Size:F2}");
            await writer.WriteLineAsync();
            await writer.WriteLineAsync($"WMO Bounds: {originalResult.WmoBounds.Min:F2} to {originalResult.WmoBounds.Max:F2}");
            await writer.WriteLineAsync($"WMO Center: {originalResult.WmoBounds.Center:F2}");
            await writer.WriteLineAsync($"WMO Size: {originalResult.WmoBounds.Size:F2}");
            await writer.WriteLineAsync();

            // Show detailed statistics for the better result
            var bestResult = rotatedResult.MatchPercentage > originalResult.MatchPercentage ? rotatedResult : originalResult;
            var bestResultType = rotatedResult.MatchPercentage > originalResult.MatchPercentage ? "180° ROTATED" : "ORIGINAL";
            
            if (bestResult.Matches.Count > 0)
            {
                await writer.WriteLineAsync($"=== DISTANCE STATISTICS ({bestResultType}) ===");
                var distances = bestResult.Matches.Select(m => m.Distance).OrderBy(d => d).ToList();
                await writer.WriteLineAsync($"Min Distance: {distances.First():F3}");
                await writer.WriteLineAsync($"Max Distance: {distances.Last():F3}");
                await writer.WriteLineAsync($"Average Distance: {distances.Average():F3}");
                await writer.WriteLineAsync($"Median Distance: {distances[distances.Count / 2]:F3}");
                await writer.WriteLineAsync();

                await writer.WriteLineAsync($"=== TOP 20 CLOSEST MATCHES ({bestResultType}) ===");
                var topMatches = bestResult.Matches.OrderBy(m => m.Distance).Take(20);
                foreach (var match in topMatches)
                {
                    await writer.WriteLineAsync($"MSCN[{match.MscnIndex}] {match.MscnVertex:F2} <-> WMO[{match.WmoIndex}] {match.WmoVertex:F2} (dist: {match.Distance:F3})");
                }
            }

            ConsoleLogger.WriteLine($"Report generated - Original: {originalResult.Matches.Count} matches ({originalResult.MatchPercentage:F1}%), Rotated: {rotatedResult.Matches.Count} matches ({rotatedResult.MatchPercentage:F1}%)");
        }
        
        private static async Task GenerateTransformationReport(
            List<(string name, SpatialCorrelationResult result, List<Vector3> transformedVertices)> transformationResults,
            List<Vector3> mscnVertices,
            List<Vector3> wmoVertices,
            float tolerance,
            string reportPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath) ?? ".");

            using var writer = new StreamWriter(reportPath);

            await writer.WriteLineAsync("=== MSCN-WMO Coordinate Transformation Analysis Report ===");
            await writer.WriteLineAsync($"Generated: {DateTime.Now}");
            await writer.WriteLineAsync($"Tolerance: {tolerance:F1} units");
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== SUMMARY ===");
            await writer.WriteLineAsync($"MSCN Vertices: {mscnVertices.Count:N0}");
            await writer.WriteLineAsync($"WMO Vertices: {wmoVertices.Count:N0}");
            await writer.WriteLineAsync($"Transformations Tested: {transformationResults.Count}");
            await writer.WriteLineAsync();
            
            // Find best transformation
            var bestTransformation = transformationResults.First();
            await writer.WriteLineAsync($"=== BEST TRANSFORMATION: {bestTransformation.name} ===");
            await writer.WriteLineAsync($"Spatial Matches: {bestTransformation.result.Matches.Count:N0}");
            await writer.WriteLineAsync($"MSCN Match Rate: {bestTransformation.result.MatchPercentage:F1}% ({bestTransformation.result.MatchedMscnVertices}/{bestTransformation.result.TotalMscnVertices})");
            await writer.WriteLineAsync($"WMO Coverage: {(float)bestTransformation.result.MatchedWmoVertices / bestTransformation.result.TotalWmoVertices * 100f:F1}% ({bestTransformation.result.MatchedWmoVertices}/{bestTransformation.result.TotalWmoVertices})");
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== ALL TRANSFORMATION RESULTS (Best to Worst) ===");
            foreach (var (name, result, _) in transformationResults)
            {
                await writer.WriteLineAsync($"{name,-25}: {result.Matches.Count,6} matches ({result.MatchPercentage,5:F1}%)");
            }
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== SPATIAL BOUNDS (Original MSCN) ===");
            var originalResult = transformationResults.First(t => t.name == "Original").result;
            await writer.WriteLineAsync($"MSCN Bounds: {originalResult.MscnBounds.Min:F2} to {originalResult.MscnBounds.Max:F2}");
            await writer.WriteLineAsync($"MSCN Center: {originalResult.MscnBounds.Center:F2}");
            await writer.WriteLineAsync($"MSCN Size: {originalResult.MscnBounds.Size:F2}");
            await writer.WriteLineAsync();
            await writer.WriteLineAsync($"WMO Bounds: {originalResult.WmoBounds.Min:F2} to {originalResult.WmoBounds.Max:F2}");
            await writer.WriteLineAsync($"WMO Center: {originalResult.WmoBounds.Center:F2}");
            await writer.WriteLineAsync($"WMO Size: {originalResult.WmoBounds.Size:F2}");
            await writer.WriteLineAsync();

            // Show detailed statistics for the best result
            if (bestTransformation.result.Matches.Count > 0)
            {
                await writer.WriteLineAsync($"=== DISTANCE STATISTICS ({bestTransformation.name}) ===");
                var distances = bestTransformation.result.Matches.Select(m => m.Distance).OrderBy(d => d).ToList();
                await writer.WriteLineAsync($"Min Distance: {distances.First():F3}");
                await writer.WriteLineAsync($"Max Distance: {distances.Last():F3}");
                await writer.WriteLineAsync($"Average Distance: {distances.Average():F3}");
                await writer.WriteLineAsync($"Median Distance: {distances[distances.Count / 2]:F3}");
                await writer.WriteLineAsync();

                await writer.WriteLineAsync($"=== TOP 20 CLOSEST MATCHES ({bestTransformation.name}) ===");
                var topMatches = bestTransformation.result.Matches.OrderBy(m => m.Distance).Take(20);
                foreach (var match in topMatches)
                {
                    await writer.WriteLineAsync($"MSCN[{match.MscnIndex}] {match.MscnVertex:F2} <-> WMO[{match.WmoIndex}] {match.WmoVertex:F2} (dist: {match.Distance:F3})");
                }
            }
            else
            {
                await writer.WriteLineAsync("=== NO MATCHES FOUND ===");
                await writer.WriteLineAsync("No spatial correlation detected with any coordinate transformation.");
                await writer.WriteLineAsync("This suggests either:");
                await writer.WriteLineAsync("1. MSCN and WMO data represent different coordinate spaces");
                await writer.WriteLineAsync("2. A more complex transformation is required");
                await writer.WriteLineAsync("3. The tolerance value needs adjustment");
                await writer.WriteLineAsync("4. The datasets represent different geometric features");
            }

            var bestPercentage = bestTransformation.result.MatchPercentage;
            ConsoleLogger.WriteLine($"Transformation report generated - Best: {bestTransformation.name} ({bestPercentage:F1}% correlation)");
        }
        
        private static async Task GenerateSpatialFilteringReport(
            (SpatialCorrelationResult originalResult, SpatialCorrelationResult filteredResult, SpatialCorrelationResult rotatedFilteredResult, List<Vector3> filteredMscnVertices, List<Vector3> rotatedFilteredMscnVertices) spatialResults,
            List<Vector3> mscnVertices,
            List<Vector3> wmoVertices,
            float tolerance,
            string reportPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath) ?? ".");

            using var writer = new StreamWriter(reportPath);

            await writer.WriteLineAsync("=== MSCN-WMO Spatial Filtering Analysis Report ===");
            await writer.WriteLineAsync($"Generated: {DateTime.Now}");
            await writer.WriteLineAsync($"Tolerance: {tolerance:F1} units");
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== SUMMARY ===");
            await writer.WriteLineAsync($"Original MSCN Vertices: {mscnVertices.Count:N0}");
            await writer.WriteLineAsync($"Filtered MSCN Vertices: {spatialResults.filteredMscnVertices.Count:N0}");
            await writer.WriteLineAsync($"WMO Vertices: {wmoVertices.Count:N0}");
            await writer.WriteLineAsync($"Filter Reduction: {(1.0f - (float)spatialResults.filteredMscnVertices.Count / mscnVertices.Count) * 100:F1}%");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync("=== CORRELATION RESULTS ===");
            await writer.WriteLineAsync($"Original (All MSCN): {spatialResults.originalResult.MatchPercentage:F1}% ({spatialResults.originalResult.Matches.Count:N0} matches)");
            await writer.WriteLineAsync($"Spatially Filtered: {spatialResults.filteredResult.MatchPercentage:F1}% ({spatialResults.filteredResult.Matches.Count:N0} matches)");
            await writer.WriteLineAsync($"Filtered + 180° Rotated: {spatialResults.rotatedFilteredResult.MatchPercentage:F1}% ({spatialResults.rotatedFilteredResult.Matches.Count:N0} matches)");
            await writer.WriteLineAsync();
            
            // Determine best approach
            var bestResult = spatialResults.originalResult;
            var bestName = "Original (All MSCN)";
            
            if (spatialResults.filteredResult.MatchPercentage > bestResult.MatchPercentage)
            {
                bestResult = spatialResults.filteredResult;
                bestName = "Spatially Filtered";
            }
            
            if (spatialResults.rotatedFilteredResult.MatchPercentage > bestResult.MatchPercentage)
            {
                bestResult = spatialResults.rotatedFilteredResult;
                bestName = "Filtered + 180° Rotated";
            }
            
            await writer.WriteLineAsync($"=== BEST APPROACH: {bestName} ({bestResult.MatchPercentage:F1}% correlation) ===");
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== SPATIAL BOUNDS ===");
            await writer.WriteLineAsync($"Original MSCN Bounds: {spatialResults.originalResult.MscnBounds.Min:F2} to {spatialResults.originalResult.MscnBounds.Max:F2}");
            await writer.WriteLineAsync($"Original MSCN Center: {spatialResults.originalResult.MscnBounds.Center:F2}");
            await writer.WriteLineAsync($"Original MSCN Size: {spatialResults.originalResult.MscnBounds.Size:F2}");
            await writer.WriteLineAsync();
            
            if (spatialResults.filteredMscnVertices.Count > 0)
            {
                var filteredBounds = CalculateBounds(spatialResults.filteredMscnVertices);
                await writer.WriteLineAsync($"Filtered MSCN Bounds: {filteredBounds.Min:F2} to {filteredBounds.Max:F2}");
                await writer.WriteLineAsync($"Filtered MSCN Center: {filteredBounds.Center:F2}");
                await writer.WriteLineAsync($"Filtered MSCN Size: {filteredBounds.Size:F2}");
                await writer.WriteLineAsync();
            }
            
            await writer.WriteLineAsync($"WMO Bounds: {spatialResults.originalResult.WmoBounds.Min:F2} to {spatialResults.originalResult.WmoBounds.Max:F2}");
            await writer.WriteLineAsync($"WMO Center: {spatialResults.originalResult.WmoBounds.Center:F2}");
            await writer.WriteLineAsync($"WMO Size: {spatialResults.originalResult.WmoBounds.Size:F2}");
            await writer.WriteLineAsync();

            // Show detailed statistics for the best result
            if (bestResult.Matches.Count > 0)
            {
                await writer.WriteLineAsync($"=== DISTANCE STATISTICS ({bestName}) ===");
                var distances = bestResult.Matches.Select(m => m.Distance).OrderBy(d => d).ToList();
                await writer.WriteLineAsync($"Min Distance: {distances.First():F3}");
                await writer.WriteLineAsync($"Max Distance: {distances.Last():F3}");
                await writer.WriteLineAsync($"Average Distance: {distances.Average():F3}");
                await writer.WriteLineAsync($"Median Distance: {distances[distances.Count / 2]:F3}");
                await writer.WriteLineAsync();

                await writer.WriteLineAsync($"=== TOP 20 CLOSEST MATCHES ({bestName}) ===");
                var topMatches = bestResult.Matches.OrderBy(m => m.Distance).Take(20);
                foreach (var match in topMatches)
                {
                    await writer.WriteLineAsync($"MSCN[{match.MscnIndex}] {match.MscnVertex:F2} <-> WMO[{match.WmoIndex}] {match.WmoVertex:F2} (dist: {match.Distance:F3})");
                }
            }
            else
            {
                await writer.WriteLineAsync("=== NO MATCHES FOUND ===");
                await writer.WriteLineAsync("No spatial correlation detected even with filtering and rotation.");
                await writer.WriteLineAsync("This suggests:");
                await writer.WriteLineAsync("1. MSCN anchors represent collision data, not exact geometry");
                await writer.WriteLineAsync("2. Different tolerance values may be needed");
                await writer.WriteLineAsync("3. WMO placement transformation is more complex than 180° rotation");
                await writer.WriteLineAsync("4. MSCN and WMO represent fundamentally different data types");
            }

            var bestPercentage = bestResult.MatchPercentage;
            ConsoleLogger.WriteLine($"Spatial filtering report generated - Best: {bestName} ({bestPercentage:F1}% correlation)");
        }
        
        private static async Task GenerateNormalizationReport(
            (SpatialCorrelationResult originalResult, SpatialCorrelationResult normalizedResult, SpatialCorrelationResult rotatedNormalizedResult, List<Vector3> normalizedMscnVertices, List<Vector3> rotatedNormalizedVertices) normalizationResults,
            List<Vector3> mscnVertices,
            List<Vector3> wmoVertices,
            float tolerance,
            string reportPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath) ?? ".");

            using var writer = new StreamWriter(reportPath);

            await writer.WriteLineAsync("=== MSCN-WMO Coordinate Normalization Analysis Report ===");
            await writer.WriteLineAsync($"Generated: {DateTime.Now}");
            await writer.WriteLineAsync($"Tolerance: {tolerance:F1} units");
            await writer.WriteLineAsync();

            await writer.WriteLineAsync("=== SUMMARY ===");
            await writer.WriteLineAsync($"Original MSCN Vertices: {mscnVertices.Count:N0}");
            await writer.WriteLineAsync($"Normalized MSCN Vertices: {normalizationResults.normalizedMscnVertices.Count:N0}");
            await writer.WriteLineAsync($"WMO Vertices: {wmoVertices.Count:N0}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync("=== COORDINATE TRANSFORMATION PROCESS ===");
            var originalBounds = CalculateBounds(mscnVertices);
            var wmoBounds = CalculateBounds(wmoVertices);
            var normalizedBounds = CalculateBounds(normalizationResults.normalizedMscnVertices);
            
            await writer.WriteLineAsync($"Step 1 - Original MSCN (World Space):");
            await writer.WriteLineAsync($"  Center: ({originalBounds.Center.X:F2}, {originalBounds.Center.Y:F2}, {originalBounds.Center.Z:F2})");
            await writer.WriteLineAsync($"  Size: ({originalBounds.Size.X:F2}, {originalBounds.Size.Y:F2}, {originalBounds.Size.Z:F2})");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync($"Step 2 - WMO Reference (Local Space):");
            await writer.WriteLineAsync($"  Center: ({wmoBounds.Center.X:F2}, {wmoBounds.Center.Y:F2}, {wmoBounds.Center.Z:F2})");
            await writer.WriteLineAsync($"  Size: ({wmoBounds.Size.X:F2}, {wmoBounds.Size.Y:F2}, {wmoBounds.Size.Z:F2})");
            await writer.WriteLineAsync();
            
            var scaleFactorX = wmoBounds.Size.X / originalBounds.Size.X;
            var scaleFactorY = wmoBounds.Size.Y / originalBounds.Size.Y;
            var scaleFactorZ = wmoBounds.Size.Z / originalBounds.Size.Z;
            var uniformScale = (scaleFactorX + scaleFactorY) / 2.0f;
            
            await writer.WriteLineAsync($"Step 3 - Normalization Transform:");
            await writer.WriteLineAsync($"  Translation: -{originalBounds.Center.X:F2}, -{originalBounds.Center.Y:F2}, -{originalBounds.Center.Z:F2}");
            await writer.WriteLineAsync($"  Scale Factors: X={scaleFactorX:F4}, Y={scaleFactorY:F4}, Z={scaleFactorZ:F4}");
            await writer.WriteLineAsync($"  Uniform Scale: {uniformScale:F4}");
            await writer.WriteLineAsync($"  Final Translation: +{wmoBounds.Center.X:F2}, +{wmoBounds.Center.Y:F2}, +{wmoBounds.Center.Z:F2}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync($"Step 4 - Normalized MSCN (Local Space):");
            await writer.WriteLineAsync($"  Center: ({normalizedBounds.Center.X:F2}, {normalizedBounds.Center.Y:F2}, {normalizedBounds.Center.Z:F2})");
            await writer.WriteLineAsync($"  Size: ({normalizedBounds.Size.X:F2}, {normalizedBounds.Size.Y:F2}, {normalizedBounds.Size.Z:F2})");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync("=== CORRELATION RESULTS ===");
            await writer.WriteLineAsync($"Original (World Space): {normalizationResults.originalResult.MatchPercentage:F1}% ({normalizationResults.originalResult.Matches.Count:N0} matches)");
            await writer.WriteLineAsync($"Normalized (Local Space): {normalizationResults.normalizedResult.MatchPercentage:F1}% ({normalizationResults.normalizedResult.Matches.Count:N0} matches)");
            await writer.WriteLineAsync($"Normalized + 180° Rotated: {normalizationResults.rotatedNormalizedResult.MatchPercentage:F1}% ({normalizationResults.rotatedNormalizedResult.Matches.Count:N0} matches)");
            await writer.WriteLineAsync();
            
            // Determine best approach
            var bestResult = normalizationResults.originalResult;
            var bestName = "Original (World Space)";
            
            if (normalizationResults.normalizedResult.MatchPercentage > bestResult.MatchPercentage)
            {
                bestResult = normalizationResults.normalizedResult;
                bestName = "Normalized (Local Space)";
            }
            
            if (normalizationResults.rotatedNormalizedResult.MatchPercentage > bestResult.MatchPercentage)
            {
                bestResult = normalizationResults.rotatedNormalizedResult;
                bestName = "Normalized + 180° Rotated";
            }
            
            await writer.WriteLineAsync($"=== BEST APPROACH: {bestName} ({bestResult.MatchPercentage:F1}% correlation) ===");
            await writer.WriteLineAsync();

            // Show detailed statistics for the best result
            if (bestResult.Matches.Count > 0)
            {
                await writer.WriteLineAsync($"=== DISTANCE STATISTICS ({bestName}) ===");
                var distances = bestResult.Matches.Select(m => m.Distance).OrderBy(d => d).ToList();
                await writer.WriteLineAsync($"Min Distance: {distances.First():F3}");
                await writer.WriteLineAsync($"Max Distance: {distances.Last():F3}");
                await writer.WriteLineAsync($"Average Distance: {distances.Average():F3}");
                await writer.WriteLineAsync($"Median Distance: {distances[distances.Count / 2]:F3}");
                await writer.WriteLineAsync();

                await writer.WriteLineAsync($"=== TOP 20 CLOSEST MATCHES ({bestName}) ===");
                var topMatches = bestResult.Matches.OrderBy(m => m.Distance).Take(20);
                foreach (var match in topMatches)
                {
                    await writer.WriteLineAsync($"MSCN[{match.MscnIndex}] {match.MscnVertex:F2} <-> WMO[{match.WmoIndex}] {match.WmoVertex:F2} (dist: {match.Distance:F3})");
                }
            }
            else
            {
                await writer.WriteLineAsync("=== NO MATCHES FOUND ===");
                await writer.WriteLineAsync("No spatial correlation detected even after coordinate normalization.");
                await writer.WriteLineAsync("This suggests:");
                await writer.WriteLineAsync("1. MSCN anchors represent collision/anchor points, not exact geometry vertices");
                await writer.WriteLineAsync("2. The datasets may represent different levels of detail or abstraction");
                await writer.WriteLineAsync("3. Additional coordinate transformations may be needed (rotation angles, axis swaps)");
                await writer.WriteLineAsync("4. The tolerance value may need adjustment for collision-to-geometry matching");
                await writer.WriteLineAsync("5. MSCN and WMO may represent fundamentally different spatial features");
            }

            var bestPercentage = bestResult.MatchPercentage;
            ConsoleLogger.WriteLine($"Normalization report generated - Best: {bestName} ({bestPercentage:F1}% correlation)");
        }
        
        private static async Task ExportVisualizationObjects(List<Vector3> mscnVertices, List<Vector3> wmoVertices, string outputDir, string? groupFilter)
        {
            ConsoleLogger.WriteLine("Exporting visualization objects...");
            
            // Export raw MSCN vertices as point cloud
            var mscnObjPath = Path.Combine(outputDir, "mscn_anchors.obj");
            await ExportVerticesAsObj(mscnVertices, mscnObjPath, "MSCN Anchors");
            
            // Export raw WMO vertices as point cloud
            var wmoObjPath = Path.Combine(outputDir, $"wmo_vertices{(groupFilter != null ? $"_{groupFilter}" : "")}.obj");
            await ExportVerticesAsObj(wmoVertices, wmoObjPath, $"WMO Vertices{(groupFilter != null ? $" ({groupFilter})" : "")}");
            
            // Create and export MSCN clusters
            var tolerance = 10.0f; // Use larger tolerance for clustering visualization
            var mscnClusters = CreateSpatialClusters(mscnVertices, tolerance);
            var clustersObjPath = Path.Combine(outputDir, "mscn_clusters.obj");
            await ExportClustersAsObj(mscnClusters, clustersObjPath);
            
            // Export WMO bounding box
            var wmoBounds = CalculateBounds(wmoVertices);
            var boundsObjPath = Path.Combine(outputDir, "wmo_bounds.obj");
            await ExportBoundsAsObj(wmoBounds, boundsObjPath);
            
            ConsoleLogger.WriteLine($"Visualization objects exported to {outputDir}:");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(mscnObjPath)} ({mscnVertices.Count} vertices)");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(wmoObjPath)} ({wmoVertices.Count} vertices)");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(clustersObjPath)} ({mscnClusters.Count} clusters)");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(boundsObjPath)} (bounding box)");
        }
        
        private static async Task ExportRotatedVisualizationObjects(List<Vector3> rotatedMscnVertices, List<Vector3> wmoVertices, string outputDir, string? groupFilter)
        {
            ConsoleLogger.WriteLine("Exporting rotated visualization objects...");
            
            // Export rotated MSCN vertices as point cloud
            var rotatedMscnObjPath = Path.Combine(outputDir, "mscn_anchors_rotated180.obj");
            await ExportVerticesAsObj(rotatedMscnVertices, rotatedMscnObjPath, "MSCN Anchors (180° Rotated)");
            
            // Create and export rotated MSCN clusters
            var tolerance = 10.0f; // Use larger tolerance for clustering visualization
            var rotatedMscnClusters = CreateSpatialClusters(rotatedMscnVertices, tolerance);
            var rotatedClustersObjPath = Path.Combine(outputDir, "mscn_clusters_rotated180.obj");
            await ExportClustersAsObj(rotatedMscnClusters, rotatedClustersObjPath);
            
            ConsoleLogger.WriteLine($"Rotated visualization objects exported:");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(rotatedMscnObjPath)} ({rotatedMscnVertices.Count} vertices)");
            ConsoleLogger.WriteLine($"  - {Path.GetFileName(rotatedClustersObjPath)} ({rotatedMscnClusters.Count} clusters)");
        }
        
        private static async Task ExportVerticesAsObj(List<Vector3> vertices, string filePath, string comment)
        {
            using var writer = new StreamWriter(filePath);
            await writer.WriteLineAsync($"# {comment}");
            await writer.WriteLineAsync($"# {vertices.Count} vertices");
            await writer.WriteLineAsync();
            
            foreach (var vertex in vertices)
            {
                await writer.WriteLineAsync($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
        }
        
        private static async Task ExportClustersAsObj(List<(Vector3 center, Vector3 size, int count)> clusters, string filePath)
        {
            using var writer = new StreamWriter(filePath);
            await writer.WriteLineAsync($"# MSCN Clusters");
            await writer.WriteLineAsync($"# {clusters.Count} clusters");
            await writer.WriteLineAsync();
            
            int vertexIndex = 1;
            foreach (var (center, size, count) in clusters)
            {
                await writer.WriteLineAsync($"# Cluster: center=({center.X:F2}, {center.Y:F2}, {center.Z:F2}), size=({size.X:F2}, {size.Y:F2}, {size.Z:F2}), vertices={count}");
                
                // Create a simple bounding box for each cluster
                var min = center - size * 0.5f;
                var max = center + size * 0.5f;
                
                // 8 vertices of bounding box
                await writer.WriteLineAsync($"v {min.X:F6} {min.Y:F6} {min.Z:F6}");
                await writer.WriteLineAsync($"v {max.X:F6} {min.Y:F6} {min.Z:F6}");
                await writer.WriteLineAsync($"v {max.X:F6} {max.Y:F6} {min.Z:F6}");
                await writer.WriteLineAsync($"v {min.X:F6} {max.Y:F6} {min.Z:F6}");
                await writer.WriteLineAsync($"v {min.X:F6} {min.Y:F6} {max.Z:F6}");
                await writer.WriteLineAsync($"v {max.X:F6} {min.Y:F6} {max.Z:F6}");
                await writer.WriteLineAsync($"v {max.X:F6} {max.Y:F6} {max.Z:F6}");
                await writer.WriteLineAsync($"v {min.X:F6} {max.Y:F6} {max.Z:F6}");
                
                // 12 edges of bounding box (as lines)
                var baseIdx = vertexIndex;
                await writer.WriteLineAsync($"l {baseIdx} {baseIdx + 1}");
                await writer.WriteLineAsync($"l {baseIdx + 1} {baseIdx + 2}");
                await writer.WriteLineAsync($"l {baseIdx + 2} {baseIdx + 3}");
                await writer.WriteLineAsync($"l {baseIdx + 3} {baseIdx}");
                await writer.WriteLineAsync($"l {baseIdx + 4} {baseIdx + 5}");
                await writer.WriteLineAsync($"l {baseIdx + 5} {baseIdx + 6}");
                await writer.WriteLineAsync($"l {baseIdx + 6} {baseIdx + 7}");
                await writer.WriteLineAsync($"l {baseIdx + 7} {baseIdx + 4}");
                await writer.WriteLineAsync($"l {baseIdx} {baseIdx + 4}");
                await writer.WriteLineAsync($"l {baseIdx + 1} {baseIdx + 5}");
                await writer.WriteLineAsync($"l {baseIdx + 2} {baseIdx + 6}");
                await writer.WriteLineAsync($"l {baseIdx + 3} {baseIdx + 7}");
                
                vertexIndex += 8;
            }
        }
        
        private static async Task ExportBoundsAsObj((Vector3 Min, Vector3 Max, Vector3 Center, Vector3 Size) bounds, string filePath)
        {
            using var writer = new StreamWriter(filePath);
            await writer.WriteLineAsync($"# WMO Bounding Box");
            await writer.WriteLineAsync($"# Center: ({bounds.Center.X:F2}, {bounds.Center.Y:F2}, {bounds.Center.Z:F2})");
            await writer.WriteLineAsync($"# Size: ({bounds.Size.X:F2}, {bounds.Size.Y:F2}, {bounds.Size.Z:F2})");
            await writer.WriteLineAsync();
            
            // 8 vertices of bounding box
            await writer.WriteLineAsync($"v {bounds.Min.X:F6} {bounds.Min.Y:F6} {bounds.Min.Z:F6}");
            await writer.WriteLineAsync($"v {bounds.Max.X:F6} {bounds.Min.Y:F6} {bounds.Min.Z:F6}");
            await writer.WriteLineAsync($"v {bounds.Max.X:F6} {bounds.Max.Y:F6} {bounds.Min.Z:F6}");
            await writer.WriteLineAsync($"v {bounds.Min.X:F6} {bounds.Max.Y:F6} {bounds.Min.Z:F6}");
            await writer.WriteLineAsync($"v {bounds.Min.X:F6} {bounds.Min.Y:F6} {bounds.Max.Z:F6}");
            await writer.WriteLineAsync($"v {bounds.Max.X:F6} {bounds.Min.Y:F6} {bounds.Max.Z:F6}");
            await writer.WriteLineAsync($"v {bounds.Max.X:F6} {bounds.Max.Y:F6} {bounds.Max.Z:F6}");
            await writer.WriteLineAsync($"v {bounds.Min.X:F6} {bounds.Max.Y:F6} {bounds.Max.Z:F6}");
            
            // 12 edges of bounding box (as lines)
            await writer.WriteLineAsync($"l 1 2");
            await writer.WriteLineAsync($"l 2 3");
            await writer.WriteLineAsync($"l 3 4");
            await writer.WriteLineAsync($"l 4 1");
            await writer.WriteLineAsync($"l 5 6");
            await writer.WriteLineAsync($"l 6 7");
            await writer.WriteLineAsync($"l 7 8");
            await writer.WriteLineAsync($"l 8 5");
            await writer.WriteLineAsync($"l 1 5");
            await writer.WriteLineAsync($"l 2 6");
            await writer.WriteLineAsync($"l 3 7");
            await writer.WriteLineAsync($"l 4 8");
        }

        public class SpatialCorrelationResult
        {
            public List<SpatialMatch> Matches { get; } = new();
            public int TotalMatches { get; set; }
            public int TotalMscnVertices { get; set; }
            public int TotalWmoVertices { get; set; }
            public int MatchedMscnVertices { get; set; }
            public int MatchedWmoVertices { get; set; }
            public float MatchPercentage { get; set; }
            public (Vector3 Min, Vector3 Max, Vector3 Center, Vector3 Size) MscnBounds { get; set; }
            public (Vector3 Min, Vector3 Max, Vector3 Center, Vector3 Size) WmoBounds { get; set; }
        }

        public class SpatialMatch
        {
            public int MscnIndex { get; set; }
            public int WmoIndex { get; set; }
            public float Distance { get; set; }
            public Vector3 MscnVertex { get; set; }
            public Vector3 WmoVertex { get; set; }
        }
    }
}
