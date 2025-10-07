using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Analyzes PM4 data for banding/layering patterns to understand how object geometry is subdivided
    /// Based on the hypothesis that objects are spread across "bands" or "layers" in the PM4 structure
    /// </summary>
    public class Pm4DataBandingAnalyzer
    {
        public class TileAnalysis
        {
            public byte TileX { get; set; }
            public byte TileY { get; set; }
            public int LinkCount { get; set; }
            public int SurfaceCount { get; set; }
            public int VertexCount { get; set; }
            public int TriangleCount { get; set; }
            public List<uint> SurfaceKeys { get; set; } = new();
            public List<uint> ParentIds { get; set; } = new();
            public (Vector3 min, Vector3 max)? SpatialBounds { get; set; }
        }

        public class BandingAnalysisReport
        {
            public Dictionary<(byte x, byte y), TileAnalysis> TileData { get; set; } = new();
            public Dictionary<uint, List<(byte x, byte y)>> SurfaceKeyToTiles { get; set; } = new();
            public Dictionary<uint, List<(byte x, byte y)>> ParentIdToTiles { get; set; } = new();
            public List<string> UnexploredChunks { get; set; } = new();
            public Dictionary<string, int> ChunkDataSizes { get; set; } = new();
        }

        /// <summary>
        /// Analyze PM4 scene for data banding/layering patterns
        /// </summary>
        public static BandingAnalysisReport AnalyzeDataBanding(Pm4Scene scene)
        {
            ConsoleLogger.WriteLine("=== PM4 Data Banding Analysis ===");
            
            var report = new BandingAnalysisReport();
            
            // Analyze MSLK entries by tile
            if (scene.Links != null)
            {
                ConsoleLogger.WriteLine($"Analyzing {scene.Links.Count} MSLK entries...");
                
                foreach (var link in scene.Links)
                {
                    if (link.TryDecodeTileCoordinates(out int tileX, out int tileY))
                    {
                        var tileKey = ((byte)tileX, (byte)tileY);
                        
                        if (!report.TileData.ContainsKey(tileKey))
                        {
                            report.TileData[tileKey] = new TileAnalysis 
                            { 
                                TileX = (byte)tileX, 
                                TileY = (byte)tileY 
                            };
                        }
                        
                        var tileData = report.TileData[tileKey];
                        tileData.LinkCount++;
                        
                        if (!tileData.ParentIds.Contains(link.ParentId))
                        {
                            tileData.ParentIds.Add(link.ParentId);
                        }
                        
                        // Track ParentId spread across tiles
                        if (!report.ParentIdToTiles.ContainsKey(link.ParentId))
                        {
                            report.ParentIdToTiles[link.ParentId] = new List<(byte, byte)>();
                        }
                        if (!report.ParentIdToTiles[link.ParentId].Contains(tileKey))
                        {
                            report.ParentIdToTiles[link.ParentId].Add(tileKey);
                        }
                    }
                }
            }
            
            // Analyze MSUR surfaces by tile (if we can correlate them)
            if (scene.Surfaces != null)
            {
                ConsoleLogger.WriteLine($"Analyzing {scene.Surfaces.Count} MSUR surfaces...");
                
                foreach (var surface in scene.Surfaces)
                {
                    // Try to correlate surfaces with tiles via SurfaceKey patterns
                    var surfaceKey = surface.SurfaceKey;
                    
                    // Extract potential tile coordinates from SurfaceKey (hypothesis)
                    byte potentialTileX = (byte)(surfaceKey & 0xFF);
                    byte potentialTileY = (byte)((surfaceKey >> 8) & 0xFF);
                    var potentialTileKey = (potentialTileX, potentialTileY);
                    
                    if (report.TileData.ContainsKey(potentialTileKey))
                    {
                        var tileData = report.TileData[potentialTileKey];
                        tileData.SurfaceCount++;
                        
                        if (!tileData.SurfaceKeys.Contains(surfaceKey))
                        {
                            tileData.SurfaceKeys.Add(surfaceKey);
                        }
                    }
                    
                    // Track SurfaceKey spread across potential tiles
                    if (!report.SurfaceKeyToTiles.ContainsKey(surfaceKey))
                    {
                        report.SurfaceKeyToTiles[surfaceKey] = new List<(byte, byte)>();
                    }
                    if (!report.SurfaceKeyToTiles[surfaceKey].Contains(potentialTileKey))
                    {
                        report.SurfaceKeyToTiles[surfaceKey].Add(potentialTileKey);
                    }
                }
            }
            
            // Analyze spatial bounds per tile
            CalculateTileSpatialBounds(scene, report);
            
            // Look for unexplored/ignored chunk data
            AnalyzeUnexploredChunks(scene, report);
            
            return report;
        }
        
        private static void CalculateTileSpatialBounds(Pm4Scene scene, BandingAnalysisReport report)
        {
            if (scene.Vertices == null || scene.Links == null) return;
            
            foreach (var tileKvp in report.TileData)
            {
                var tileKey = tileKvp.Key;
                var tileData = tileKvp.Value;
                
                var tileVertices = new List<Vector3>();
                
                // Collect vertices from MSLK entries in this tile
                foreach (var link in scene.Links)
                {
                    if (link.TryDecodeTileCoordinates(out int tileX, out int tileY) &&
                        (byte)tileX == tileKey.x && (byte)tileY == tileKey.y)
                    {
                        // Get vertices from this link via MSPI indices
                        for (int i = link.MspiFirstIndex; i < link.MspiFirstIndex + link.MspiIndexCount && i < scene.Indices.Count; i++)
                        {
                            int vertexIndex = scene.Indices[i];
                            if (vertexIndex >= 0 && vertexIndex < scene.Vertices.Count)
                            {
                                tileVertices.Add(scene.Vertices[vertexIndex]);
                            }
                        }
                    }
                }
                
                if (tileVertices.Count > 0)
                {
                    var minX = tileVertices.Min(v => v.X);
                    var minY = tileVertices.Min(v => v.Y);
                    var minZ = tileVertices.Min(v => v.Z);
                    var maxX = tileVertices.Max(v => v.X);
                    var maxY = tileVertices.Max(v => v.Y);
                    var maxZ = tileVertices.Max(v => v.Z);
                    
                    tileData.SpatialBounds = (new Vector3(minX, minY, minZ), new Vector3(maxX, maxY, maxZ));
                    tileData.VertexCount = tileVertices.Count;
                }
            }
        }
        
        private static void AnalyzeUnexploredChunks(Pm4Scene scene, BandingAnalysisReport report)
        {
            // Look at chunks we might be ignoring
            if (scene.ExtraChunks != null)
            {
                foreach (var chunk in scene.ExtraChunks)
                {
                    var chunkType = chunk.GetSignature();
                    // Try to get size if the chunk supports it
                    var chunkSize = 0;
                    try
                    {
                        if (chunk is IBinarySerializable serializable)
                        {
                            chunkSize = (int)serializable.GetSize();
                        }
                    }
                    catch
                    {
                        chunkSize = 0; // Unknown size
                    }
                    
                    report.UnexploredChunks.Add(chunkType);
                    report.ChunkDataSizes[chunkType] = chunkSize;
                }
            }
            
            // Check if MPRR has unexplored data
            if (scene.Properties != null && scene.Properties.Count > 0)
            {
                report.ChunkDataSizes["MPRR_Properties"] = scene.Properties.Count;
            }
            
            // Check captured raw data
            if (scene.CapturedRawData != null)
            {
                foreach (var kvp in scene.CapturedRawData)
                {
                    report.ChunkDataSizes[$"Raw_{kvp.Key}"] = kvp.Value.Length;
                }
            }
        }
        
        /// <summary>
        /// Export detailed banding analysis report
        /// </summary>
        public static void ExportBandingReport(BandingAnalysisReport report, string outputPath)
        {
            using var writer = new StreamWriter(outputPath);
            
            writer.WriteLine("=== PM4 Data Banding Analysis Report ===");
            writer.WriteLine($"Generated: {DateTime.Now}");
            writer.WriteLine();
            
            writer.WriteLine("=== TILE DISTRIBUTION ===");
            foreach (var tileKvp in report.TileData.OrderBy(kvp => kvp.Key.x).ThenBy(kvp => kvp.Key.y))
            {
                var tile = tileKvp.Key;
                var data = tileKvp.Value;
                
                writer.WriteLine($"Tile ({tile.x}, {tile.y}):");
                writer.WriteLine($"  Links: {data.LinkCount}");
                writer.WriteLine($"  Surfaces: {data.SurfaceCount}");
                writer.WriteLine($"  Vertices: {data.VertexCount}");
                writer.WriteLine($"  Unique ParentIds: {data.ParentIds.Count}");
                writer.WriteLine($"  Unique SurfaceKeys: {data.SurfaceKeys.Count}");
                
                if (data.SpatialBounds.HasValue)
                {
                    var bounds = data.SpatialBounds.Value;
                    writer.WriteLine($"  Spatial Bounds: ({bounds.min.X:F1}, {bounds.min.Y:F1}, {bounds.min.Z:F1}) to ({bounds.max.X:F1}, {bounds.max.Y:F1}, {bounds.max.Z:F1})");
                }
                writer.WriteLine();
            }
            
            writer.WriteLine("=== CROSS-TILE OBJECT DISTRIBUTION ===");
            writer.WriteLine("ParentIds spanning multiple tiles:");
            foreach (var parentKvp in report.ParentIdToTiles.Where(kvp => kvp.Value.Count > 1))
            {
                writer.WriteLine($"  ParentId 0x{parentKvp.Key:X8}: {string.Join(", ", parentKvp.Value.Select(t => $"({t.Item1},{t.Item2})"))}");
            }
            
            writer.WriteLine();
            writer.WriteLine("SurfaceKeys spanning multiple tiles:");
            foreach (var surfaceKvp in report.SurfaceKeyToTiles.Where(kvp => kvp.Value.Count > 1))
            {
                writer.WriteLine($"  SurfaceKey 0x{surfaceKvp.Key:X8}: {string.Join(", ", surfaceKvp.Value.Select(t => $"({t.Item1},{t.Item2})"))}");
            }
            
            writer.WriteLine();
            writer.WriteLine("=== UNEXPLORED DATA ===");
            foreach (var chunk in report.UnexploredChunks.Distinct())
            {
                writer.WriteLine($"  {chunk}: {report.ChunkDataSizes.GetValueOrDefault(chunk, 0)} bytes");
            }
            
            writer.WriteLine();
            writer.WriteLine("=== HYPOTHESES TO INVESTIGATE ===");
            writer.WriteLine("1. Objects with ParentIds spanning multiple tiles may be 'banded' across tiles");
            writer.WriteLine("2. SurfaceKeys spanning multiple tiles suggest cross-tile surface linking");
            writer.WriteLine("3. Unexplored chunks may contain crucial cross-reference data");
            writer.WriteLine("4. Spatial bounds overlaps between tiles may indicate object continuity");
        }
    }
}
