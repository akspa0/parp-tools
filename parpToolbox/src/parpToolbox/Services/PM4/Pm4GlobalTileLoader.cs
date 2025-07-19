using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Loads PM4/PD4 files as a unified global tile system, handling missing tiles gracefully.
    /// This solves the fundamental issue where vertex indices reference data from adjacent tiles.
    /// </summary>
    public class Pm4GlobalTileLoader
    {
        public const int TILE_GRID_SIZE = 64; // 64x64 grid = 4096 possible tiles
        public const int MAX_TILES = TILE_GRID_SIZE * TILE_GRID_SIZE;
        
        /// <summary>
        /// Represents a tile coordinate in the global grid.
        /// </summary>
        public record TileCoordinate(int X, int Y)
        {
            public static TileCoordinate FromFileName(string fileName)
            {
                // Parse tile coordinates from filename like "Stormwind_37_49.pm4"
                var parts = Path.GetFileNameWithoutExtension(fileName).Split('_');
                if (parts.Length >= 3 && int.TryParse(parts[^2], out int x) && int.TryParse(parts[^1], out int y))
                {
                    return new TileCoordinate(x, y);
                }
                throw new ArgumentException($"Cannot parse tile coordinates from filename: {fileName}");
            }
            
            public int ToLinearIndex() => Y * TILE_GRID_SIZE + X;
            public static TileCoordinate FromLinearIndex(int index) => new(index % TILE_GRID_SIZE, index / TILE_GRID_SIZE);
        }
        
        /// <summary>
        /// Represents a loaded tile with its coordinate and scene data.
        /// </summary>
        public record LoadedTile(TileCoordinate Coordinate, Pm4Scene Scene, string SourceFile);
        
        /// <summary>
        /// Represents the unified global scene combining all tiles.
        /// </summary>
        public class GlobalScene
        {
            public List<Vector3> GlobalVertices { get; } = new();
            public List<int> GlobalIndices { get; } = new();
            public List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry> GlobalSurfaces { get; } = new();
            public List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry> GlobalLinks { get; } = new();
            public Dictionary<TileCoordinate, LoadedTile> LoadedTiles { get; } = new();
            public Dictionary<TileCoordinate, int> TileVertexOffsets { get; } = new();
            public Dictionary<TileCoordinate, int> TileIndexOffsets { get; } = new();
            public Dictionary<TileCoordinate, int> TileSurfaceOffsets { get; } = new();
            
            public int TotalLoadedTiles => LoadedTiles.Count;
            public int TotalMissingTiles => MAX_TILES - TotalLoadedTiles;
        }
        
        /// <summary>
        /// Loads all PM4/PD4 files in a directory as a unified global scene.
        /// </summary>
        public static GlobalScene LoadRegion(string directoryPath, string filePattern = "*.pm4")
        {
            ConsoleLogger.WriteLine($"Loading PM4/PD4 region from: {directoryPath}");
            ConsoleLogger.WriteLine($"File pattern: {filePattern}");
            
            var globalScene = new GlobalScene();
            var pm4Adapter = new Pm4Adapter();
            
            // Find all PM4/PD4 files in the directory
            var files = Directory.GetFiles(directoryPath, filePattern, SearchOption.TopDirectoryOnly);
            ConsoleLogger.WriteLine($"Found {files.Length} files matching pattern");
            
            // Parse tile coordinates and load files
            var loadedTiles = new List<LoadedTile>();
            foreach (var file in files)
            {
                try
                {
                    var coordinate = TileCoordinate.FromFileName(Path.GetFileName(file));
                    var scene = pm4Adapter.Load(file);
                    var tile = new LoadedTile(coordinate, scene, file);
                    loadedTiles.Add(tile);
                    globalScene.LoadedTiles[coordinate] = tile;
                    
                    ConsoleLogger.WriteLine($"  Loaded tile ({coordinate.X}, {coordinate.Y}): {scene.Vertices.Count} vertices, {scene.Surfaces.Count} surfaces");
                }
                catch (Exception ex)
                {
                    ConsoleLogger.WriteLine($"  Failed to load {Path.GetFileName(file)}: {ex.Message}");
                }
            }
            
            ConsoleLogger.WriteLine($"Successfully loaded {loadedTiles.Count} tiles out of {MAX_TILES} possible tiles");
            ConsoleLogger.WriteLine($"Missing tiles: {globalScene.TotalMissingTiles}");
            
            // Build unified global scene
            BuildGlobalScene(globalScene, loadedTiles);
            
            return globalScene;
        }
        
        /// <summary>
        /// Builds the unified global scene by combining all loaded tiles.
        /// </summary>
        private static void BuildGlobalScene(GlobalScene globalScene, List<LoadedTile> loadedTiles)
        {
            ConsoleLogger.WriteLine("Building unified global scene...");
            
            int currentVertexOffset = 0;
            int currentIndexOffset = 0;
            int currentSurfaceOffset = 0;
            
            // Sort tiles by coordinate for consistent processing
            var sortedTiles = loadedTiles.OrderBy(t => t.Coordinate.Y).ThenBy(t => t.Coordinate.X).ToList();
            
            foreach (var tile in sortedTiles)
            {
                // Record offsets for this tile
                globalScene.TileVertexOffsets[tile.Coordinate] = currentVertexOffset;
                globalScene.TileIndexOffsets[tile.Coordinate] = currentIndexOffset;
                globalScene.TileSurfaceOffsets[tile.Coordinate] = currentSurfaceOffset;
                
                // Add vertices (with global coordinate adjustment if needed)
                foreach (var vertex in tile.Scene.Vertices)
                {
                    // Vertices are already in global coordinates, just add them
                    globalScene.GlobalVertices.Add(vertex);
                }
                
                // Add indices (adjusted for global vertex offset)
                foreach (var index in tile.Scene.Indices)
                {
                    globalScene.GlobalIndices.Add(index + currentVertexOffset);
                }
                
                // Add surfaces (adjusted for global index offset)
                foreach (var surface in tile.Scene.Surfaces)
                {
                    // For now, add the original surface - proper adjustment will be implemented later
                    // TODO: Adjust MsviFirstIndex to account for global index offset
                    globalScene.GlobalSurfaces.Add(surface);
                }
                
                // Add links (may need adjustment for global references)
                globalScene.GlobalLinks.AddRange(tile.Scene.Links);
                
                // Update offsets for next tile
                currentVertexOffset += tile.Scene.Vertices.Count;
                currentIndexOffset += tile.Scene.Indices.Count;
                currentSurfaceOffset += tile.Scene.Surfaces.Count;
                
                ConsoleLogger.WriteLine($"  Processed tile ({tile.Coordinate.X}, {tile.Coordinate.Y}): " +
                                      $"vertices {globalScene.TileVertexOffsets[tile.Coordinate]}-{currentVertexOffset-1}, " +
                                      $"indices {globalScene.TileIndexOffsets[tile.Coordinate]}-{currentIndexOffset-1}");
            }
            
            ConsoleLogger.WriteLine($"Global scene built:");
            ConsoleLogger.WriteLine($"  Total vertices: {globalScene.GlobalVertices.Count}");
            ConsoleLogger.WriteLine($"  Total indices: {globalScene.GlobalIndices.Count}");
            ConsoleLogger.WriteLine($"  Total surfaces: {globalScene.GlobalSurfaces.Count}");
            ConsoleLogger.WriteLine($"  Total links: {globalScene.GlobalLinks.Count}");
        }
        
        /// <summary>
        /// Converts a GlobalScene to a standard Pm4Scene for compatibility with existing exporters.
        /// </summary>
        public static Pm4Scene ToStandardScene(GlobalScene globalScene)
        {
            return new Pm4Scene
            {
                Vertices = globalScene.GlobalVertices,
                Triangles = new List<(int A, int B, int C)>(), // Will be computed from indices
                Surfaces = globalScene.GlobalSurfaces,
                Links = globalScene.GlobalLinks,
                Indices = globalScene.GlobalIndices,
                Placements = new List<ParpToolbox.Formats.P4.Chunks.Common.MprlChunk.Entry>(), // TODO: Combine from tiles
                Properties = new List<ParpToolbox.Formats.P4.Chunks.Common.MprrChunk.Entry>(), // TODO: Combine from tiles
                Groups = new List<SurfaceGroup>() // TODO: Implement proper grouping
            };
        }
    }
}
