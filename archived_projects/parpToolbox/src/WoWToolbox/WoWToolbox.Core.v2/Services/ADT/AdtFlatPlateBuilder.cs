using System;
using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Services.ADT
{
    /// <summary>
    /// Utility to generate a flat ADT-style terrain plate.  Each 533.3333-unit tile is subdivided
    /// into a 16×16 chunk grid and each chunk into an 8×8 square grid, yielding a 129×129 vertex
    /// grid for the tile (standard ADT height-map resolution).  Faces are written as two triangles
    /// per square.
    /// </summary>
    public static class AdtFlatPlateBuilder
    {
        public const float TileSize = 533.3333f;
        private const int SquaresPerTileAxis = 128;          // 16 chunks * 8 squares
        private const int VerticesPerTileAxis = SquaresPerTileAxis + 1; // 129
        private const float QuadSize = TileSize; // For simplified tile
        private const float SquareSize = TileSize / SquaresPerTileAxis; // 4.16667f
        private const int MapTiles = 64; // full ADT grid per map

        // Converts ADT tile indices (0..63,0..63) to WoW world-space origin (lower-left of tile)
        private static (float x, float y) TileOrigin(int tileX, int tileY)
        {
            // WoW zero-point is bottom-left; our original plate builder assumed top-left.
            // Flip both axes so that (0,0) in ADT (north-west) ends up at top-left in OBJ viewer.
            float x = (MapTiles - 1 - tileX) * TileSize;
            float y = (MapTiles - 1 - tileY) * TileSize;
            return (x, y);
        }

        /// <summary>
        /// Generates vertices and triangle indices for a single flat ADT tile at (tileX,tileY).
        /// The origin (0,0) is the top-left of the map, +X east, +Y south.
        /// </summary>
        public static void BuildTile(int tileX, int tileY, List<Vector3> verts, List<(int,int,int)> tris)
        {
            int vertStart = verts.Count; // index offset for this tile
            var origin = TileOrigin(tileX, tileY);
            float baseX = origin.x;
                        float baseY = origin.y;

            // 1. vertices
            for (int row = 0; row < VerticesPerTileAxis; row++)
            {
                for (int col = 0; col < VerticesPerTileAxis; col++)
                {
                    float x = baseX + col * SquareSize;
                    float y = baseY + row * SquareSize;
                    verts.Add(new Vector3(x, y, 0f));
                }
            }

            // 2. faces (two tris per square)
            for (int row = 0; row < SquaresPerTileAxis; row++)
            {
                for (int col = 0; col < SquaresPerTileAxis; col++)
                {
                    int topLeft = vertStart + row * VerticesPerTileAxis + col;
                    int topRight = topLeft + 1;
                    int bottomLeft = topLeft + VerticesPerTileAxis;
                    int bottomRight = bottomLeft + 1;

                    tris.Add((topLeft, bottomLeft, bottomRight));
                    tris.Add((topLeft, bottomRight, topRight));
                }
            }
        }
        /// <summary>
        /// Generates a single-quad (two-triangle) flat tile for coarse representation.
        /// </summary>
        public static void BuildSimpleTile(int tileX, int tileY, List<Vector3> verts, List<(int,int,int)> tris)
        {
            int vStart = verts.Count;
            var origin = TileOrigin(tileX, tileY);
            float baseX = origin.x;
                        float baseY = origin.y;
            verts.Add(new Vector3(baseX,               baseY,               0f)); // 0
            verts.Add(new Vector3(baseX + QuadSize,    baseY,               0f)); // 1
            verts.Add(new Vector3(baseX + QuadSize,    baseY + QuadSize,    0f)); // 2
            verts.Add(new Vector3(baseX,               baseY + QuadSize,    0f)); // 3
            tris.Add((vStart,     vStart+2, vStart+1));
            tris.Add((vStart,     vStart+3, vStart+2));
        }
    }
}
