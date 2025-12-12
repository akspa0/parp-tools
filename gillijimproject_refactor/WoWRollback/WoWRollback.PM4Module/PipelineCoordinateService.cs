using System.Numerics;

namespace WoWRollback.PM4Module
{
    public static class PipelineCoordinateService
    {
        private const float TileSize = 533.33333f;
        private const float HalfMapExtent = 32f * TileSize; // 17066.66656 (center of map)

        /// <summary>
        /// Converts PM4 local/tile coordinates to ADT Placement coordinates.
        /// PM4 vertices are relative to the tile's coordinate space.
        /// We rebase them to global ADT placement coords using the tile indices.
        /// </summary>
        /// <param name="pm4LocalPos">The PM4 vertex in local/tile coordinates (may need axis swap)</param>
        /// <param name="tileX">The tile X index from the PM4 filename</param>
        /// <param name="tileY">The tile Y index from the PM4 filename</param>
        public static Vector3 Pm4ToAdtPosition(Vector3 pm4LocalPos, int tileX, int tileY)
        {
            // PM4/Recast uses Y-up coordinate system, WoW uses Z-up
            // Swap: Input.Y is height, Input.Z is horizontal
            float localX = pm4LocalPos.X;
            float localY = pm4LocalPos.Z;  // Horizontal axis (swapped from Y-up Z)
            float height = pm4LocalPos.Y;  // Height axis (swapped from Y-up Y)
            
            // Calculate tile origin in ADT placement coords
            // ADT placement: tile (0,0) starts at placement coord (0,0)
            // Each tile spans TileSize (533.33) units
            float tileOriginX = tileX * TileSize;
            float tileOriginY = tileY * TileSize;
            
            // PM4 local coords are typically in range 0..533 within the tile
            // Add tile origin to get global placement coords
            float placementX = tileOriginX + localX;
            float placementY = height;
            float placementZ = tileOriginY + localY;

            // Handle coordinate wrapping for values exceeding map boundaries
            // Full map extent is 64 * 533.33 = 34133.33
            const float FullMapExtent = 64f * TileSize;
            
            // If coordinates exceed map bounds, they may be encoded as wrapping values
            // Subtract full map extent to bring them back into valid range
            if (placementX > FullMapExtent)
                placementX -= FullMapExtent;
            if (placementZ > FullMapExtent)
                placementZ -= FullMapExtent;
            
            // Also handle potential negative overflow (values that wrapped the other way)
            if (placementX < 0)
                placementX += FullMapExtent;
            if (placementZ < 0)
                placementZ += FullMapExtent;

            return new Vector3(placementX, placementY, placementZ);
        }

        /// <summary>
        /// Legacy method - converts Server/World coordinates to ADT Placement coordinates.
        /// Use Pm4ToAdtPosition for PM4 data instead.
        /// </summary>
        public static Vector3 ServerToAdtPosition(Vector3 serverPos)
        {
            // This is for world coords centered at (0,0), not PM4 tile-local coords
            float placementX = HalfMapExtent - serverPos.Y;
            float placementY = serverPos.Z;  // Height
            float placementZ = HalfMapExtent - serverPos.X;

            return new Vector3(placementX, placementY, placementZ);
        }
    }
}
