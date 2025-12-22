using System.Numerics;

namespace WoWRollback.PM4Module
{
    public static class PipelineCoordinateService
    {
        private const float TileSize = 533.33333f;
        private const float HalfMapExtent = 32f * TileSize; // 17066.66656 (center of map)

        /// <summary>
        /// Converts PM4 local/tile coordinates to ADT MODF/MDDF Placement coordinates.
        /// 
        /// PM4 vertices are in tile-local space (0-533 within each tile).
        /// MODF/MDDF positions use ADT placement coords where:
        ///   ADT_X = 17066 - WorldY (west-east axis)
        ///   ADT_Y = Height (up)
        ///   ADT_Z = 17066 - WorldX (north-south axis)
        /// 
        /// This means tile (0,0) which is at world pos (17066, 17066) becomes
        /// placement (0, height, 0) after the transform.
        /// </summary>
        public static Vector3 Pm4ToAdtPosition(Vector3 pm4LocalPos, int tileX, int tileY)
        {
            // PM4/Recast uses Y-up: X=forward, Y=up, Z=right
            // WoW uses Z-up: X=north, Y=west, Z=up
            float localX = pm4LocalPos.X;  // Forward in PM4 = North in WoW
            float height = pm4LocalPos.Y;  // Up in both systems
            float localZ = pm4LocalPos.Z;  // Right in PM4 = West in WoW
            
            // Convert tile-local to world coordinates
            // Tile (0,0) is at top-left = world (17066, 17066)
            // Each tile is 533.33 units, moving south-east as tile indices increase
            float worldX = HalfMapExtent - (tileX * TileSize + localX);
            float worldY = HalfMapExtent - (tileY * TileSize + localZ);
            
            // Convert to ADT MODF/MDDF placement coordinates
            // According to wowdev.wiki: placement_X = 17066 - worldY, placement_Z = 17066 - worldX
            float placementX = HalfMapExtent - worldY;  // = tileY * TileSize + localZ
            float placementY = height;                   // Height stays the same
            float placementZ = HalfMapExtent - worldX;   // = tileX * TileSize + localX

            return new Vector3(placementX, placementY, placementZ);
        }

        /// <summary>
        /// Legacy method - converts Server/World coordinates to ADT Placement coordinates.
        /// Server coords: X=north(+), Y=west(+), Z=up
        /// Placement coords: X=17066-Y, Y=Z, Z=17066-X
        /// </summary>
        public static Vector3 ServerToAdtPosition(Vector3 serverPos)
        {
            // Server world coords are centered at (0,0) with:
            // +X = north, +Y = west, +Z = up
            // Placement coords: X = 17066 - serverY, Y = serverZ, Z = 17066 - serverX
            float placementX = HalfMapExtent - serverPos.Y;
            float placementY = serverPos.Z;  // Height
            float placementZ = HalfMapExtent - serverPos.X;

            return new Vector3(placementX, placementY, placementZ);
        }
        
        /// <summary>
        /// Get tile indices from world position.
        /// </summary>
        public static (int TileX, int TileY) WorldPosToTile(float worldX, float worldY)
        {
            int tileX = (int)((HalfMapExtent - worldX) / TileSize);
            int tileY = (int)((HalfMapExtent - worldY) / TileSize);
            return (tileX, tileY);
        }
    }
}
