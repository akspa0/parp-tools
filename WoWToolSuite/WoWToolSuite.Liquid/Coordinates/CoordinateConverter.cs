using System;

namespace WowToolSuite.Liquid.Coordinates
{
    public static class CoordinateConverter
    {
        // World of Warcraft uses a tile size of 533.33 yards per ADT
        public const float TILE_SIZE = 533.33f;
        
        // The world is 64x64 tiles (ADTs)
        public const int WORLD_SIZE = 64;
        
        // ADT at (32,32) is the center of the world (0,0 in world coordinates)
        public const int WORLD_CENTER_OFFSET = 32;
        
        // Maximum world coordinate at corner
        public const float WORLD_MAX_COORD = WORLD_CENTER_OFFSET * TILE_SIZE;
        
        /// <summary>
        /// Converts world coordinates to ADT coordinates
        /// </summary>
        /// <param name="worldX">X coordinate in world space</param>
        /// <param name="worldY">Y coordinate in world space</param>
        /// <returns>ADT coordinates as (x, y)</returns>
        public static (int X, int Y) WorldToAdtCoordinates(float worldX, float worldY)
        {
            // FIXED: WoW's coordinate system has X and Y swapped compared to our implementation
            // Convert world coordinates to ADT indices
            // ADT coordinates are inverse to world coordinates: 
            // As world coord increases, ADT index decreases from the max value (63)
            
            // Swap X and Y to match WoW's coordinate system
            int adtY = WORLD_CENTER_OFFSET - (int)Math.Ceiling(worldX / TILE_SIZE);
            int adtX = WORLD_CENTER_OFFSET - (int)Math.Ceiling(worldY / TILE_SIZE);
            
            // Ensure ADT coordinates are within valid range (0-63)
            adtX = Math.Clamp(adtX, 0, WORLD_SIZE - 1);
            adtY = Math.Clamp(adtY, 0, WORLD_SIZE - 1);
            
            return (adtX, adtY);
        }
        
        /// <summary>
        /// Converts ADT coordinates to world coordinates (top left corner of the ADT)
        /// </summary>
        /// <param name="adtX">ADT X coordinate</param>
        /// <param name="adtY">ADT Y coordinate</param>
        /// <returns>World coordinates as (x, y) of the top left corner</returns>
        public static (float X, float Y) AdtToWorldCoordinates(int adtX, int adtY)
        {
            // FIXED: WoW's coordinate system has X and Y swapped compared to our implementation
            // Convert ADT indices to world coordinates
            // Higher ADT coordinates = lower world coordinates
            
            // Swap X and Y to match WoW's coordinate system
            float worldY = WORLD_MAX_COORD - (adtX * TILE_SIZE);
            float worldX = WORLD_MAX_COORD - (adtY * TILE_SIZE);
            
            return (worldX, worldY);
        }
        
        /// <summary>
        /// Debug method to test coordinate conversions
        /// </summary>
        public static void DebugCoordinates()
        {
            Console.WriteLine("ADT Coordinate Mapping Examples:");
            Console.WriteLine("--------------------------------");
            
            // Test corner ADT (0,0)
            var worldCorner = AdtToWorldCoordinates(0, 0);
            Console.WriteLine($"ADT (0,0) top left: ({worldCorner.X:F2}, {worldCorner.Y:F2})");
            
            // Test center ADT (32,32)
            var worldCenter = AdtToWorldCoordinates(32, 32);
            Console.WriteLine($"ADT (32,32) top left: ({worldCenter.X:F2}, {worldCenter.Y:F2})");
            
            // Test the examples mentioned
            var test1 = AdtToWorldCoordinates(32, 0);
            Console.WriteLine($"ADT (32,0) top left: ({test1.X:F2}, {test1.Y:F2})");
            
            var test2 = AdtToWorldCoordinates(32, 1);
            Console.WriteLine($"ADT (32,1) top left: ({test2.X:F2}, {test2.Y:F2})");
            
            // Test development_19_27.adt
            var testSpecific = AdtToWorldCoordinates(19, 27);
            Console.WriteLine($"ADT (19,27) top left: ({testSpecific.X:F2}, {testSpecific.Y:F2})");
            
            // Round-trip test for the specific coordinates mentioned by the user
            var userCoords1 = (2666.0f, 6933.0f); // Top left of development_19_27.adt according to user
            var adtFromUser1 = WorldToAdtCoordinates(userCoords1.Item1, userCoords1.Item2);
            Console.WriteLine($"User coords ({userCoords1.Item1:F2}, {userCoords1.Item2:F2}) → ADT ({adtFromUser1.X}, {adtFromUser1.Y})");
            
            var userCoords2 = (2133.0f, 6100.0f); // Bottom right of development_19_27.adt according to user
            var adtFromUser2 = WorldToAdtCoordinates(userCoords2.Item1, userCoords2.Item2);
            Console.WriteLine($"User coords ({userCoords2.Item1:F2}, {userCoords2.Item2:F2}) → ADT ({adtFromUser2.X}, {adtFromUser2.Y})");
        }
    }
} 