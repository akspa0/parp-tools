using System;
using System.Numerics; // For Vector3
using System.Runtime.InteropServices;

namespace WoWToolbox.Navigation.PM4
{
    // Structure definition based on docs/wowdev.wiki/PD4.md#MSVT
    // Size = 12 bytes
    [StructLayout(LayoutKind.Sequential, Pack = 1, Size = 12)]
    public struct MsvtVertex
    {
        public int X; // Offset 0
        public int Y; // Offset 4
        public int Z; // Offset 8

        /// <summary>
        /// Converts PM4 MSVT vertex integer coordinates to WoW world coordinates.
        /// Applies coordinate system swap (X -> -Y, Y -> -X) and Z scaling.
        /// Based on docs/wowdev.wiki/PD4.md#MSVT transformation formulas.
        /// </summary>
        /// <returns>Vector3 world coordinates.</returns>
        public Vector3 ToWorldCoordinates()
        {
            // Apply transformation based on documentation directly to integer fields
            // X_wow = Offset_Y - Y_pm4_int
            // Y_wow = Offset_X - X_pm4_int
            // Z_wow = Z_pm4_int / 36.0f
            float worldX = Constants.TileSize * 1.5f - Y; 
            float worldY = Constants.TileSize * 1.5f - X;
            float worldZ = (float)Z / 36.0f; // Cast Z to float BEFORE dividing

            return new Vector3(worldX, worldY, worldZ);
        }
    }
} 