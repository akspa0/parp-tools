using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using Warcraft.NET.Files.Interfaces;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;
using System.Numerics;

namespace WCAnalyzer.Core.Utilities
{
    /// <summary>
    /// Provides wrapper methods for accessing chunk data in a backwards-compatible way.
    /// This class maintains compatibility with code that expects properties and methods that may have changed.
    /// </summary>
    public static class ChunkParsingWrapper
    {
        #region MPRLChunk.ServerPositionData wrappers
        
        /// <summary>
        /// Gets the X coordinate from a ServerPositionData entry.
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <param name="logger">Optional logger for logging detailed information.</param>
        /// <returns>The X coordinate.</returns>
        public static float GetCoordinateX(MPRLChunk.ServerPositionData data, ILogger? logger = null)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
                
            // Return the X position directly
            return data.PositionX;
        }
        
        /// <summary>
        /// Gets the Y coordinate from a ServerPositionData entry.
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <param name="logger">Optional logger for logging detailed information.</param>
        /// <returns>The Y coordinate.</returns>
        public static float GetCoordinateY(MPRLChunk.ServerPositionData data, ILogger? logger = null)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
                
            // Return the Y position directly
            return data.PositionY;
        }
        
        /// <summary>
        /// Gets the Z coordinate from a ServerPositionData entry.
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <param name="logger">Optional logger for logging detailed information.</param>
        /// <returns>The Z coordinate.</returns>
        public static float GetCoordinateZ(MPRLChunk.ServerPositionData data, ILogger? logger = null)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
                
            // Return the Z position directly
            return data.PositionZ;
        }
        
        /// <summary>
        /// Gets Value1 from a ServerPositionData entry.
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <returns>Value1.</returns>
        public static int GetValue1(MPRLChunk.ServerPositionData data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
                
            return data.Value0x14;
        }
        
        /// <summary>
        /// Gets Value2 from a ServerPositionData entry.
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <returns>Value2.</returns>
        public static int GetValue2(MPRLChunk.ServerPositionData data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
                
            // Use Value0x06 instead of Value0x18 which doesn't exist
            return data.Value0x06;
        }
        
        /// <summary>
        /// Gets Value3 from a ServerPositionData entry.
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <returns>Value3.</returns>
        public static int GetValue3(MPRLChunk.ServerPositionData data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
                
            // Use Value0x16 instead of Value0x1c which doesn't exist
            return data.Value0x16;
        }
        
        /// <summary>
        /// Safely gets the SpecialValue from a ServerPositionData entry
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <param name="logger">Optional logger.</param>
        /// <returns>The special value.</returns>
        [Obsolete("Use direct property access (entry.Value0x04) instead. This method will be removed in a future version.")]
        public static int GetSpecialValue(MPRLChunk.ServerPositionData data, ILogger? logger = null)
        {
            try
            {
                // Logic to get the special value
                // This would be based on your specific implementation
                return data?.Value0x04 ?? 0;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error accessing special value from ServerPositionData");
                return 0;
            }
        }
        
        /// <summary>
        /// Checks if this is a control record
        /// </summary>
        public static bool IsControlRecord(MPRLChunk.ServerPositionData data, ILogger? logger = null)
        {
            try
            {
                // Logic to determine if this is a control record
                // This is a placeholder for your specific implementation
                return data != null && data.Value0x00 == 0 && data.Value0x02 == -1;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error determining if ServerPositionData is a control record");
                return false;
            }
        }
        
        #endregion
        
        #region MSVTChunk.VertexData wrappers
        
        /// <summary>
        /// Safely gets the X value from a VertexData entry
        /// </summary>
        public static float GetX(MSVTChunk.VertexData data, ILogger? logger = null)
        {
            try
            {
                return data?.WorldX ?? 0.0f;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error accessing X property from VertexData");
                return 0.0f;
            }
        }
        
        /// <summary>
        /// Safely gets the Y value from a VertexData entry
        /// </summary>
        public static float GetY(MSVTChunk.VertexData data, ILogger? logger = null)
        {
            try
            {
                return data?.WorldY ?? 0.0f;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error accessing Y property from VertexData");
                return 0.0f;
            }
        }
        
        /// <summary>
        /// Safely gets the Z value from a VertexData entry
        /// </summary>
        public static float GetZ(MSVTChunk.VertexData data, ILogger? logger = null)
        {
            try
            {
                return data?.WorldZ ?? 0.0f;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error accessing Z property from VertexData");
                return 0.0f;
            }
        }
        
        /// <summary>
        /// Gets the Flag1 value (placeholder - add your actual logic)
        /// </summary>
        public static int GetFlag1(MSVTChunk.VertexData data, ILogger? logger = null)
        {
            try
            {
                // Replace with your actual implementation
                // This is a placeholder since we don't see Flag1 in the provided class
                return 0;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error accessing Flag1 property from VertexData");
                return 0;
            }
        }
        
        /// <summary>
        /// Gets the Flag2 value (placeholder - add your actual logic)
        /// </summary>
        public static int GetFlag2(MSVTChunk.VertexData data, ILogger? logger = null)
        {
            try
            {
                // Replace with your actual implementation
                // This is a placeholder since we don't see Flag2 in the provided class
                return 0;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error accessing Flag2 property from VertexData");
                return 0;
            }
        }
        
        #endregion
        
        #region Simplified overloads for PM4TerrainExporter
        
        /// <summary>
        /// Simplified overload for GetSpecialValue that doesn't require a logger
        /// </summary>
        [Obsolete("Use direct property access (entry.Value0x04) instead. This method will be removed in a future version.")]
        public static int GetSpecialValue(MPRLChunk.ServerPositionData entry)
        {
            return GetSpecialValue(entry, null);
        }
        
        /// <summary>
        /// Gets the position as a Vector3
        /// </summary>
        public static Vector3 GetPosition(MPRLChunk.ServerPositionData entry, ILogger? logger = null)
        {
            return new Vector3(
                GetCoordinateX(entry, logger),
                GetCoordinateY(entry, logger),
                GetCoordinateZ(entry, logger)
            );
        }
        
        /// <summary>
        /// Gets the world coordinates from a ServerPositionData entry.
        /// </summary>
        /// <param name="data">The ServerPositionData.</param>
        /// <returns>The world coordinates.</returns>
        public static Vector3 GetWorldCoordinates(MPRLChunk.ServerPositionData data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
                
            // Convert file coordinates to world coordinates
            return ChunkParsingUtility.TransformToWorldCoordinates(
                GetCoordinateX(data),
                GetCoordinateY(data),
                GetCoordinateZ(data)
            );
        }
        
        #endregion

        #region Extension Methods

        /// <summary>
        /// Extension method to get CoordinateX from ServerPositionData
        /// </summary>
        public static float CoordinateX(this MPRLChunk.ServerPositionData data)
        {
            return data.PositionX;
        }

        /// <summary>
        /// Extension method to get CoordinateY from ServerPositionData
        /// </summary>
        public static float CoordinateY(this MPRLChunk.ServerPositionData data)
        {
            return data.PositionY;
        }

        /// <summary>
        /// Extension method to get CoordinateZ from ServerPositionData
        /// </summary>
        public static float CoordinateZ(this MPRLChunk.ServerPositionData data)
        {
            return data.PositionZ;
        }

        /// <summary>
        /// Extension method to check if a ServerPositionData entry is a control record
        /// </summary>
        public static bool IsControlRecord(this MPRLChunk.ServerPositionData data)
        {
            return data.Value0x00 == 0 && data.Value0x02 == -1;
        }

        /// <summary>
        /// Extension method to get the special value from a ServerPositionData entry
        /// </summary>
        public static int SpecialValue(this MPRLChunk.ServerPositionData data)
        {
            return data.Value0x04;
        }

        /// <summary>
        /// Extension method to get Value1 from a ServerPositionData entry
        /// </summary>
        public static int Value1(this MPRLChunk.ServerPositionData data)
        {
            return data.Value0x14;
        }

        /// <summary>
        /// Extension method to get Value2 from a ServerPositionData entry
        /// </summary>
        public static int Value2(this MPRLChunk.ServerPositionData data)
        {
            return data.Value0x06;
        }

        /// <summary>
        /// Extension method to get Value3 from a ServerPositionData entry
        /// </summary>
        public static int Value3(this MPRLChunk.ServerPositionData data)
        {
            return data.Value0x16;
        }

        /// <summary>
        /// Extension method to check if a ServerPositionData entry is a special entry
        /// </summary>
        public static bool IsSpecialEntry(this MPRLChunk.ServerPositionData data)
        {
            return data.Value0x02 == -1;
        }

        #endregion

        #region MSVTChunk.VertexData Extension Methods

        /// <summary>
        /// Extension method to get X from VertexData
        /// </summary>
        public static float X(this MSVTChunk.VertexData data)
        {
            return data.WorldX;
        }

        /// <summary>
        /// Extension method to get Y from VertexData
        /// </summary>
        public static float Y(this MSVTChunk.VertexData data)
        {
            return data.WorldY;
        }

        /// <summary>
        /// Extension method to get Z from VertexData
        /// </summary>
        public static float Z(this MSVTChunk.VertexData data)
        {
            return data.WorldZ;
        }

        /// <summary>
        /// Extension method to get Flag1 from VertexData (placeholder)
        /// </summary>
        public static int Flag1(this MSVTChunk.VertexData data)
        {
            // This is a placeholder since the actual property doesn't exist
            return 0;
        }

        /// <summary>
        /// Extension method to get Flag2 from VertexData (placeholder)
        /// </summary>
        public static int Flag2(this MSVTChunk.VertexData data)
        {
            // This is a placeholder since the actual property doesn't exist
            return 0;
        }

        #endregion

        #region PD4 Chunk Extension Methods

        /// <summary>
        /// Extension method to get Crc from MCRCChunk
        /// </summary>
        public static uint Crc(this Models.PD4.Chunks.MCRCChunk chunk)
        {
            return chunk.CRCValue;
        }

        /// <summary>
        /// Extension method to get Flags from MSHDChunk
        /// </summary>
        public static uint Flags(this Models.PD4.Chunks.MSHDChunk chunk)
        {
            // This is a placeholder since the actual property doesn't exist
            return 0;
        }

        /// <summary>
        /// Extension method to get Value0x04 from MSHDChunk
        /// </summary>
        public static uint Value0x04(this Models.PD4.Chunks.MSHDChunk chunk)
        {
            // This is a placeholder since the actual property doesn't exist
            return chunk.Width;
        }

        /// <summary>
        /// Extension method to get Value0x08 from MSHDChunk
        /// </summary>
        public static uint Value0x08(this Models.PD4.Chunks.MSHDChunk chunk)
        {
            // This is a placeholder since the actual property doesn't exist
            return chunk.Height;
        }

        /// <summary>
        /// Extension method to get Value0x0c from MSHDChunk
        /// </summary>
        public static uint Value0x0c(this Models.PD4.Chunks.MSHDChunk chunk)
        {
            // This is a placeholder since the actual property doesn't exist
            return 0;
        }

        /// <summary>
        /// Extension method to get Positions from MSPVChunk
        /// </summary>
        public static System.Collections.Generic.List<System.Numerics.Vector3> Positions(this Models.PD4.Chunks.MSPVChunk chunk)
        {
            return chunk.Vertices;
        }

        /// <summary>
        /// Extension method to get Indices from MSPIChunk
        /// </summary>
        public static System.Collections.Generic.List<uint> Indices(this Models.PD4.Chunks.MSPIChunk chunk)
        {
            return chunk.Indices;
        }

        /// <summary>
        /// Extension method to get Indices from MSVIChunk
        /// </summary>
        public static System.Collections.Generic.List<Models.PD4.Chunks.MSVIChunk.VertexInfo> Indices(this Models.PD4.Chunks.MSVIChunk chunk)
        {
            return chunk.Entries;
        }

        #endregion

        #region MPRRChunk.PositionReference Extension Methods

        /// <summary>
        /// Extension method to get Value1 from PositionReference
        /// </summary>
        public static int Value1(this Models.PM4.Chunks.MPRRChunk.PositionReference data)
        {
            return data.Value0x00;
        }

        /// <summary>
        /// Extension method to get Value2 from PositionReference
        /// </summary>
        public static int Value2(this Models.PM4.Chunks.MPRRChunk.PositionReference data)
        {
            return data.Value0x02;
        }

        #endregion

        #region PM4Chunk Extension Methods

        /// <summary>
        /// Extension method to get Size from PM4Chunk
        /// </summary>
        public static uint Size(this Models.PM4.Chunks.PM4Chunk chunk)
        {
            return chunk.Size;
        }

        #endregion

        #region MVER Extension Methods

        /// <summary>
        /// Extension method to get Size from MVER
        /// </summary>
        public static uint Size(this Warcraft.NET.Files.ADT.Chunks.MVER chunk)
        {
            return 4; // MVER is typically 4 bytes (just contains a version number)
        }

        #endregion
    }
} 