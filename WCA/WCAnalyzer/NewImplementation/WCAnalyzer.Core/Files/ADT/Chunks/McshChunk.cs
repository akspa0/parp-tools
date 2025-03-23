using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Text;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents a MCSH chunk in an ADT file, containing shadow map data for terrain.
    /// </summary>
    public class McshChunk : ADTChunk
    {
        /// <summary>
        /// The MCSH chunk signature
        /// </summary>
        public const string SIGNATURE = "MCSH";
        
        /// <summary>
        /// Gets the raw shadow map data
        /// </summary>
        public byte[] RawData { get; private set; } = Array.Empty<byte>();
        
        /// <summary>
        /// Gets the size of the shadow map data
        /// </summary>
        public int DataSize => RawData.Length;

        /// <summary>
        /// Initializes a new instance of the <see cref="McshChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public McshChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }

        /// <summary>
        /// Parses the chunk data
        /// </summary>
        public override void Parse()
        {
            if (Data == null || Data.Length == 0)
            {
                AddError("No data to parse for MCSH chunk");
                return;
            }
            
            try
            {
                // Store the raw data for later access and analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"MCSH: Read {Data.Length} bytes of shadow map data");
                
                // Log additional information about the shadow map structure if debug logging is enabled
                if (Logger?.IsEnabled(LogLevel.Debug) == true)
                {
                    LogDataSummary();
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MCSH chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Logs a summary of the shadow map data for debugging purposes
        /// </summary>
        private void LogDataSummary()
        {
            if (RawData.Length == 0)
                return;
                
            try
            {
                // Log data size
                Logger?.LogDebug($"MCSH: Data size: {RawData.Length} bytes");
                
                // Check if the data size matches expected shadow map dimensions
                // Shadow maps are typically grid-based
                int mapSize = (int)Math.Sqrt(RawData.Length);
                if (mapSize * mapSize == RawData.Length)
                {
                    Logger?.LogDebug($"MCSH: Data appears to be a square shadow map of {mapSize}x{mapSize}");
                }
                
                // Check if data might represent an array of common data structures
                if (RawData.Length % 4 == 0)
                    Logger?.LogDebug($"MCSH: Data size is divisible by 4 ({RawData.Length / 4} elements if 4-byte values)");
                if (RawData.Length % 8 == 0)
                    Logger?.LogDebug($"MCSH: Data size is divisible by 8 ({RawData.Length / 8} elements if 8-byte values)");
                
                // Log the first few bytes to check if they might be intensity values
                if (RawData.Length >= 16)
                {
                    StringBuilder sb = new StringBuilder("MCSH: First 16 bytes (possible shadow intensity values): ");
                    for (int i = 0; i < 16; i++)
                    {
                        sb.Append(RawData[i].ToString("X2"));
                        sb.Append(' ');
                    }
                    Logger?.LogDebug(sb.ToString());
                }
                
                // Check for byte patterns
                if (RawData.Length >= 16)
                {
                    bool allZeros = true;
                    bool allOnes = true;
                    for (int i = 0; i < 16; i++)
                    {
                        if (RawData[i] != 0) allZeros = false;
                        if (RawData[i] != 255) allOnes = false;
                    }
                    
                    if (allZeros)
                        Logger?.LogDebug("MCSH: First 16 bytes are all zeros (possibly no shadow)");
                    if (allOnes)
                        Logger?.LogDebug("MCSH: First 16 bytes are all 255 (possibly full shadow)");
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MCSH data summary: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Writes the chunk data to the specified writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                AddError("Cannot write to null writer");
                return;
            }
            
            try
            {
                // Write the raw shadow map data
                writer.Write(RawData);
                
                Logger?.LogDebug($"MCSH: Wrote {RawData.Length} bytes of shadow map data");
            }
            catch (Exception ex)
            {
                AddError($"Error writing MCSH chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Gets the raw data at the specified offset
        /// </summary>
        /// <param name="offset">The offset into the data</param>
        /// <param name="count">The number of bytes to read</param>
        /// <returns>The raw data or an empty array if parameters are invalid</returns>
        public byte[] GetRawData(int offset, int count)
        {
            if (offset < 0 || offset >= RawData.Length)
            {
                AddError($"Offset must be between 0 and {RawData.Length - 1}, got {offset}");
                return Array.Empty<byte>();
            }
            
            if (count < 0 || offset + count > RawData.Length)
            {
                AddError($"Count must be between 0 and {RawData.Length - offset}, got {count}");
                return Array.Empty<byte>();
            }
            
            try
            {
                byte[] result = new byte[count];
                Array.Copy(RawData, offset, result, 0, count);
                return result;
            }
            catch (Exception ex)
            {
                AddError($"Error getting raw data: {ex.Message}");
                return Array.Empty<byte>();
            }
        }
        
        /// <summary>
        /// Attempts to interpret the data as a square shadow map
        /// </summary>
        /// <returns>The shadow intensity values as a jagged array, or null if not a perfect square</returns>
        public byte[][]? TryGetAsShadowMap()
        {
            int mapSize = (int)Math.Sqrt(RawData.Length);
            if (mapSize * mapSize != RawData.Length)
            {
                Logger?.LogDebug($"MCSH: Data size {RawData.Length} is not a perfect square, cannot interpret as shadow map");
                return null;
            }
                
            try
            {
                byte[][] map = new byte[mapSize][];
                for (int y = 0; y < mapSize; y++)
                {
                    map[y] = new byte[mapSize];
                    for (int x = 0; x < mapSize; x++)
                    {
                        map[y][x] = RawData[y * mapSize + x];
                    }
                }
                Logger?.LogDebug($"MCSH: Successfully created {mapSize}x{mapSize} shadow map");
                return map;
            }
            catch (Exception ex)
            {
                AddError($"Error creating shadow map: {ex.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Gets the shadow intensity value at the specified coordinates
        /// </summary>
        /// <param name="x">The X coordinate</param>
        /// <param name="y">The Y coordinate</param>
        /// <returns>The shadow intensity value (0-255) or 0 if coordinates are invalid</returns>
        public byte GetShadowIntensity(int x, int y)
        {
            int mapSize = (int)Math.Sqrt(RawData.Length);
            if (mapSize * mapSize != RawData.Length)
            {
                AddError($"Shadow data size {RawData.Length} is not a perfect square");
                return 0;
            }
            
            if (x < 0 || x >= mapSize || y < 0 || y >= mapSize)
            {
                AddError($"Shadow map coordinates ({x}, {y}) out of bounds for {mapSize}x{mapSize} map");
                return 0;
            }
            
            try
            {
                return RawData[y * mapSize + x];
            }
            catch (Exception ex)
            {
                AddError($"Error getting shadow intensity at ({x}, {y}): {ex.Message}");
                return 0;
            }
        }
        
        /// <summary>
        /// Gets a hexadecimal representation of the data for debugging
        /// </summary>
        /// <param name="maxLength">The maximum number of bytes to include</param>
        /// <returns>A string containing the hexadecimal representation</returns>
        public string GetHexDump(int maxLength = 128)
        {
            if (RawData.Length == 0)
                return "[Empty]";
                
            int length = Math.Min(RawData.Length, maxLength);
            StringBuilder sb = new StringBuilder(length * 3);
            
            for (int i = 0; i < length; i++)
            {
                sb.Append(RawData[i].ToString("X2"));
                sb.Append(' ');
                
                if ((i + 1) % 16 == 0)
                    sb.Append('\n');
            }
            
            if (length < RawData.Length)
                sb.Append($"... ({RawData.Length - length} more bytes)");
                
            return sb.ToString();
        }
    }
} 