using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using System.Text;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MSUR chunk - Contains surface/material data for the PM4 format
    /// This chunk typically stores material properties, textures, and rendering parameters
    /// </summary>
    public class MsurChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSUR")
        /// </summary>
        public const string SIGNATURE = "MSUR";
        
        /// <summary>
        /// Represents a surface entry in the MSUR chunk
        /// </summary>
        public class SurfaceData
        {
            /// <summary>
            /// Gets or sets the material ID
            /// </summary>
            public ushort MaterialId { get; set; }
            
            /// <summary>
            /// Gets or sets the MSVI index
            /// </summary>
            public ushort MsviIndex { get; set; }
            
            /// <summary>
            /// Gets or sets the flags
            /// According to documentation, this contains various rendering flags
            /// </summary>
            public uint Flags { get; set; }
            
            /// <summary>
            /// Gets or sets the unk0x08 value
            /// Documentation notes this is always 0 in version_48
            /// </summary>
            public uint Unk0x08 { get; set; }
            
            /// <summary>
            /// Gets or sets the unk0x0c value
            /// Documentation notes this is always 0 in version_48
            /// </summary>
            public uint Unk0x0C { get; set; }
            
            /// <summary>
            /// Gets or sets the unk0x10 value
            /// Possible padding according to documentation
            /// </summary>
            public uint Unk0x10 { get; set; }
            
            /// <summary>
            /// Returns a string representation of this surface
            /// </summary>
            public override string ToString()
            {
                return $"Surface: Material={MaterialId}, MSVI={MsviIndex}, Flags=0x{Flags:X8}";
            }
        }
        
        /// <summary>
        /// Gets the list of surfaces
        /// </summary>
        public List<SurfaceData> Surfaces { get; private set; } = new List<SurfaceData>();
        
        /// <summary>
        /// Creates a new MSUR chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MsurChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }
        
        /// <summary>
        /// Creates a new empty MSUR chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MsurChunk(ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
        }
        
        /// <summary>
        /// Parses the chunk data
        /// </summary>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public override bool Parse()
        {
            try
            {
                if (Data.Length < 4)
                {
                    LogWarning($"MSUR chunk data is too small: {Data.Length} bytes");
                    return false;
                }
                
                Surfaces.Clear();
                
                using (MemoryStream ms = new MemoryStream(Data))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    // Each surface entry is 20 bytes according to documentation
                    int entrySize = 20;
                    int entryCount = Data.Length / entrySize;
                    
                    if (Data.Length % entrySize != 0)
                    {
                        LogWarning($"MSUR chunk data size ({Data.Length}) is not a multiple of entry size ({entrySize} bytes)");
                    }
                    
                    LogInformation($"Surface entry count: {entryCount}");
                    
                    for (int i = 0; i < entryCount; i++)
                    {
                        var surface = new SurfaceData
                        {
                            MaterialId = reader.ReadUInt16(),
                            MsviIndex = reader.ReadUInt16(),
                            Flags = reader.ReadUInt32(),
                            Unk0x08 = reader.ReadUInt32(),
                            Unk0x0C = reader.ReadUInt32(),
                            Unk0x10 = reader.ReadUInt32()
                        };
                        
                        Surfaces.Add(surface);
                        
                        LogDebug($"Surface {i}: Material={surface.MaterialId}, MSVI={surface.MsviIndex}, Flags=0x{surface.Flags:X8}");
                    }
                    
                    LogInformation($"Parsed {Surfaces.Count} surface entries");
                    
                    IsParsed = true;
                    return true;
                }
            }
            catch (Exception ex)
            {
                LogError($"Error parsing MSUR chunk: {ex.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// Gets the size of the surface data
        /// </summary>
        /// <returns>Size in bytes</returns>
        public int GetDataSize()
        {
            return _data.Length;
        }
        
        /// <summary>
        /// Attempts to get the material name if present
        /// The name is typically stored as a null-terminated string in the data
        /// </summary>
        /// <returns>Material name or empty string if not found</returns>
        public string GetMaterialName()
        {
            // Try to find a null-terminated string at the beginning of the data
            int nullTerminator = Array.IndexOf(_data, (byte)0);
            if (nullTerminator > 0)
            {
                try
                {
                    return System.Text.Encoding.ASCII.GetString(_data, 0, nullTerminator);
                }
                catch (Exception ex)
                {
                    Logger?.LogWarning($"Failed to extract material name: {ex.Message}");
                }
            }
            
            return string.Empty;
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            string materialName = GetMaterialName();
            if (!string.IsNullOrEmpty(materialName))
            {
                return $"{SIGNATURE} Chunk: Material '{materialName}', {GetDataSize()} bytes";
            }
            
            return $"{SIGNATURE} Chunk: {GetDataSize()} bytes";
        }
        
        /// <summary>
        /// Writes this chunk to a byte array
        /// </summary>
        /// <returns>Byte array containing chunk data</returns>
        public override byte[] Write()
        {
            return _data;
        }
        
        /// <summary>
        /// Gets a surface at the specified index
        /// </summary>
        /// <param name="index">The surface index</param>
        /// <returns>The surface at the specified index, or null if index is out of range</returns>
        public SurfaceData? GetSurface(int index)
        {
            if (index >= 0 && index < Surfaces.Count)
                return Surfaces[index];
                
            LogWarning($"Surface index {index} is out of range (0-{Surfaces.Count - 1})");
            return null;
        }
        
        /// <summary>
        /// Helper method to get a binary string representation of the flags
        /// Useful for debugging
        /// </summary>
        /// <param name="surface">The surface to get flags for</param>
        /// <returns>A binary string representation of the flags</returns>
        public string GetFlagsBinary(SurfaceData surface)
        {
            StringBuilder sb = new StringBuilder(32);
            uint flags = surface.Flags;
            
            for (int i = 31; i >= 0; i--)
            {
                sb.Append((flags & (1u << i)) != 0 ? '1' : '0');
                
                // Add a space every 8 bits for readability
                if (i % 8 == 0 && i > 0)
                    sb.Append(' ');
            }
            
            return sb.ToString();
        }
    }
} 