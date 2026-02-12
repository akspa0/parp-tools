using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCIN chunk - Contains pointers to MCNK chunks and their sizes.
    /// Only used in pre-Cataclysm ADT files; removed with split files in Cataclysm.
    /// </summary>
    public class McinChunk : ADTChunk
    {
        /// <summary>
        /// The MCIN chunk signature
        /// </summary>
        public const string SIGNATURE = "MCIN";

        /// <summary>
        /// Gets the list of map chunk entries
        /// </summary>
        public List<MapChunkEntry> Entries { get; } = new List<MapChunkEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="McinChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McinChunk(byte[] data, ILogger? logger = null)
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
                AddError("No data to parse for MCIN chunk");
                return;
            }

            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Each MCIN entry is 16 bytes (4 uint32 values)
                    int entrySize = 16;
                    int entryCount = Data.Length / entrySize;

                    if (entryCount != 256)
                    {
                        AddError($"Expected 256 entries (16x16 grid), but found {entryCount}");
                    }

                    for (int i = 0; i < entryCount; i++)
                    {
                        var entry = new MapChunkEntry
                        {
                            Offset = reader.ReadUInt32(),     // Absolute offset in the file
                            Size = reader.ReadUInt32(),       // Size of the MCNK chunk
                            Flags = reader.ReadUInt32(),      // Always 0 in file (used by client)
                            AsyncId = reader.ReadUInt32()     // Not used in file (used by client)
                        };

                        Entries.Add(entry);
                        Logger?.LogDebug($"MCIN: Entry {i} - Offset: 0x{entry.Offset:X8}, Size: {entry.Size}");
                    }

                    Logger?.LogDebug($"MCIN: Parsed {Entries.Count} map chunk entries");
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MCIN chunk: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets a map chunk entry by its index in the grid
        /// </summary>
        /// <param name="x">The X coordinate in the 16x16 grid (0-15)</param>
        /// <param name="y">The Y coordinate in the 16x16 grid (0-15)</param>
        /// <returns>The map chunk entry at the specified coordinates, or null if out of range</returns>
        public MapChunkEntry? GetMapChunkEntry(int x, int y)
        {
            if (x < 0 || x >= 16 || y < 0 || y >= 16)
            {
                AddError($"Map chunk coordinates must be between 0 and 15, got ({x}, {y})");
                return null;
            }

            int index = y * 16 + x;
            if (index >= Entries.Count)
            {
                AddError($"Map chunk index {index} is out of range (0-{Entries.Count - 1})");
                return null;
            }

            return Entries[index];
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
                // Write each entry
                foreach (var entry in Entries)
                {
                    writer.Write(entry.Offset);
                    writer.Write(entry.Size);
                    writer.Write(entry.Flags);
                    writer.Write(entry.AsyncId);
                }
            }
            catch (Exception ex)
            {
                AddError($"Error writing MCIN chunk: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Represents an entry in the MCIN chunk, which points to a MCNK chunk
    /// </summary>
    public class MapChunkEntry
    {
        /// <summary>
        /// Gets or sets the absolute offset of the MCNK chunk in the file
        /// </summary>
        public uint Offset { get; set; }

        /// <summary>
        /// Gets or sets the size of the MCNK chunk in bytes
        /// </summary>
        public uint Size { get; set; }

        /// <summary>
        /// Gets or sets the flags (always 0 in the file, used by the client)
        /// </summary>
        public uint Flags { get; set; }

        /// <summary>
        /// Gets or sets the async ID (not in the ADT file, used by the client)
        /// </summary>
        public uint AsyncId { get; set; }
    }
} 