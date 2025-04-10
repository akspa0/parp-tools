using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCDD chunk - Contains detail doodad information for a map chunk
    /// </summary>
    public class McddChunk : ADTChunk
    {
        /// <summary>
        /// The MCDD chunk signature
        /// </summary>
        public const string SIGNATURE = "MCDD";

        /// <summary>
        /// Structure representing a detail doodad entry
        /// </summary>
        public class DetailDoodadEntry
        {
            /// <summary>
            /// Gets or sets the file data ID for the doodad model
            /// </summary>
            public uint FileDataId { get; set; }

            /// <summary>
            /// Gets or sets the position of the doodad
            /// </summary>
            public Vector3 Position { get; set; }

            /// <summary>
            /// Gets or sets the rotation of the doodad (in radians)
            /// </summary>
            public Vector3 Rotation { get; set; }

            /// <summary>
            /// Gets or sets the scale of the doodad
            /// </summary>
            public float Scale { get; set; }

            /// <summary>
            /// Gets or sets the flags for the doodad
            /// </summary>
            public uint Flags { get; set; }
        }

        /// <summary>
        /// Gets the list of detail doodad entries
        /// </summary>
        public List<DetailDoodadEntry> DetailDoodads { get; private set; } = new List<DetailDoodadEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="McddChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McddChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MCDD chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCDD chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Each entry is typically 36 bytes
                // 4 (FileDataId) + 12 (Position) + 12 (Rotation) + 4 (Scale) + 4 (Flags)
                const int EntrySize = 36;
                
                // Validate data size
                if (Data.Length % EntrySize != 0)
                {
                    Logger?.LogWarning($"MCDD chunk data size {Data.Length} is not a multiple of {EntrySize}");
                }
                
                // Calculate how many entries we should have
                int count = Data.Length / EntrySize;
                
                // Read detail doodad entries
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        var entry = new DetailDoodadEntry
                        {
                            FileDataId = reader.ReadUInt32(),
                            Position = new Vector3(
                                reader.ReadSingle(), // X
                                reader.ReadSingle(), // Y
                                reader.ReadSingle()  // Z
                            ),
                            Rotation = new Vector3(
                                reader.ReadSingle(), // X
                                reader.ReadSingle(), // Y
                                reader.ReadSingle()  // Z
                            ),
                            Scale = reader.ReadSingle(),
                            Flags = reader.ReadUInt32()
                        };
                        
                        DetailDoodads.Add(entry);
                    }
                    catch (EndOfStreamException)
                    {
                        Logger?.LogWarning($"MCDD: Unexpected end of stream while reading detail doodad entry {i}");
                        break;
                    }
                }
                
                Logger?.LogDebug($"MCDD: Read {DetailDoodads.Count} detail doodad entries");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCDD chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Writes the chunk data to a binary writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger?.LogError("Cannot write MCDD chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the data size (36 bytes per entry)
                int dataSize = DetailDoodads.Count * 36;
                writer.Write(dataSize);
                
                // Write each detail doodad entry
                foreach (var entry in DetailDoodads)
                {
                    writer.Write(entry.FileDataId);
                    
                    // Position
                    writer.Write(entry.Position.X);
                    writer.Write(entry.Position.Y);
                    writer.Write(entry.Position.Z);
                    
                    // Rotation
                    writer.Write(entry.Rotation.X);
                    writer.Write(entry.Rotation.Y);
                    writer.Write(entry.Rotation.Z);
                    
                    // Scale and flags
                    writer.Write(entry.Scale);
                    writer.Write(entry.Flags);
                }
                
                Logger?.LogDebug($"MCDD: Wrote {DetailDoodads.Count} detail doodad entries");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCDD chunk: {ex.Message}");
                throw;
            }
        }
    }
} 