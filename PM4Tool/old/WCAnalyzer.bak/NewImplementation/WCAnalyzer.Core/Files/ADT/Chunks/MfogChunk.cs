using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MFOG chunk - Contains information about fog in the ADT
    /// </summary>
    public class MfogChunk : ADTChunk
    {
        /// <summary>
        /// The MFOG chunk signature
        /// </summary>
        public const string SIGNATURE = "MFOG";

        /// <summary>
        /// Structure representing fog information
        /// </summary>
        public class FogInfo
        {
            /// <summary>
            /// Gets or sets the fog flags
            /// </summary>
            public uint Flags { get; set; }

            /// <summary>
            /// Gets or sets the position of the fog
            /// </summary>
            public Vector3 Position { get; set; }

            /// <summary>
            /// Gets or sets the small fog radius
            /// </summary>
            public float SmallRadius { get; set; }

            /// <summary>
            /// Gets or sets the large fog radius
            /// </summary>
            public float LargeRadius { get; set; }

            /// <summary>
            /// Gets or sets the fog end
            /// </summary>
            public float End { get; set; }

            /// <summary>
            /// Gets or sets the fog start multiplier
            /// </summary>
            public float StartMultiplier { get; set; }

            /// <summary>
            /// Gets or sets the color of the fog
            /// </summary>
            public uint Color { get; set; }
            
            /// <summary>
            /// Gets the red component of the fog color (0-255)
            /// </summary>
            public byte Red => (byte)((Color >> 16) & 0xFF);
            
            /// <summary>
            /// Gets the green component of the fog color (0-255)
            /// </summary>
            public byte Green => (byte)((Color >> 8) & 0xFF);
            
            /// <summary>
            /// Gets the blue component of the fog color (0-255)
            /// </summary>
            public byte Blue => (byte)(Color & 0xFF);
            
            /// <summary>
            /// Gets the alpha component of the fog color (0-255)
            /// </summary>
            public byte Alpha => (byte)((Color >> 24) & 0xFF);
        }

        /// <summary>
        /// Gets the list of fog information
        /// </summary>
        public List<FogInfo> FogInfos { get; private set; } = new List<FogInfo>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MfogChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MfogChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MFOG chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MFOG chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Each fog entry is 48 bytes
                const int FogEntrySize = 48;
                
                // Calculate the number of fog entries
                int count = Data.Length / FogEntrySize;
                
                if (Data.Length % FogEntrySize != 0)
                {
                    Logger?.LogWarning($"MFOG chunk data size {Data.Length} is not a multiple of {FogEntrySize}");
                }
                
                // Read fog entries
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        var fogInfo = new FogInfo
                        {
                            Flags = reader.ReadUInt32(),
                            Position = new Vector3(
                                reader.ReadSingle(), // X
                                reader.ReadSingle(), // Y
                                reader.ReadSingle()  // Z
                            ),
                            SmallRadius = reader.ReadSingle(),
                            LargeRadius = reader.ReadSingle(),
                            End = reader.ReadSingle(),
                            StartMultiplier = reader.ReadSingle(),
                            Color = reader.ReadUInt32()
                        };
                        
                        // Skip 16 bytes (4 uint values) - reserved/unused
                        reader.BaseStream.Seek(16, SeekOrigin.Current);
                        
                        FogInfos.Add(fogInfo);
                    }
                    catch (EndOfStreamException)
                    {
                        Logger?.LogWarning($"MFOG: Unexpected end of stream while reading fog entry {i}");
                        break;
                    }
                }
                
                Logger?.LogDebug($"MFOG: Read {FogInfos.Count} fog entries");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MFOG chunk: {ex.Message}");
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
                Logger?.LogError("Cannot write MFOG chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the data size (48 bytes per fog entry)
                int dataSize = FogInfos.Count * 48;
                writer.Write(dataSize);
                
                // Write each fog entry
                foreach (var fogInfo in FogInfos)
                {
                    writer.Write(fogInfo.Flags);
                    writer.Write(fogInfo.Position.X);
                    writer.Write(fogInfo.Position.Y);
                    writer.Write(fogInfo.Position.Z);
                    writer.Write(fogInfo.SmallRadius);
                    writer.Write(fogInfo.LargeRadius);
                    writer.Write(fogInfo.End);
                    writer.Write(fogInfo.StartMultiplier);
                    writer.Write(fogInfo.Color);
                    
                    // Write 16 bytes (4 uint values) of zeros for reserved/unused fields
                    writer.Write(0U);
                    writer.Write(0U);
                    writer.Write(0U);
                    writer.Write(0U);
                }
                
                Logger?.LogDebug($"MFOG: Wrote {FogInfos.Count} fog entries");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MFOG chunk: {ex.Message}");
                throw;
            }
        }
    }
} 