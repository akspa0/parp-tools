using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MDDF chunk in an ADT file, containing doodad (M2 model) placement information.
    /// </summary>
    public class MddfChunk : ADTChunk
    {
        /// <summary>
        /// The MDDF chunk signature
        /// </summary>
        public const string SIGNATURE = "MDDF";

        /// <summary>
        /// Flags for doodad placement.
        /// </summary>
        [Flags]
        public enum MDDFFlags : uint
        {
            /// <summary>
            /// No flags.
            /// </summary>
            None = 0x0,
            
            /// <summary>
            /// Biodome flag - sets internal flags to | 0x800 (WDOODADDEF.var0xC).
            /// </summary>
            Biodome = 0x1,
            
            /// <summary>
            /// Shrubbery flag - actual meaning unknown.
            /// </summary>
            Shrubbery = 0x2,
            
            /// <summary>
            /// Unknown flag 4.
            /// </summary>
            Unknown4 = 0x4,
            
            /// <summary>
            /// Unknown flag 8.
            /// </summary>
            Unknown8 = 0x8,
            
            /// <summary>
            /// Unknown flag 10.
            /// </summary>
            Unknown10 = 0x10,
            
            /// <summary>
            /// Liquid known flag.
            /// </summary>
            LiquidKnown = 0x20,
            
            /// <summary>
            /// Entry is a file data ID instead of an index into MMID.
            /// </summary>
            EntryIsFileDataId = 0x40,
            
            /// <summary>
            /// Unknown flag 100.
            /// </summary>
            Unknown100 = 0x100,
            
            /// <summary>
            /// Accept projected textures.
            /// </summary>
            AcceptProjTextures = 0x1000
        }

        /// <summary>
        /// Class representing a doodad definition entry.
        /// </summary>
        public class DoodadDefinition
        {
            /// <summary>
            /// Gets or sets the name ID or file data ID.
            /// </summary>
            public uint NameId { get; set; }
            
            /// <summary>
            /// Gets or sets the unique ID for the doodad.
            /// </summary>
            public uint UniqueId { get; set; }
            
            /// <summary>
            /// Gets or sets the position of the doodad.
            /// </summary>
            public Vector3 Position { get; set; }
            
            /// <summary>
            /// Gets or sets the rotation of the doodad in degrees.
            /// </summary>
            public Vector3 Rotation { get; set; }
            
            /// <summary>
            /// Gets or sets the scale of the doodad (1024 = 1.0f).
            /// </summary>
            public ushort Scale { get; set; }
            
            /// <summary>
            /// Gets or sets the flags for the doodad.
            /// </summary>
            public MDDFFlags Flags { get; set; }
        }
        
        /// <summary>
        /// Gets the list of doodad definitions.
        /// </summary>
        public List<DoodadDefinition> Definitions { get; } = new List<DoodadDefinition>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MddfChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public MddfChunk(byte[] data, ILogger? logger = null)
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
                AddError("No data to parse for MDDF chunk");
                return;
            }

            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Calculate number of entries (each entry is 36 bytes)
                    int entryCount = Data.Length / 36;
                    
                    for (int i = 0; i < entryCount; i++)
                    {
                        var def = new DoodadDefinition
                        {
                            NameId = reader.ReadUInt32(),
                            UniqueId = reader.ReadUInt32(),
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
                            Scale = reader.ReadUInt16(),
                            Flags = (MDDFFlags)reader.ReadUInt16()
                        };
                        
                        Definitions.Add(def);
                        Logger?.LogDebug($"MDDF: Parsed doodad entry {i}: NameId={def.NameId}, UniqueId={def.UniqueId}, Position={def.Position}");
                    }
                    
                    Logger?.LogDebug($"MDDF: Parsed {Definitions.Count} doodad definitions");
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MDDF chunk: {ex.Message}");
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
                // Write each doodad definition
                foreach (var def in Definitions)
                {
                    writer.Write(def.NameId);
                    writer.Write(def.UniqueId);
                    
                    // Position
                    writer.Write(def.Position.X);
                    writer.Write(def.Position.Y);
                    writer.Write(def.Position.Z);
                    
                    // Rotation
                    writer.Write(def.Rotation.X);
                    writer.Write(def.Rotation.Y);
                    writer.Write(def.Rotation.Z);
                    
                    writer.Write(def.Scale);
                    writer.Write((ushort)def.Flags);
                }
            }
            catch (Exception ex)
            {
                AddError($"Error writing MDDF chunk: {ex.Message}");
            }
        }
    }
} 