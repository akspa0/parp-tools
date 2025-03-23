using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents a MODF chunk in an ADT file, containing WMO (World Map Object) placement information.
    /// </summary>
    public class ModfChunk : ADTChunk
    {
        /// <summary>
        /// The MODF chunk signature
        /// </summary>
        public const string SIGNATURE = "MODF";

        /// <summary>
        /// Flags for WMO placement.
        /// </summary>
        [Flags]
        public enum MODFFlags : uint
        {
            /// <summary>
            /// No flags.
            /// </summary>
            None = 0x0,
            
            /// <summary>
            /// Destroys WMO when attacked.
            /// </summary>
            DestroyOnAttacked = 0x1,
            
            /// <summary>
            /// Used for taxi paths.
            /// </summary>
            UsedForTaxiPath = 0x2,
            
            /// <summary>
            /// Entry is a file data ID instead of an index into MWID.
            /// </summary>
            EntryIsFileDataId = 0x40
        }

        /// <summary>
        /// Class representing a WMO placement definition.
        /// </summary>
        public class WmoDefinition
        {
            /// <summary>
            /// Gets or sets the name ID or file data ID.
            /// </summary>
            public uint NameId { get; set; }
            
            /// <summary>
            /// Gets or sets the unique ID for the WMO.
            /// </summary>
            public uint UniqueId { get; set; }
            
            /// <summary>
            /// Gets or sets the position of the WMO.
            /// </summary>
            public Vector3 Position { get; set; }
            
            /// <summary>
            /// Gets or sets the rotation of the WMO in degrees.
            /// </summary>
            public Vector3 Rotation { get; set; }
            
            /// <summary>
            /// Gets or sets the lower boundary box coordinates.
            /// </summary>
            public Vector3 BoundingBoxLower { get; set; }
            
            /// <summary>
            /// Gets or sets the upper boundary box coordinates.
            /// </summary>
            public Vector3 BoundingBoxUpper { get; set; }
            
            /// <summary>
            /// Gets or sets the flags for the WMO.
            /// </summary>
            public MODFFlags Flags { get; set; }
            
            /// <summary>
            /// Gets or sets the doodad set index.
            /// </summary>
            public ushort DoodadSetIndex { get; set; }
            
            /// <summary>
            /// Gets or sets the name set index.
            /// </summary>
            public ushort NameSetIndex { get; set; }
            
            /// <summary>
            /// Gets or sets the WMO group ID (for WMO instances).
            /// </summary>
            public ushort WmoGroupId { get; set; }
        }
        
        /// <summary>
        /// Gets the list of WMO definitions.
        /// </summary>
        public List<WmoDefinition> Definitions { get; } = new List<WmoDefinition>();

        /// <summary>
        /// Initializes a new instance of the <see cref="ModfChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public ModfChunk(byte[] data, ILogger? logger = null)
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
                AddError("No data to parse for MODF chunk");
                return;
            }

            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Calculate number of entries (each entry is 64 bytes)
                    int entryCount = Data.Length / 64;
                    
                    for (int i = 0; i < entryCount; i++)
                    {
                        var def = new WmoDefinition
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
                            BoundingBoxLower = new Vector3(
                                reader.ReadSingle(), // X
                                reader.ReadSingle(), // Y
                                reader.ReadSingle()  // Z
                            ),
                            BoundingBoxUpper = new Vector3(
                                reader.ReadSingle(), // X
                                reader.ReadSingle(), // Y
                                reader.ReadSingle()  // Z
                            ),
                            Flags = (MODFFlags)reader.ReadUInt16(),
                            DoodadSetIndex = reader.ReadUInt16(),
                            NameSetIndex = reader.ReadUInt16(),
                            WmoGroupId = reader.ReadUInt16()
                        };
                        
                        Definitions.Add(def);
                        Logger?.LogDebug($"MODF: Parsed WMO entry {i}: NameId={def.NameId}, UniqueId={def.UniqueId}, Position={def.Position}");
                    }
                    
                    Logger?.LogDebug($"MODF: Parsed {Definitions.Count} WMO definitions");
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MODF chunk: {ex.Message}");
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
                // Write each WMO definition
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
                    
                    // Bounding Box Lower
                    writer.Write(def.BoundingBoxLower.X);
                    writer.Write(def.BoundingBoxLower.Y);
                    writer.Write(def.BoundingBoxLower.Z);
                    
                    // Bounding Box Upper
                    writer.Write(def.BoundingBoxUpper.X);
                    writer.Write(def.BoundingBoxUpper.Y);
                    writer.Write(def.BoundingBoxUpper.Z);
                    
                    writer.Write((ushort)def.Flags);
                    writer.Write(def.DoodadSetIndex);
                    writer.Write(def.NameSetIndex);
                    writer.Write(def.WmoGroupId);
                }
            }
            catch (Exception ex)
            {
                AddError($"Error writing MODF chunk: {ex.Message}");
            }
        }
    }
} 