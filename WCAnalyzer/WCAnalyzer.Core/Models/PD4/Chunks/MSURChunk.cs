using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MSUR chunk - Contains surface data.
    /// </summary>
    public class MSURChunk : PD4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSUR";

        /// <summary>
        /// Gets the surface data entries.
        /// </summary>
        public List<SurfaceEntry> Entries { get; private set; } = new List<SurfaceEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSURChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSURChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Entries.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each entry is 48 bytes (0x30)
                int entrySize = 48;
                int entryCount = Data.Length / entrySize;

                for (int i = 0; i < entryCount; i++)
                {
                    var startPosition = ms.Position;
                    
                    var entry = new SurfaceEntry
                    {
                        Index = i,
                        ID = reader.ReadUInt32(),
                        Flags = reader.ReadUInt32(),
                        MSPIFirstIndex = reader.ReadUInt32(),
                        MSPIIndexCount = reader.ReadUInt32(),
                        MSVIFirstIndex = reader.ReadUInt32(),
                        MSVIIndexCount = reader.ReadUInt32()
                    };

                    // Read the material name (20 bytes, null-terminated)
                    byte[] materialNameBytes = reader.ReadBytes(20);
                    int endPos = 0;
                    while (endPos < materialNameBytes.Length && materialNameBytes[endPos] != 0)
                    {
                        endPos++;
                    }
                    entry.MaterialName = Encoding.ASCII.GetString(materialNameBytes, 0, endPos);

                    // Skip to the end of this entry
                    ms.Position = startPosition + entrySize;
                    
                    Entries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents a surface data entry.
        /// </summary>
        public class SurfaceEntry
        {
            /// <summary>
            /// Gets or sets the index of this entry in the chunk.
            /// </summary>
            public int Index { get; set; }

            /// <summary>
            /// Gets or sets the surface ID.
            /// </summary>
            public uint ID { get; set; }

            /// <summary>
            /// Gets or sets the surface flags.
            /// </summary>
            public uint Flags { get; set; }

            /// <summary>
            /// Gets or sets the index of the first MSPI vertex.
            /// </summary>
            public uint MSPIFirstIndex { get; set; }

            /// <summary>
            /// Gets or sets the count of MSPI vertices.
            /// </summary>
            public uint MSPIIndexCount { get; set; }

            /// <summary>
            /// Gets or sets the index of the first MSVI entry.
            /// </summary>
            public uint MSVIFirstIndex { get; set; }

            /// <summary>
            /// Gets or sets the count of MSVI entries.
            /// </summary>
            public uint MSVIIndexCount { get; set; }

            /// <summary>
            /// Gets or sets the material name.
            /// </summary>
            public string MaterialName { get; set; } = string.Empty;

            /// <summary>
            /// Returns a string representation of this entry.
            /// </summary>
            public override string ToString()
            {
                return $"Surface[{Index}]: ID={ID}, Material=\"{MaterialName}\", Flags={Flags}, MSPI[{MSPIFirstIndex}:{MSPIFirstIndex + MSPIIndexCount - 1}], MSVI[{MSVIFirstIndex}:{MSVIFirstIndex + MSVIIndexCount - 1}]";
            }
        }
    }
} 