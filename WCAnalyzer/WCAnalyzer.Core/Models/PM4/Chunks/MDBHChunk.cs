using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MDBH chunk - Contains destructible building header data.
    /// </summary>
    public class MDBHChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MDBH";

        /// <summary>
        /// Gets the destructible building entries.
        /// </summary>
        public List<DestructibleBuildingHeader> Entries { get; private set; } = new List<DestructibleBuildingHeader>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MDBHChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MDBHChunk(byte[] data) : base(data)
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
                // First field is the count of entries
                uint count = reader.ReadUInt32();

                for (int i = 0; i < count; i++)
                {
                    // Read the index chunk
                    string indexSignature = new string(reader.ReadChars(4));
                    uint indexSize = reader.ReadUInt32();
                    byte[] indexData = reader.ReadBytes((int)indexSize);

                    // Read the three filename chunks
                    string[] filenameSignatures = new string[3];
                    uint[] filenameSizes = new uint[3];
                    byte[][] filenameData = new byte[3][];

                    for (int j = 0; j < 3; j++)
                    {
                        filenameSignatures[j] = new string(reader.ReadChars(4));
                        filenameSizes[j] = reader.ReadUInt32();
                        filenameData[j] = reader.ReadBytes((int)filenameSizes[j]);
                    }

                    var header = new DestructibleBuildingHeader
                    {
                        Index = indexSignature == "MDBI" ? ParseMDBI(indexData) : 0,
                        Filenames = new string[3]
                    };

                    // Parse each filename chunk
                    for (int j = 0; j < 3; j++)
                    {
                        if (filenameSignatures[j] == "MDBF")
                        {
                            header.Filenames[j] = ParseMDBF(filenameData[j]);
                        }
                    }

                    Entries.Add(header);
                }
            }
        }

        /// <summary>
        /// Parses the MDBI chunk data.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <returns>The destructible building index.</returns>
        private uint ParseMDBI(byte[] data)
        {
            using (var ms = new MemoryStream(data))
            using (var reader = new BinaryReader(ms))
            {
                return reader.ReadUInt32();
            }
        }

        /// <summary>
        /// Parses the MDBF chunk data.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <returns>The destructible building filename.</returns>
        private string ParseMDBF(byte[] data)
        {
            using (var ms = new MemoryStream(data))
            using (var reader = new BinaryReader(ms))
            {
                // Read until we find a null terminator or reach the end of the data
                List<char> chars = new List<char>();
                while (ms.Position < ms.Length)
                {
                    char c = reader.ReadChar();
                    if (c == 0)
                        break;
                    chars.Add(c);
                }
                return new string(chars.ToArray());
            }
        }

        /// <summary>
        /// Represents a destructible building header.
        /// </summary>
        public class DestructibleBuildingHeader
        {
            /// <summary>
            /// Gets or sets the destructible building index.
            /// </summary>
            public uint Index { get; set; }

            /// <summary>
            /// Gets or sets the destructible building filenames.
            /// </summary>
            public string[] Filenames { get; set; }
        }
    }
} 