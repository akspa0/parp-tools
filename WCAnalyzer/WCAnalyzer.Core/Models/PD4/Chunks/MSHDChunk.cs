using System;
using System.IO;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MSHD chunk - Contains shadow data. According to documentation:
    /// struct SMMapTerrainShadow
    /// {
    ///    uint32_t Width;
    ///    uint32_t Height;
    ///    uint8_t Shadows[Width*Height];
    /// };
    /// </summary>
    public class MSHDChunk : PD4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSHD";

        /// <summary>
        /// Gets the shadow width.
        /// </summary>
        public uint Width { get; private set; }

        /// <summary>
        /// Gets the shadow height.
        /// </summary>
        public uint Height { get; private set; }

        /// <summary>
        /// Gets the shadow data.
        /// </summary>
        public byte[] ShadowData { get; private set; } = new byte[0];

        /// <summary>
        /// Initializes a new instance of the <see cref="MSHDChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSHDChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Read header values
                Width = reader.ReadUInt32();
                Height = reader.ReadUInt32();

                // Expected size = Width * Height
                long expectedSize = Width * Height;
                long remainingData = ms.Length - ms.Position;

                if (remainingData >= expectedSize)
                {
                    // Read shadow data
                    ShadowData = reader.ReadBytes((int)expectedSize);
                }
                else
                {
                    // If data is incomplete, allocate an empty array
                    ShadowData = new byte[0];
                }
            }
        }

        /// <summary>
        /// Gets the shadow value at the specified position.
        /// </summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        /// <returns>The shadow value at the specified position, or 0 if out of bounds.</returns>
        public byte GetShadowAt(int x, int y)
        {
            if (x < 0 || x >= Width || y < 0 || y >= Height || ShadowData.Length == 0)
            {
                return 0;
            }

            int index = y * (int)Width + x;
            if (index < ShadowData.Length)
            {
                return ShadowData[index];
            }

            return 0;
        }

        /// <summary>
        /// Returns a string representation of this chunk.
        /// </summary>
        public override string ToString()
        {
            return $"MSHDChunk: Width={Width}, Height={Height}, ShadowData.Length={ShadowData.Length}";
        }
    }
} 