using System.IO;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MCRC chunk - Contains CRC data.
    /// </summary>
    public class MCRCChunk : PD4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MCRC";

        /// <summary>
        /// Gets the CRC value.
        /// </summary>
        public uint CRCValue { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MCRCChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MCRCChunk(byte[] data) : base(data)
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
                CRCValue = reader.ReadUInt32();
            }
        }

        /// <summary>
        /// Returns a string representation of this chunk.
        /// </summary>
        public override string ToString()
        {
            return $"MCRCChunk: CRC=0x{CRCValue:X8}";
        }
    }
} 