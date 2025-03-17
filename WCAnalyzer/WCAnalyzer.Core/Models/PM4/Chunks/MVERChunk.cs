using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MVER chunk - Contains version information.
    /// </summary>
    public class MVERChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MVER";

        /// <summary>
        /// Gets the version number.
        /// </summary>
        public uint Version { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MVERChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MVERChunk(byte[] data) : base(data)
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
                Version = reader.ReadUInt32();
            }
        }
    }
} 