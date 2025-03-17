using System;
using System.Text;

namespace WCAnalyzer.Core.Models.PD4
{
    /// <summary>
    /// Represents a generic PD4 chunk that doesn't have a specific implementation.
    /// </summary>
    public class GenericChunk : PD4Chunk
    {
        private readonly string _signature;

        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => _signature;

        /// <summary>
        /// Initializes a new instance of the <see cref="GenericChunk"/> class.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The chunk data.</param>
        /// <exception cref="ArgumentException">Thrown if the signature is null or empty.</exception>
        public GenericChunk(string signature, byte[] data) : base(data)
        {
            if (string.IsNullOrEmpty(signature))
            {
                throw new ArgumentException("Signature cannot be null or empty.", nameof(signature));
            }

            _signature = signature;
        }

        /// <summary>
        /// Reads and parses the chunk data. For generic chunks, this does nothing.
        /// </summary>
        protected override void ReadData()
        {
            // Generic chunks don't perform any specific parsing
        }

        /// <summary>
        /// Returns a string representation of this chunk.
        /// </summary>
        public override string ToString()
        {
            return $"GenericChunk: Signature={Signature}, Size={Data.Length}";
        }

        /// <summary>
        /// Gets a hex dump of the chunk data, limited to a specified number of bytes.
        /// </summary>
        /// <param name="maxBytes">The maximum number of bytes to include in the dump. Set to 0 for all bytes.</param>
        /// <returns>A hex dump string of the chunk data.</returns>
        public string GetHexDump(int maxBytes = 128)
        {
            if (Data == null || Data.Length == 0)
            {
                return "(empty)";
            }

            var sb = new StringBuilder();
            int length = maxBytes > 0 && maxBytes < Data.Length ? maxBytes : Data.Length;

            for (int i = 0; i < length; i++)
            {
                if (i > 0 && i % 16 == 0)
                {
                    sb.AppendLine();
                }
                else if (i > 0)
                {
                    sb.Append(' ');
                }

                sb.Append(Data[i].ToString("X2"));
            }

            if (length < Data.Length)
            {
                sb.Append($"... ({Data.Length - length} more bytes)");
            }

            return sb.ToString();
        }
    }
} 