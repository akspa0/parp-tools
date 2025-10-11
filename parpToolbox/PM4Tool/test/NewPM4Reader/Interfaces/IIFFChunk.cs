namespace NewPM4Reader.Interfaces
{
    /// <summary>
    /// Interface for IFF-style chunks in a file format.
    /// </summary>
    public interface IIFFChunk
    {
        /// <summary>
        /// Gets the signature/identifier of the chunk.
        /// </summary>
        string Signature { get; }
    }
} 