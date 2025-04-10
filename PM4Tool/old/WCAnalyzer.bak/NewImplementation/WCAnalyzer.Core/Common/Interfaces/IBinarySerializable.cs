namespace WCAnalyzer.Core.Common.Interfaces
{
    /// <summary>
    /// Interface for types that can be serialized to and from binary data
    /// </summary>
    public interface IBinarySerializable
    {
        /// <summary>
        /// Parse binary data
        /// </summary>
        /// <returns>True if parsing was successful, false otherwise</returns>
        bool Parse();
        
        /// <summary>
        /// Write the object to binary data
        /// </summary>
        /// <returns>The binary representation</returns>
        byte[] Write();
    }
} 