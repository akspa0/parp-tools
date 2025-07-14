namespace ParpToolbox.Formats.PM4.Chunks;

using System.IO;

/// <summary>
/// Minimal replacement for the legacy Warcraft.NET <c>IBinarySerializable</c> interface.
/// Provides a simple contract for models that can load / save themselves from binary data.
/// </summary>
internal interface IBinarySerializable
{
    /// <summary>Populates this instance from a byte array.</summary>
    void LoadBinaryData(byte[] inData);

    /// <summary>Populates this instance by reading from an open <see cref="BinaryReader"/>.</summary>
    void Load(BinaryReader br);

    /// <summary>Serialises this instance to a byte array starting at an optional offset.</summary>
    byte[] Serialize(long offset = 0);

    /// <summary>Gets the binary size of this structure when serialised.</summary>
    uint GetSize();
}

/// <summary>
/// Contract for binary chunks that have a FourCC signature.
/// </summary>
internal interface IIffChunk
{
    /// <summary>Gets the four-character ASCII signature of the chunk (e.g. "MSLK").</summary>
    string GetSignature();
}
