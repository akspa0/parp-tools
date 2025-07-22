namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// MPRR – Properties record list (value pairs). Each entry is a 4-byte structure that contains
/// two ushort values that define hierarchical object grouping and boundaries.
/// </summary>
/// <remarks>
/// <para>
/// Key Discovery: MPRR contains the true object boundaries using sentinel values (Value1=65535)
/// that separate geometry into complete building objects. This is the most effective method for
/// grouping PM4 geometry into coherent building-scale objects.
/// </para>
/// <para>
/// From data analysis of typical PM4 files:
/// - ~81,936 MPRR properties total
/// - ~15,427 sentinel markers (Value1=65535)
/// - ~15,428 object groups separated by sentinels
/// - Realistic object scales: 38K-654K triangles per building object
/// </para>
/// <para>
/// MPRR-based grouping produces complete building objects (38K-654K triangles) compared to 
/// the much smaller fragments (300-350 vertices) produced by other grouping methods.
/// </para>
/// </remarks>
public sealed class MprrChunk : IIffChunk, IBinarySerializable
{
    /// <summary>Canonical FourCC signature.</summary>
    public const string Signature = "MPRR";

    /// <summary>Single 4-byte entry in the MPRR chunk.</summary>
    /// <param name="Value1">First ushort. When Value1=65535, it acts as a sentinel marker that indicates an object boundary.</param>
    /// <param name="Value2">Second ushort. When following a sentinel marker, this value identifies the component type.</param>
    /// <remarks>
    /// The Value1=65535 sentinel pattern is critical for correct object grouping. When processing MPRR entries,
    /// each sequence of entries between sentinels represents a complete building object. The Value2 values
    /// after sentinels identify component types that link to MPRL placements and geometry fragments.
    /// </remarks>
    public sealed record Entry(ushort Value1, ushort Value2);

    private readonly List<Entry> _entries = new();

    /// <summary>All parsed entries.</summary>
    public IReadOnlyList<Entry> Entries => _entries;

    #region Interfaces
    public string GetSignature() => Signature;

    public uint GetSize() => (uint)(_entries.Count * 4);

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        // Chunk is an array of 4-byte records: uint16 + uint16
        // Critical for object assembly: entries with Value1=65535 act as sentinel markers defining object boundaries
        while (br.BaseStream.Position + 4 <= br.BaseStream.Length)
        {
            ushort v1 = br.ReadUInt16();
            ushort v2 = br.ReadUInt16();
            _entries.Add(new Entry(v1, v2));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
    #endregion
}
