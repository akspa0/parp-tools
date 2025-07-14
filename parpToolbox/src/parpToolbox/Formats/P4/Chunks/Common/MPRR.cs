namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// MPRR – Properties record list (value pairs). Each entry is a 4-byte structure that appears
/// to encode two ushort values. Exact semantics remain unknown but the chunk is required by the
/// legacy exporters and some analysis utilities.
/// </summary>
internal sealed class MprrChunk : IIffChunk, IBinarySerializable
{
    /// <summary>Canonical FourCC signature.</summary>
    public const string Signature = "MPRR";

    /// <summary>Single 4-byte entry.</summary>
    /// <param name="Value1">First ushort.</param>
    /// <param name="Value2">Second ushort.</param>
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
