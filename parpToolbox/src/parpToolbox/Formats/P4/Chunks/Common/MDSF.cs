namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// MDSF – Surface-to-Material lookup. Each 8-byte entry maps an MSUR surface index to an MDOS material index.
/// </summary>
internal sealed class MdsfChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MDSF";

    public sealed record Entry(uint MsurIndex, uint MdosIndex);
    private readonly List<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries;

    public string GetSignature() => Signature;
    public uint GetSize() => (uint)(_entries.Count * 8);

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br, (uint)inData.Length);
    }

    public void Load(BinaryReader br)
    {
        // This variant is for interface compliance. It's unsafe; prefer the size-aware version.
        Load(br, (uint)(br.BaseStream.Length - br.BaseStream.Position));
    }

    public void Load(BinaryReader br, uint chunkSize)
    {
        var endOffset = br.BaseStream.Position + chunkSize;
        while (br.BaseStream.Position + 8 <= endOffset)
        {
            uint msurIdx = br.ReadUInt32();
            uint mdosIdx = br.ReadUInt32();
            _entries.Add(new Entry(msurIdx, mdosIdx));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
