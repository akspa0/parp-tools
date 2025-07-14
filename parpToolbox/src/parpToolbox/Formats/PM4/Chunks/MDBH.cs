namespace ParpToolbox.Formats.PM4.Chunks;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

/// <summary>
/// MDBH – Destructible Building Header. Stores a count followed by (index, null-terminated UTF8 filename) pairs.
/// The filename typically refers to a WMO model or similar asset.
/// </summary>
internal sealed class MdbhChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MDBH";

    public sealed record Entry(uint Index, string Filename);

    private readonly List<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries;

    public string GetSignature() => Signature;
    public uint GetSize() => 0; // variable due to strings

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        if (br.BaseStream.Length - br.BaseStream.Position < 4)
            throw new InvalidDataException("MDBH too small for count field");
        uint count = br.ReadUInt32();
        for (int i = 0; i < count; i++)
        {
            if (br.BaseStream.Position + 4 > br.BaseStream.Length)
                break; // truncated
            uint idx = br.ReadUInt32();
            var sb = new StringBuilder();
            byte b;
            while (br.BaseStream.Position < br.BaseStream.Length && (b = br.ReadByte()) != 0)
                sb.Append((char)b);
            string filename = sb.ToString();
            _entries.Add(new Entry(idx, filename));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
