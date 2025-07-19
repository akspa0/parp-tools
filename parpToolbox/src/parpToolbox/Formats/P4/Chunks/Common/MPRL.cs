namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

/// <summary>
/// MPRL – Placement list (likely doodad/prop positions). Structure confirmed 24-byte per entry.
/// Field meaning largely unknown but contains a position vector at 0x08.
/// </summary>
public sealed class MprlChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MPRL";

    public sealed record Entry(
        ushort Unknown0,
        short Unknown2,
        ushort Unknown4,
        ushort Unknown6,
        Vector3 Position,
        short Unknown14,
        ushort Unknown16);

    private readonly List<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries;

    public string GetSignature() => Signature;
    public uint GetSize() => (uint)(_entries.Count * 24);

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        while (br.BaseStream.Position + 24 <= br.BaseStream.Length)
        {
            ushort u0 = br.ReadUInt16();
            short s2 = br.ReadInt16();
            ushort u4 = br.ReadUInt16();
            ushort u6 = br.ReadUInt16();
            float px = br.ReadSingle();
            float py = br.ReadSingle();
            float pz = br.ReadSingle();
            short s14 = br.ReadInt16();
            ushort u16 = br.ReadUInt16();
            _entries.Add(new Entry(u0, s2, u4, u6, new Vector3(px, py, pz), s14, u16));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
