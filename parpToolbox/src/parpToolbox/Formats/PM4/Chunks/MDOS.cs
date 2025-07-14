namespace ParpToolbox.Formats.PM4.Chunks;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// MDOS – Destructible object states / material definitions.
/// Layout in the wild varies; at minimum first two uint32 are confirmed.
/// Additional optional fields are read only if present.
/// </summary>
internal sealed class MdosChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MDOS";

    public sealed record Entry(uint BuildingIndex, uint DestructionState, uint? NameId, ushort? DoodadSet, uint? Flags, float? Scale);

    private readonly List<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries;

    public string GetSignature() => Signature;
    public uint GetSize() => 0; // variable, serialize unused

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            uint idx = br.ReadUInt32();
            uint state = br.ReadUInt32();
            uint? nameId = null;
            ushort? doodadSet = null;
            uint? flags = null;
            float? scale = null;

            long remaining = br.BaseStream.Length - br.BaseStream.Position;
            if (remaining >= 4) { nameId = br.ReadUInt32(); remaining -= 4; }
            if (remaining >= 2) { doodadSet = br.ReadUInt16(); remaining -= 2; }
            if (remaining >= 4) { flags = br.ReadUInt32(); remaining -= 4; }
            if (remaining >= 4) { scale = br.ReadSingle(); remaining -= 4; }
            // ignore any extra unknown bytes for now
            if (remaining > 0) br.BaseStream.Seek(remaining, SeekOrigin.Current);

            _entries.Add(new Entry(idx, state, nameId, doodadSet, flags, scale));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
