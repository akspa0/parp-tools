namespace ParpToolbox.Formats.P4.Chunks.Common;

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
            uint idx = br.ReadUInt32();
            uint state = br.ReadUInt32();
            uint? nameId = null;
            ushort? doodadSet = null;
            uint? flags = null;
            float? scale = null;

            if (br.BaseStream.Position + 4 <= endOffset) { nameId = br.ReadUInt32(); }
            if (br.BaseStream.Position + 2 <= endOffset) { doodadSet = br.ReadUInt16(); }
            if (br.BaseStream.Position + 4 <= endOffset) { flags = br.ReadUInt32(); }
            if (br.BaseStream.Position + 4 <= endOffset) { scale = br.ReadSingle(); }

            // Seek to the end of the known entry or chunk to handle unknown extra bytes gracefully
            if (br.BaseStream.Position < endOffset)
            {
                br.BaseStream.Seek(endOffset - br.BaseStream.Position, SeekOrigin.Current);
            }

            _entries.Add(new Entry(idx, state, nameId, doodadSet, flags, scale));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
