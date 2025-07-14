namespace ParpToolbox.Formats.PM4.Chunks;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

/// <summary>
/// MSUR – Surface metadata chunk. Each entry is 64 bytes as documented in msur_surface_reference.md.
/// This loader parses the confirmed fields and stores them in <see cref="Entry"/> records.
/// Unknown / padding bytes are kept for diagnostic purposes.
/// </summary>
internal sealed class MsurChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSUR";

    public sealed record Entry(
        byte Flags,
        byte SurfaceGroupKey,
        bool IsM2Bucket,
        byte IndexCount,
        ushort SurfaceAttributeMask,
        bool IsLiquidCandidate,
        Vector3 RawNormal,
        float RawPlaneD,
        Vector3 NormalizedNormal,
        float SurfaceHeight,
        uint MsviFirstIndex,
        int MdosIndex,
        uint SurfaceKey,
        uint PackedParams);

    private readonly List<Entry> _entries = new();
    public IReadOnlyList<Entry> Entries => _entries;

    public string GetSignature() => Signature;
    public uint GetSize() => (uint)(_entries.Count * 64);

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        while (br.BaseStream.Position + 64 <= br.BaseStream.Length)
        {
            byte flags = br.ReadByte();
            byte groupKey = br.ReadByte();
            bool isM2 = (flags & 0x10) != 0;
            byte indexCount = br.ReadByte();
            br.ReadByte(); // Unknown_0x02 padding
            ushort attrMask = br.ReadUInt16();
            bool isLiquid = (attrMask & 0x000C) != 0;
            float rawNx = br.ReadSingle();
            float rawNy = br.ReadSingle();
            float rawNz = br.ReadSingle();
            float rawD = br.ReadSingle();
            Vector3 rawNormal = new(rawNx, rawNy, rawNz);
            // Normalised normal follows
            float normX = br.ReadSingle();
            float normY = br.ReadSingle();
            float normZ = br.ReadSingle();
            float surfaceHeight = br.ReadSingle();
            Vector3 normal = new(normX, normY, normZ);
            uint firstIndex = br.ReadUInt32();
            int mdosIndex = br.ReadInt32();
            uint surfaceKey = br.ReadUInt32();
            uint packedParams = br.ReadUInt32();

            _entries.Add(new Entry(flags, groupKey, isM2, indexCount, attrMask, isLiquid, rawNormal, rawD, normal, surfaceHeight, firstIndex, mdosIndex, surfaceKey, packedParams));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
