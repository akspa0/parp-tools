namespace ParpToolbox.Formats.PM4.Chunks;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Triangle list chunk (MSPI). Stores 16-bit vertex indices grouped by 3.
/// Some historical files store 32-bit; this loader autodetects by length.
/// </summary>
internal sealed class MspiChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSPI";

    private readonly List<(int A,int B,int C)> _triangles = new();
    public IReadOnlyList<(int A,int B,int C)> Triangles => _triangles;

    public string GetSignature() => Signature;
    public uint GetSize() => (uint)(_triangles.Count * 6); // assumes 16-bit indices

    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        long bytes = br.BaseStream.Length - br.BaseStream.Position;
        bool use32 = bytes % 12 == 0; // 4*3 ==12
        if (!use32 && bytes % 6 != 0)
            throw new InvalidDataException("MSPI size not divisible by 6 or 12 bytes – unknown index width.");

        int triCount = (int)(bytes / (use32 ? 12 : 6));
        for (int i = 0; i < triCount; i++)
        {
            int a,b,c;
            if (use32)
            {
                a = (int)br.ReadUInt32();
                b = (int)br.ReadUInt32();
                c = (int)br.ReadUInt32();
            }
            else
            {
                a = br.ReadUInt16();
                b = br.ReadUInt16();
                c = br.ReadUInt16();
            }
            _triangles.Add((a,b,c));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
