namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Unified index buffer (MSVI) – flat list of vertex indices (16- or 32-bit).
/// Consumers should read <see cref="Indices"/> sequentially; every 3 indices form a triangle.
/// </summary>
internal sealed class MsviChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSVI";

    private readonly List<int> _indices = new();
    public IReadOnlyList<int> Indices => _indices;

    public string GetSignature() => Signature;
    public uint GetSize() => (uint)(_indices.Count * 2); // minimal representation

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
        long bytes = chunkSize;
        if (bytes % 4 != 0)
            throw new InvalidDataException("MSVI data length not divisible by 4, which is required for 32-bit indices.");

        int indexCount = (int)(bytes / 4);
        _indices.Clear();
        _indices.Capacity = indexCount;
        for (int i = 0; i < indexCount; i++)
        {
            _indices.Add((int)br.ReadUInt32());
        }
    }

    public IEnumerable<(int A,int B,int C)> Triangulate()
    {
        for (int i = 0; i + 2 < _indices.Count; i += 3)
            yield return (_indices[i], _indices[i + 1], _indices[i + 2]);
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
