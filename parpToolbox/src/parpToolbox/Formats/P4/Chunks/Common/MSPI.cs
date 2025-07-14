namespace ParpToolbox.Formats.P4.Chunks.Common;

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

    private readonly List<(int A, int B, int C)> _triangles = new();
    public IReadOnlyList<(int A, int B, int C)> Triangles => _triangles;

    public string GetSignature() => Signature;

    public uint GetSize() => (uint)(_triangles.Count * 6); // assume 16-bit for size estimation

    /// <summary>
    /// Parses triangle index data using vertexCount to decide 16- vs 32-bit width.
    /// </summary>
    /// <summary>
    /// Interface implementation that delegates to <see cref="Load(BinaryReader,int)"/> with a
    /// vertexCount of 0, falling back to size heuristics.
    /// </summary>
    public void LoadBinaryData(byte[] inData)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br, vertexCount: 0);
    }

    /// <summary>
    /// Parses triangle index data using vertexCount to decide 16- vs 32-bit width.
    /// </summary>
    public void LoadBinaryData(byte[] inData, int vertexCount)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br, vertexCount);
    }

    public void Load(BinaryReader br) => throw new NotSupportedException("Call the overload with vertexCount");

    public void Load(BinaryReader br, int vertexCount)
    {
        long bytes = br.BaseStream.Length - br.BaseStream.Position;

        // Heuristic: if vertexCount > 65535 it must be 32-bit
        bool use32 = vertexCount > ushort.MaxValue;

        if (!use32)
        {
            // If data length cannot be evenly divided by 6, fallback to 32-bit.
            if (bytes % 6 != 0 && bytes % 12 == 0)
                use32 = true;
        }

        if (use32 && bytes % 12 != 0)
            throw new InvalidDataException("MSPI size not valid for 32-bit indices (multiple of 12 bytes)");
        if (!use32 && bytes % 6 != 0)
            throw new InvalidDataException("MSPI size not valid for 16-bit indices (multiple of 6 bytes)");

        int triCount = (int)(bytes / (use32 ? 12 : 6));
        _triangles.Capacity = triCount;

        for (int i = 0; i < triCount; i++)
        {
            int a, b, c;
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
            _triangles.Add((a, b, c));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
