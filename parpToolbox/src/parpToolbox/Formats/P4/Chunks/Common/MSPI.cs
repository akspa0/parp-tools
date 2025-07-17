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
    private readonly List<int> _indices = new();
    public IReadOnlyList<(int A, int B, int C)> Triangles => _triangles;
    public IReadOnlyList<int> Indices => _indices;

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

        // Allow non-multiple-of-6|12 sizes (may indicate triangle strip). Only reject odd byte counts.
        if (bytes % (use32 ? 4 : 2) != 0)
            throw new InvalidDataException("MSPI data length not divisible by index width");

        int indexCount = (int)(bytes / (use32 ? 4 : 2));
        _indices.Clear();
        _indices.Capacity = indexCount;
        for (int i = 0; i < indexCount; i++)
        {
            int val = use32 ? (int)br.ReadUInt32() : br.ReadUInt16();
            _indices.Add(val);
        }
        var indices = _indices;
        // Determine if data is triangle list or strip
        if (indices.Count % 3 == 0)
        {
            // Triangle list
            _triangles.Clear();
            for (int i = 0; i < indices.Count; i += 3)
                _triangles.Add((indices[i], indices[i + 1], indices[i + 2]));
        }
        else if (indices.Count >= 3)
        {
            // Triangle strip (alternating winding)
            bool flip = false;
            for (int i = 2; i < indices.Count; i++)
            {
                int a = indices[i - 2];
                int b = indices[i - 1];
                int c = indices[i];
                if (flip)
                {
                    (b, c) = (c, b);
                }
                _triangles.Add((a, b, c));
                flip = !flip;
            }
        }
        // else: too few indices -> leave _triangles empty
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
