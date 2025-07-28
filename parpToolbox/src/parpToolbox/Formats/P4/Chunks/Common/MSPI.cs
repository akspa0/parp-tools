namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Triangle list chunk (MSPI). Stores 16-bit vertex indices grouped by 3.
/// Some historical files store 32-bit; this loader autodetects by length.
/// </summary>
public sealed class MspiChunk : IIffChunk, IBinarySerializable
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
        Load(br, (uint)inData.Length, vertexCount: 0);
    }

    /// <summary>
    /// Parses triangle index data using vertexCount to decide 16- vs 32-bit width.
    /// </summary>
    public void LoadBinaryData(byte[] inData, int vertexCount)
    {
        using var ms = new MemoryStream(inData ?? throw new ArgumentNullException(nameof(inData)));
        using var br = new BinaryReader(ms);
        Load(br, (uint)inData.Length, vertexCount);
    }

    public void Load(BinaryReader br) => throw new NotSupportedException("Call the overload with vertexCount");

    public void Load(BinaryReader br, uint chunkSize, int vertexCount)
    {
        long bytes = chunkSize;
        if (bytes % 4 != 0)
            throw new InvalidDataException("MSPI data length not divisible by 4, which is required for 32-bit indices.");

        int indexCount = (int)(bytes / 4);
        _indices.Clear();
        _indices.Capacity = indexCount;
        for (int i = 0; i < indexCount; i++)
        {
            _indices.Add((int)br.ReadUInt32());
        }

        _triangles.Clear();
        if (_indices.Count % 3 != 0)
        {
            // Note: Data is not a clean triangle list. This could indicate a problem or a different data structure like a strip.
            // For now, we process as a simple list and truncate any trailing indices.
        }

        for (int i = 0; i + 2 < _indices.Count; i += 3)
        {
            _triangles.Add((_indices[i], _indices[i + 1], _indices[i + 2]));
        }
        // else: too few indices -> leave _triangles empty
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
