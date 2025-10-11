namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

/// <summary>
/// Render vertex buffer (MSVT) – coordinates appear in (Y, X, Z) order in the file.
/// Supports 12-byte XYZ stride or 24-byte stride with 3 unknown floats.
/// </summary>
internal sealed class MsvtChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSVT";

    private readonly List<Vector3> _vertices = new();
    public IReadOnlyList<Vector3> Vertices => _vertices;

    public string GetSignature() => Signature;

    public uint GetSize() => (uint)(_vertices.Count * 12);

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
        long bytesRemaining = chunkSize;
        if (bytesRemaining % 12 != 0)
            throw new InvalidDataException("MSVT length not divisible by 12 bytes, the required stride for a Vector3.");

        int count = (int)(bytesRemaining / 12);
        _vertices.Clear();
        _vertices.Capacity = count;
        for (int i = 0; i < count; i++)
        {
            float y = br.ReadSingle();
            float x = br.ReadSingle();
            float z = br.ReadSingle();
            _vertices.Add(new Vector3(x, y, z)); // reorder -> (X,Y,Z)
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
