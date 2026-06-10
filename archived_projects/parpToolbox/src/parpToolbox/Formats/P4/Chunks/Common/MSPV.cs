namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

/// <summary>
/// Navigation vertex buffer (MSPV) – holds raw world-space coordinates.
/// Stride may be either 12 bytes (XYZ) or 24 bytes (XYZ + 3 unknown floats).
/// </summary>
internal sealed class MspvChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSPV";

    private readonly List<Vector3> _vertices = new();
    public IReadOnlyList<Vector3> Vertices => _vertices;

    public string GetSignature() => Signature;

    public uint GetSize() => (uint)(_vertices.Count * 12); // stored size without padding

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
            throw new InvalidDataException("MSPV length not divisible by 12 bytes, the required stride for a Vector3.");

        int count = (int)(bytesRemaining / 12);
        _vertices.Clear();
        _vertices.Capacity = count;
        for (int i = 0; i < count; i++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            _vertices.Add(new Vector3(x, y, z));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
