namespace ParpToolbox.Formats.PM4.Chunks;

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
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        long bytesRemaining = br.BaseStream.Length - br.BaseStream.Position;
        if (bytesRemaining % 12 != 0 && bytesRemaining % 24 != 0)
            throw new InvalidDataException("MSPV length not divisible by 12 or 24 bytes – unknown stride.");

        int stride = bytesRemaining % 24 == 0 ? 24 : 12;
        int count = (int)(bytesRemaining / stride);
        for (int i = 0; i < count; i++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            if (stride == 24)
            {
                br.ReadSingle(); // skip unknown
                br.ReadSingle();
                br.ReadSingle();
            }
            _vertices.Add(new Vector3(x, y, z));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
