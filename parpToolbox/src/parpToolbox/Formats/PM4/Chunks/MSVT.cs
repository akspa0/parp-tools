namespace ParpToolbox.Formats.PM4.Chunks;

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
        Load(br);
    }

    public void Load(BinaryReader br)
    {
        long bytesRemaining = br.BaseStream.Length - br.BaseStream.Position;
        if (bytesRemaining % 12 != 0 && bytesRemaining % 24 != 0)
            throw new InvalidDataException("MSVT length not divisible by 12 or 24 bytes – unknown stride.");

        int stride = bytesRemaining % 24 == 0 ? 24 : 12;
        int count = (int)(bytesRemaining / stride);
        for (int i = 0; i < count; i++)
        {
            float y = br.ReadSingle();
            float x = br.ReadSingle();
            float z = br.ReadSingle();
            if (stride == 24)
            {
                br.ReadSingle(); br.ReadSingle(); br.ReadSingle(); // skip unknown floats
            }
            _vertices.Add(new Vector3(x, y, z)); // reorder -> (X,Y,Z)
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
