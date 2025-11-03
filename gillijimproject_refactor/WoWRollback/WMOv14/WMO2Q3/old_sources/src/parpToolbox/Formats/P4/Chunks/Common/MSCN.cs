namespace ParpToolbox.Formats.P4.Chunks.Common;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

/// <summary>
/// MSCN – Collision / exterior vertex list. Each entry is 12 bytes (Vector3 float XYZ).
/// </summary>
internal sealed class MscnChunk : IIffChunk, IBinarySerializable
{
    public const string Signature = "MSCN";

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
        var endOffset = br.BaseStream.Position + chunkSize;
        const int stride = 12;
        while (br.BaseStream.Position + stride <= endOffset)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            _vertices.Add(new Vector3(x, y, z));
        }
    }

    public byte[] Serialize(long offset = 0) => throw new NotSupportedException();
}
