using System.IO;

namespace AlphaWDTReader.ReferencePort;

public static class McnkReaders
{
    private const int VertexCount = 145;

    // MCVT: 145 float32 heights
    public static float[]? ReadMcvt145(BinaryReader br)
    {
        int bytes = VertexCount * 4; // 580
        var data = br.ReadBytes(bytes);
        if (data.Length != bytes) return null;
        var heights = new float[VertexCount];
        for (int i = 0; i < VertexCount; i++)
            heights[i] = BitConverter.ToSingle(data, i * 4);
        return heights;
    }

    // MCNR: 145 * 3 signed bytes -> floats in [-1,1]
    public static float[]? ReadMcnr145(BinaryReader br)
    {
        const int components = 3;
        int bytes = VertexCount * components; // 435
        var data = br.ReadBytes(bytes);
        if (data.Length != bytes) return null;
        var normals = new float[bytes];
        for (int i = 0; i < VertexCount; i++)
        {
            int b = i * components;
            normals[b + 0] = (sbyte)data[b + 0] / 127f;
            normals[b + 1] = (sbyte)data[b + 1] / 127f;
            normals[b + 2] = (sbyte)data[b + 2] / 127f;
        }
        return normals;
    }
}
