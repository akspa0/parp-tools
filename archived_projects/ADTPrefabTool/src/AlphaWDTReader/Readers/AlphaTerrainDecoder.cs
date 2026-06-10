using System.Runtime.InteropServices;
using System.Text;
using AlphaWDTReader.Model;

namespace AlphaWDTReader.Readers;

public static class AlphaTerrainDecoder
{
    // Alpha-era MCVT commonly stores 145 float32 heights (9x9 + 8x8 pattern)
    private const int VertexCount = 145;
    private const int McvtBytes = VertexCount * 4; // 580

    public static float[]? ReadHeights(string filePath, AlphaChunkIndex chunk)
    {
        if (chunk.OfsMCVT == 0 || chunk.SizeMCVT < McvtBytes) return null;
        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        fs.Position = chunk.OfsMCVT;
        var data = br.ReadBytes(McvtBytes);
        if (data.Length != McvtBytes) return null;

        var heights = new float[VertexCount];
        // Convert little-endian bytes to floats
        for (int i = 0; i < VertexCount; i++)
        {
            heights[i] = BitConverter.ToSingle(data, i * 4);
        }
        return heights;
    }

    // MCNR: 145 * 3 signed bytes (nx,ny,nz). Normalize to [-1,1].
    public static float[]? ReadNormals(string filePath, AlphaChunkIndex chunk)
    {
        const int components = 3;
        int requiredBytes = VertexCount * components; // 435
        if (chunk.OfsMCNR == 0 || chunk.SizeMCNR < requiredBytes) return null;

        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        fs.Position = chunk.OfsMCNR;
        var data = br.ReadBytes(requiredBytes);
        if (data.Length != requiredBytes) return null;

        var normals = new float[VertexCount * components];
        for (int i = 0; i < VertexCount; i++)
        {
            int baseIdx = i * components;
            normals[baseIdx + 0] = (sbyte)data[baseIdx + 0] / 127f;
            normals[baseIdx + 1] = (sbyte)data[baseIdx + 1] / 127f;
            normals[baseIdx + 2] = (sbyte)data[baseIdx + 2] / 127f;
        }
        return normals;
    }
}
