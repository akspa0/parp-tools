using System.Numerics;
using Silk.NET.OpenGL;

namespace WowViewer.Core.Renderer.Terrain;

public sealed class TerrainMesh : IDisposable
{
    public int TileX { get; init; }
    public int TileY { get; init; }

    public uint Vao { get; init; }
    public uint VboVertices { get; init; }
    public uint VboChunkSlice { get; init; }
    public uint VboTexIndices { get; init; }
    public uint Ebo { get; init; }
    public uint IndexCount { get; init; }

    public int ChunkCount { get; init; }
    public Vector3 BoundsMin { get; init; }
    public Vector3 BoundsMax { get; init; }

    public uint AlphaShadowArrayTexture { get; set; }
    public uint DiffuseArrayTexture { get; set; }
    public int DiffuseLayerCount { get; set; }

    public List<string> TexturePaths { get; init; } = new();

    private GL? _gl;

    public void SetGl(GL gl) => _gl = gl;

    public void Dispose()
    {
        if (_gl == null)
            return;
        if (Vao != 0) _gl.DeleteVertexArray(Vao);
        if (VboVertices != 0) _gl.DeleteBuffer(VboVertices);
        if (VboChunkSlice != 0) _gl.DeleteBuffer(VboChunkSlice);
        if (VboTexIndices != 0) _gl.DeleteBuffer(VboTexIndices);
        if (Ebo != 0) _gl.DeleteBuffer(Ebo);
        if (AlphaShadowArrayTexture != 0) _gl.DeleteTexture(AlphaShadowArrayTexture);
        if (DiffuseArrayTexture != 0) _gl.DeleteTexture(DiffuseArrayTexture);
    }
}
