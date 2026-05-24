using System.Numerics;

namespace WowViewer.Core.Renderer.Wmo;

public sealed record WmoBatchCall(int FirstIndex, ushort IndexCount, int MaterialId, bool IsTransparent);

public struct WmoGroupMeshHandle
{
    public uint Vao;
    public uint Vbo;
    public uint Ebo;
}
