using System.Numerics;

namespace WoWViewer.Rendering;

public readonly record struct WmoOpaqueDoodadBatchItem(
    IModelRenderer Renderer,
    Matrix4x4 ModelMatrix);
