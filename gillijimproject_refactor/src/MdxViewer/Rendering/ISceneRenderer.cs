using System.Numerics;

namespace MdxViewer.Rendering;

/// <summary>
/// Common interface for all renderers (MDX, WMO, etc.)
/// </summary>
public interface ISceneRenderer : IDisposable
{
    void Render(Matrix4x4 view, Matrix4x4 proj);
    void ToggleWireframe();

    /// <summary>Number of renderable sub-objects (geosets/groups).</summary>
    int SubObjectCount { get; }

    /// <summary>Get display name for a sub-object.</summary>
    string GetSubObjectName(int index);

    /// <summary>Get/set visibility of a sub-object.</summary>
    bool GetSubObjectVisible(int index);
    void SetSubObjectVisible(int index, bool visible);
}
