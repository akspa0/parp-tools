using System.Numerics;
using WowViewer.Core.Runtime.World.Visibility;

namespace WoWViewer.Rendering;

/// <summary>
/// GPU instancing contract for opaque WMO shell placements.
/// Implementations must keep transparent, portal-sensitive, liquid, and doodad paths separate.
/// </summary>
public interface IGpuInstancedWmoRenderer
{
    bool SupportsGpuInstancedOpaque { get; }

    WmoRenderStats LastRenderStats { get; }

    /// <summary>Group admission accounting for the most recent submission. See Spec 151.</summary>
    WmoAdmissionTally LastGroupAdmission { get; }

    void BeginGpuInstanceBatch(
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor);

    void QueueGpuInstance(Matrix4x4 modelMatrix);

    void EndGpuInstanceBatch();

    void RenderOpaqueDoodadsForPlacement(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor);

    void CollectOpaqueDoodadsForPlacement(
        Matrix4x4 modelMatrix,
        Vector3 cameraPos,
        float fogEnd,
        Action<WmoOpaqueDoodadBatchItem> collect);
}
