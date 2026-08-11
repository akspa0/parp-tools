using System.Numerics;

namespace WoWViewer.Rendering;

/// <summary>
/// GPU instancing contract for opaque model placement rendering.
/// Implementations retain a CPU fallback for unsupported passes or material states.
/// </summary>
public interface IGpuInstancedModelRenderer
{
    bool SupportsGpuInstancedOpaque { get; }

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

    void QueueGpuInstance(Matrix4x4 modelMatrix, float fadeAlpha = 1.0f);

    void EndGpuInstanceBatch();
}
