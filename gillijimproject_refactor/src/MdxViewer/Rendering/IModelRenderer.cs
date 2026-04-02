using System.Numerics;

namespace MdxViewer.Rendering;

public interface IModelRenderer : ISceneRenderer
{
    Vector3 BoundsMin { get; }

    Vector3 BoundsMax { get; }

    bool RequiresUnbatchedWorldRender { get; }

    MdxAnimator? Animator { get; }

    void UpdateAnimation();

    void ApplyTextureSamplingSettings();

    void BeginBatch(
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor);

    void RenderInstance(Matrix4x4 modelMatrix, RenderPass pass, float fadeAlpha = 1.0f);

    void RenderWithTransform(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        RenderPass pass = RenderPass.Both,
        float fadeAlpha = 1.0f,
        Vector3? fogColor = null,
        float fogStart = 200f,
        float fogEnd = 1500f,
        Vector3? cameraPos = null,
        Vector3? lightDir = null,
        Vector3? lightColor = null,
        Vector3? ambientColor = null);

    void RenderBackdrop(
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

    void RenderWireframeOverlay(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3? fogColor = null,
        float fogStart = 200f,
        float fogEnd = 1500f,
        Vector3? cameraPos = null,
        Vector3? lightDir = null,
        Vector3? lightColor = null,
        Vector3? ambientColor = null);
}