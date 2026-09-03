using System.Numerics;

namespace WoWViewer.Rendering;

public interface IModelRenderer : ISceneRenderer
{
    Vector3 BoundsMin { get; }

    Vector3 BoundsMax { get; }

    /// <summary>
    /// Tight, geometry-derived model-space bounds for selection highlighting and ray picking.
    /// </summary>
    /// <remarks>
    /// Deliberately separate from <see cref="BoundsMin"/>/<see cref="BoundsMax"/>, which stay
    /// conservative because culling depends on them. M2 models report a declared header extent that
    /// is an animation/collision volume, routinely far larger than the mesh; drawing or picking
    /// against that produces a selection box that does not describe the object it selects.
    /// Implementations without tighter information fall back to the culling bounds.
    /// </remarks>
    Vector3 SelectionBoundsMin => BoundsMin;

    Vector3 SelectionBoundsMax => BoundsMax;

    bool HasTransparentWorldPass { get; }

    bool RequiresUnbatchedWorldRender { get; }

    IAnimationController? Animator { get; }

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
