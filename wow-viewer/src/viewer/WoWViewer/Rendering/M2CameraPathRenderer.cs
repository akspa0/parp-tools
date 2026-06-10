using System.Numerics;
using WoWViewer.Terrain;
using Silk.NET.OpenGL;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

public sealed class M2CameraPathRenderer : IModelRenderer
{
    private static readonly Vector3[] CameraPalette =
    [
        new(0.97f, 0.72f, 0.22f),
        new(0.24f, 0.82f, 0.93f),
        new(0.93f, 0.40f, 0.30f),
        new(0.56f, 0.83f, 0.34f),
    ];

    private readonly BoundingBoxRenderer _lineRenderer;
    private readonly IReadOnlyList<M2CameraPathOverlay> _cameraPaths;
    private readonly List<bool> _cameraVisibility = new();
    private bool _wireframe;

    public M2CameraPathRenderer(GL gl, M2CameraPathVisualization visualization, string sourceModelPath)
    {
        ArgumentNullException.ThrowIfNull(gl);
        ArgumentNullException.ThrowIfNull(visualization);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        SourceModelPath = sourceModelPath.Replace('/', '\\');
        _lineRenderer = new BoundingBoxRenderer(gl);
        _cameraPaths = visualization.Overlays;

        for (int index = 0; index < _cameraPaths.Count; index++)
            _cameraVisibility.Add(true);

        if (_cameraPaths.Count == 0)
        {
            BoundsMin = new Vector3(-1.0f, -1.0f, -1.0f);
            BoundsMax = new Vector3(1.0f, 1.0f, 1.0f);
            return;
        }

        BoundsMin = visualization.BoundsMin;
        BoundsMax = visualization.BoundsMax;
    }

    public string SourceModelPath { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }

    public bool HasTransparentWorldPass => false;

    public bool RequiresUnbatchedWorldRender => true;

    public IAnimationController? Animator => null;

    public int SubObjectCount => _cameraPaths.Count;

    public void Render(Matrix4x4 view, Matrix4x4 proj)
        => RenderWithTransform(Matrix4x4.Identity, view, proj, RenderPass.Opaque);

    public void ToggleWireframe()
        => _wireframe = !_wireframe;

    public void UpdateAnimation()
    {
    }

    public void ApplyTextureSamplingSettings()
    {
    }

    public void BeginBatch(
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor)
    {
    }

    public void RenderInstance(Matrix4x4 modelMatrix, RenderPass pass, float fadeAlpha = 1.0f)
    {
    }

    public void RenderWithTransform(
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
        Vector3? ambientColor = null)
    {
        if (pass == RenderPass.Transparent || _cameraPaths.Count == 0)
            return;

        _lineRenderer.BeginBatch();
        for (int index = 0; index < _cameraPaths.Count; index++)
        {
            if (!_cameraVisibility[index])
                continue;

            AppendOverlayGeometry(modelMatrix, _cameraPaths[index]);
        }

        _lineRenderer.FlushBatch(view, proj);
    }

    public void RenderBackdrop(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor)
    {
    }

    public void RenderWireframeOverlay(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3? fogColor = null,
        float fogStart = 200f,
        float fogEnd = 1500f,
        Vector3? cameraPos = null,
        Vector3? lightDir = null,
        Vector3? lightColor = null,
        Vector3? ambientColor = null)
    {
        bool previousWireframe = _wireframe;
        _wireframe = true;
        try
        {
            RenderWithTransform(modelMatrix, view, proj, RenderPass.Opaque, 1.0f, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
        }
        finally
        {
            _wireframe = previousWireframe;
        }
    }

    public string GetSubObjectName(int index)
        => index >= 0 && index < _cameraPaths.Count
            ? _cameraPaths[index].Name
            : $"Camera {index}";

    public bool GetSubObjectVisible(int index)
        => index >= 0 && index < _cameraVisibility.Count && _cameraVisibility[index];

    public void SetSubObjectVisible(int index, bool visible)
    {
        if (index < 0 || index >= _cameraVisibility.Count)
            return;

        _cameraVisibility[index] = visible;
    }

    public void Dispose()
        => _lineRenderer.Dispose();

    private void AppendOverlayGeometry(Matrix4x4 modelMatrix, M2CameraPathOverlay overlay)
    {
        Vector3 paletteColor = CameraPalette[Math.Abs(overlay.CameraIndex) % CameraPalette.Length];
        Vector3 cameraColor = _wireframe ? Vector3.One : paletteColor;
        Vector3 targetColor = _wireframe ? new Vector3(0.82f, 0.82f, 0.82f) : Vector3.Lerp(paletteColor, new Vector3(0.92f, 0.96f, 1.0f), 0.45f);
        Vector3 connectorColor = _wireframe ? new Vector3(0.70f, 0.70f, 0.70f) : Vector3.Lerp(paletteColor, Vector3.One, 0.2f);

        AppendPolyline(modelMatrix, overlay.CameraSamples, cameraColor);
        AppendPolyline(modelMatrix, overlay.TargetSamples, targetColor);

        int connectorCount = Math.Min(overlay.CameraSamples.Count, overlay.TargetSamples.Count);
        int connectorStride = Math.Max(1, connectorCount / 12);
        for (int index = 0; index < connectorCount; index += connectorStride)
        {
            _lineRenderer.BatchLine(
                TransformPoint(overlay.CameraSamples[index], modelMatrix),
                TransformPoint(overlay.TargetSamples[index], modelMatrix),
                connectorColor);
        }

        Vector3 cameraStart = TransformPoint(overlay.CameraSamples[0], modelMatrix);
        Vector3 cameraEnd = TransformPoint(overlay.CameraSamples[^1], modelMatrix);
        Vector3 targetStart = TransformPoint(overlay.TargetSamples[0], modelMatrix);
        Vector3 targetEnd = TransformPoint(overlay.TargetSamples[^1], modelMatrix);

        _lineRenderer.BatchPin(cameraStart, overlay.PinHeight, overlay.PinHeadSize, cameraColor);
        _lineRenderer.BatchPin(targetStart, overlay.PinHeight * 0.75f, overlay.PinHeadSize * 0.75f, targetColor);

        if (!ApproximatelyEqual(overlay.CameraSamples[0], overlay.CameraSamples[^1]))
            _lineRenderer.BatchPin(cameraEnd, overlay.PinHeight, overlay.PinHeadSize, connectorColor);

        if (!ApproximatelyEqual(overlay.TargetSamples[0], overlay.TargetSamples[^1]))
            _lineRenderer.BatchPin(targetEnd, overlay.PinHeight * 0.75f, overlay.PinHeadSize * 0.75f, connectorColor);
    }

    private void AppendPolyline(Matrix4x4 modelMatrix, IReadOnlyList<Vector3> points, Vector3 color)
    {
        if (points.Count == 1)
        {
            Vector3 point = TransformPoint(points[0], modelMatrix);
            _lineRenderer.BatchLine(point, point + new Vector3(0.001f, 0.0f, 0.0f), color);
            return;
        }

        for (int index = 1; index < points.Count; index++)
        {
            _lineRenderer.BatchLine(
                TransformPoint(points[index - 1], modelMatrix),
                TransformPoint(points[index], modelMatrix),
                color);
        }
    }

    private static bool ApproximatelyEqual(Vector3 left, Vector3 right)
        => Vector3.DistanceSquared(left, right) <= 0.0001f;

    private static Vector3 TransformPoint(Vector3 point, Matrix4x4 transform)
        => Vector3.Transform(point, transform);
}
