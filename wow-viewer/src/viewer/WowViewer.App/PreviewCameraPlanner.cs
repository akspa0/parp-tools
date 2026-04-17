using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.App;

internal enum PreviewCameraMode
{
    Frame,
    Orbit,
    Model,
}

internal sealed class PreviewCameraSettings
{
    public const string DefaultPresetName = "three_quarter";

    public PreviewCameraMode Mode { get; init; } = PreviewCameraMode.Frame;

    public string? PresetName { get; init; } = DefaultPresetName;

    public float AzimuthDegrees { get; init; } = 35.0f;

    public float ElevationDegrees { get; init; } = 25.0f;

    public float FieldOfViewDegrees { get; init; } = 60.0f;

    public float PaddingScale { get; init; } = 1.04f;

    public float ZoomFactor { get; init; } = 0.72f;

    public PreviewCameraSettings Resolve()
    {
        float azimuth = AzimuthDegrees;
        float elevation = ElevationDegrees;
        string? presetName = string.IsNullOrWhiteSpace(PresetName) ? null : PresetName;
        if (Mode == PreviewCameraMode.Orbit
            && !string.IsNullOrWhiteSpace(presetName)
            && PreviewCameraPresets.TryGetOrbit(presetName, out PreviewOrbitPreset preset))
        {
            azimuth = preset.AzimuthDegrees;
            elevation = preset.ElevationDegrees;
        }

        return new PreviewCameraSettings
        {
            Mode = Mode,
            PresetName = presetName,
            AzimuthDegrees = azimuth,
            ElevationDegrees = elevation,
            FieldOfViewDegrees = FieldOfViewDegrees,
            PaddingScale = PaddingScale,
            ZoomFactor = ZoomFactor,
        };
    }

    public void Validate()
    {
        if (FieldOfViewDegrees <= 1.0f || FieldOfViewDegrees >= 170.0f)
            throw new ArgumentOutOfRangeException(nameof(FieldOfViewDegrees), "Camera field of view must stay within 1..170 degrees.");

        if (PaddingScale <= 0.1f || !float.IsFinite(PaddingScale))
            throw new ArgumentOutOfRangeException(nameof(PaddingScale), "Camera padding scale must be finite and greater than 0.1.");

        if (ZoomFactor <= 0.05f || !float.IsFinite(ZoomFactor))
            throw new ArgumentOutOfRangeException(nameof(ZoomFactor), "Camera zoom factor must be finite and greater than 0.05.");
    }
}

internal readonly record struct PreviewOrbitPreset(string Name, float AzimuthDegrees, float ElevationDegrees);

internal readonly record struct PreviewCameraPose(Matrix4x4 View, Matrix4x4 Projection, Vector3 CameraPosition, Vector3 FocusPoint);

internal readonly record struct PreviewInteractiveFrame(Vector3 Position, Vector3 FocusPoint, float YawDegrees, float PitchDegrees);

internal static class PreviewCameraPresets
{
    private static readonly IReadOnlyDictionary<string, PreviewOrbitPreset> OrbitPresets =
        new Dictionary<string, PreviewOrbitPreset>(StringComparer.OrdinalIgnoreCase)
        {
            ["front"] = new("front", 0.0f, 15.0f),
            ["back"] = new("back", 180.0f, 15.0f),
            ["left"] = new("left", 90.0f, 15.0f),
            ["right"] = new("right", 270.0f, 15.0f),
            ["top"] = new("top", 0.0f, 80.0f),
            [PreviewCameraSettings.DefaultPresetName] = new(PreviewCameraSettings.DefaultPresetName, 35.0f, 25.0f),
        };

    public static bool TryGetOrbit(string name, out PreviewOrbitPreset preset)
    {
        if (OrbitPresets.TryGetValue(name, out preset))
            return true;

        preset = default;
        return false;
    }
}

internal static class PreviewCameraPlanner
{
    public static PreviewCameraPose CreatePose(Vector3 min, Vector3 max, PreviewCameraSettings settings, MdxSummary? summary, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(settings);

        PreviewCameraSettings resolved = settings.Resolve();
        resolved.Validate();

        if (resolved.Mode == PreviewCameraMode.Model
            && TryCreateModelPose(min, max, resolved, summary, width, height, out PreviewCameraPose modelPose))
        {
            return modelPose;
        }

        if (resolved.Mode == PreviewCameraMode.Orbit)
            return CreateOrbitPose(min, max, resolved, summary, width, height);

        return CreateLegacyFramePose(min, max, resolved, width, height, mdxMirrorX: false);
    }

    public static PreviewInteractiveFrame CreateLegacyFrameBounds(Vector3 min, Vector3 max, bool mdxMirrorX)
    {
        Vector3 center = (min + max) * 0.5f;
        Vector3 extent = max - min;
        float radius = MathF.Max(extent.Length() * 0.5f, 1.0f);

        if (mdxMirrorX)
            center.X = -center.X;

        float distance = MathF.Max(radius * 3.0f, 2.0f);
        Vector3 position = center + new Vector3(-distance, 0.0f, radius * 0.6f);
        return new PreviewInteractiveFrame(position, center, 0.0f, -15.0f);
    }

    private static PreviewCameraPose CreateLegacyFramePose(Vector3 min, Vector3 max, PreviewCameraSettings settings, int width, int height, bool mdxMirrorX)
    {
        PreviewInteractiveFrame frame = CreateLegacyFrameBounds(min, max, mdxMirrorX);
        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float fov = settings.FieldOfViewDegrees * MathF.PI / 180.0f;
        Vector3 forward = GetForwardVector(frame.YawDegrees, frame.PitchDegrees);
        Vector3 target = frame.Position + forward;
        float diameter = MathF.Max((max - min).Length(), 1.0f);
        float farClip = MathF.Max(Vector3.Distance(frame.Position, frame.FocusPoint) * 10.0f, diameter * 8.0f);
        Matrix4x4 view = Matrix4x4.CreateLookAt(frame.Position, target, Vector3.UnitZ);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(fov, aspect, 0.01f, farClip);
        return new PreviewCameraPose(view, projection, frame.Position, frame.FocusPoint);
    }

    private static PreviewCameraPose CreateOrbitPose(Vector3 min, Vector3 max, PreviewCameraSettings settings, MdxSummary? summary, int width, int height)
    {
        Vector3 center = (min + max) * 0.5f;
        Vector3 focusPoint = center;
        if (TryGetPreferredFocusPoint(summary, min, max, out Vector3 preferredFocus))
            focusPoint = preferredFocus;

        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float fov = settings.FieldOfViewDegrees * MathF.PI / 180.0f;
        Vector3 cameraDirection = GetOrbitDirection(settings.AzimuthDegrees, settings.ElevationDegrees);

        Vector3[] corners =
        [
            new(min.X, min.Y, min.Z),
            new(max.X, min.Y, min.Z),
            new(min.X, max.Y, min.Z),
            new(max.X, max.Y, min.Z),
            new(min.X, min.Y, max.Z),
            new(max.X, min.Y, max.Z),
            new(min.X, max.Y, max.Z),
            new(max.X, max.Y, max.Z),
        ];

        float radius = MathF.Max((max - min).Length() * 0.5f, 0.25f);
        Vector3 temporaryCamera = focusPoint - cameraDirection * radius;
        Vector3 up = MathF.Abs(Vector3.Dot(cameraDirection, Vector3.UnitZ)) > 0.99f ? Vector3.UnitX : Vector3.UnitZ;
        Matrix4x4 temporaryView = Matrix4x4.CreateLookAt(temporaryCamera, focusPoint, up);

        float halfFovV = fov * 0.5f;
        float halfFovH = MathF.Atan(MathF.Tan(halfFovV) * aspect);
        float maxDistance = radius;
        foreach (Vector3 corner in corners)
        {
            Vector3 viewPosition = Vector3.Transform(corner, temporaryView);
            float depth = -viewPosition.Z;
            float needV = MathF.Abs(viewPosition.Y) / MathF.Tan(halfFovV) + depth;
            float needH = MathF.Abs(viewPosition.X) / MathF.Tan(halfFovH) + depth;
            maxDistance = MathF.Max(maxDistance, MathF.Max(needV, needH));
        }

        float distance = MathF.Max(maxDistance * settings.PaddingScale * settings.ZoomFactor, 0.25f);
        Vector3 cameraPosition = focusPoint - cameraDirection * distance;
        float farClip = MathF.Max(distance * 10.0f, (max - min).Length() * 4.0f);
        Matrix4x4 view = Matrix4x4.CreateLookAt(cameraPosition, focusPoint, up);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(fov, aspect, 0.01f, farClip);
        return new PreviewCameraPose(view, projection, cameraPosition, focusPoint);
    }

    private static bool TryCreateModelPose(Vector3 min, Vector3 max, PreviewCameraSettings settings, MdxSummary? summary, int width, int height, out PreviewCameraPose pose)
    {
        pose = default;

        if (summary is null || summary.Cameras.Count == 0)
            return false;

        MdxCameraSummary camera = summary.Cameras
            .FirstOrDefault(static candidate => candidate.Name.Contains("portrait", StringComparison.OrdinalIgnoreCase))
            ?? summary.Cameras[0];

        Vector3 eye = camera.PivotPoint;
        Vector3 target = camera.TargetPivotPoint;
        Vector3 forward = target - eye;
        if (!IsFinite(eye) || !IsFinite(target) || forward.LengthSquared() <= 0.0001f)
            return false;

        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float fov = float.IsFinite(camera.FieldOfView) && camera.FieldOfView > 0.05f && camera.FieldOfView < MathF.PI - 0.05f
            ? camera.FieldOfView
            : settings.FieldOfViewDegrees * MathF.PI / 180.0f;
        float nearClip = float.IsFinite(camera.NearClip) && camera.NearClip > 0.001f ? camera.NearClip : 0.01f;
        float farClip = float.IsFinite(camera.FarClip) && camera.FarClip > nearClip + 0.1f
            ? camera.FarClip
            : MathF.Max(nearClip + 10.0f, (max - min).Length() * 8.0f);

        Vector3 up = MathF.Abs(Vector3.Dot(Vector3.Normalize(forward), Vector3.UnitZ)) > 0.99f ? Vector3.UnitX : Vector3.UnitZ;
        Matrix4x4 view = Matrix4x4.CreateLookAt(eye, target, up);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(fov, aspect, nearClip, farClip);
        pose = new PreviewCameraPose(view, projection, eye, target);
        return true;
    }

    private static Vector3 GetOrbitDirection(float azimuthDegrees, float elevationDegrees)
    {
        float elevationRadians = elevationDegrees * MathF.PI / 180.0f;
        float azimuthRadians = azimuthDegrees * MathF.PI / 180.0f;
        float cosElevation = MathF.Cos(elevationRadians);
        float sinElevation = MathF.Sin(elevationRadians);
        return Vector3.Normalize(new Vector3(
            cosElevation * MathF.Cos(azimuthRadians),
            cosElevation * MathF.Sin(azimuthRadians),
            -sinElevation));
    }

    private static Vector3 GetForwardVector(float yawDegrees, float pitchDegrees)
    {
        float yawRadians = MathF.PI / 180.0f * yawDegrees;
        float pitchRadians = MathF.PI / 180.0f * pitchDegrees;
        float cosPitch = MathF.Cos(pitchRadians);
        return Vector3.Normalize(new Vector3(
            cosPitch * MathF.Cos(yawRadians),
            cosPitch * MathF.Sin(yawRadians),
            MathF.Sin(pitchRadians)));
    }

    private static bool TryGetPreferredFocusPoint(MdxSummary? summary, Vector3 min, Vector3 max, out Vector3 focusPoint)
    {
        focusPoint = default;
        if (summary is null || summary.Cameras.Count == 0)
            return false;

        MdxCameraSummary camera = summary.Cameras
            .FirstOrDefault(static candidate => candidate.Name.Contains("portrait", StringComparison.OrdinalIgnoreCase))
            ?? summary.Cameras[0];

        Vector3 target = camera.TargetPivotPoint;
        if (!IsFinite(target) || !IsPointNearBounds(target, min, max))
            return false;

        focusPoint = target;
        return true;
    }

    private static bool IsPointNearBounds(Vector3 point, Vector3 min, Vector3 max)
    {
        Vector3 padding = Vector3.Max((max - min) * 0.5f, new Vector3(0.25f, 0.25f, 0.25f));
        Vector3 expandedMin = min - padding;
        Vector3 expandedMax = max + padding;
        return point.X >= expandedMin.X && point.X <= expandedMax.X
            && point.Y >= expandedMin.Y && point.Y <= expandedMax.Y
            && point.Z >= expandedMin.Z && point.Z <= expandedMax.Z;
    }

    private static bool IsFinite(Vector3 value) => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}