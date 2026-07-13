using System.Numerics;
using System.Text.Json;
using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Tools.ValidationCapture;

internal static class TerrainShadeCaptureManifestWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    public static string Write(
        HeadlessValidationCaptureSession session,
        ValidationCaptureTileRequest request,
        ValidationCaptureCameraFrame cameraFrame)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(request);
        if (request.Variant != ValidationCaptureVariant.TerrainShade)
            throw new ArgumentException("Terrain-shade manifests can only be written for TerrainShade requests.", nameof(request));

        ValidationCaptureVariantPolicy policy = session.VariantPolicies[request.Variant];
        TerrainShadeCaptureManifest manifest = new(
            Schema: "wow-viewer.terrain-shade-capture.v1",
            RendererContract: ValidationTerrainShadeContract.Revision,
            GuidanceOnly: true,
            DeploymentInput: false,
            CanonicalTerrainTarget: "mcvt_vertex_z",
            ClientRoot: Path.GetFullPath(session.ClientRoot),
            MapInput: session.MapInput,
            BuildLabel: session.BuildLabel ?? string.Empty,
            TileName: request.TileName,
            TileX: request.TileX,
            TileY: request.TileY,
            Resolution: session.ScenePolicy.RequestedResolution,
            Camera: new TerrainShadeCameraManifest(
                ToArray(cameraFrame.Eye),
                ToArray(cameraFrame.Target),
                ToArray(cameraFrame.Up),
                cameraFrame.WorldSpanX,
                cameraFrame.WorldSpanY),
            Lighting: new TerrainShadeLightingManifest(
                Source: "fixed_viewer_contract_not_client_light_tables",
                Direction: ToArray(ValidationTerrainShadeContract.LightDirection),
                Color: ToArray(ValidationTerrainShadeContract.LightColor),
                Ambient: ToArray(ValidationTerrainShadeContract.AmbientColor)),
            DisabledPasses: BuildDisabledPasses(policy));

        string manifestPath = Path.ChangeExtension(request.OutputPath, ".json");
        string? directory = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, JsonOptions));
        return manifestPath;
    }

    private static string[] BuildDisabledPasses(ValidationCaptureVariantPolicy policy)
    {
        List<string> disabled = [];
        if (!policy.ShowTerrainLiquids) disabled.Add("terrain_liquids");
        if (!policy.ShowObjects) disabled.Add("objects");
        if (!policy.ShowWmos) disabled.Add("wmos");
        if (!policy.ShowDoodads) disabled.Add("doodads");
        if (!policy.ShowSky) disabled.Add("sky");
        if (!policy.ShowWdl) disabled.Add("wdl");
        if (!policy.ShowWorldLiquids) disabled.Add("world_liquids");
        if (policy.TerrainShadeOnly) disabled.Add("terrain_diffuse_and_alpha_texturing");
        return disabled.ToArray();
    }

    private static float[] ToArray(Vector3 value) => [value.X, value.Y, value.Z];

    private sealed record TerrainShadeCaptureManifest(
        string Schema,
        string RendererContract,
        bool GuidanceOnly,
        bool DeploymentInput,
        string CanonicalTerrainTarget,
        string ClientRoot,
        string MapInput,
        string BuildLabel,
        string TileName,
        int TileX,
        int TileY,
        int Resolution,
        TerrainShadeCameraManifest Camera,
        TerrainShadeLightingManifest Lighting,
        string[] DisabledPasses);

    private sealed record TerrainShadeCameraManifest(
        float[] Eye,
        float[] Target,
        float[] Up,
        float WorldSpanX,
        float WorldSpanY);

    private sealed record TerrainShadeLightingManifest(
        string Source,
        float[] Direction,
        float[] Color,
        float[] Ambient);
}
