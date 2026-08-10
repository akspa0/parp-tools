using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Core.Runtime.World.SceneGraph;

public sealed record SyntheticWorldWorkloadDefinition
{
    public string FixtureName { get; init; } = "synthetic-world-v1";
    public int Seed { get; init; } = 142;
    public int ResidentRegionCount { get; init; } = 1;
    public int ChunksPerRegion { get; init; } = 16;
    public int WmoPlacements { get; init; } = 1;
    public int WmoGroupsPerPlacement { get; init; } = 4;
    public int M2Placements { get; init; } = 16;
    public int RepeatedAssetCount { get; init; } = 4;
    public int Pm4OverlayCount { get; init; } = 1;
    public int PortalLinkCount { get; init; } = 2;
    public SyntheticRenderPassMix RenderPassMix { get; init; } = new();
    public SyntheticCamera Camera { get; init; } = SyntheticCamera.Default;

    internal void Validate()
    {
        if (string.IsNullOrWhiteSpace(FixtureName))
            throw new ArgumentException("Fixture name must not be empty.", nameof(FixtureName));
        if (ResidentRegionCount < 1 || ChunksPerRegion < 1)
            throw new ArgumentOutOfRangeException(nameof(ResidentRegionCount), "Region and chunk counts must be positive.");
        if (WmoPlacements < 0 || WmoGroupsPerPlacement < 0 || M2Placements < 0
            || RepeatedAssetCount < 0 || Pm4OverlayCount < 0 || PortalLinkCount < 0)
            throw new ArgumentOutOfRangeException(nameof(WmoPlacements), "Synthetic workload counts must not be negative.");
        if (M2Placements > 0 && RepeatedAssetCount == 0)
            throw new ArgumentException("RepeatedAssetCount must be positive when M2 placements exist.", nameof(RepeatedAssetCount));

        RenderPassMix.Validate();
        Camera.Validate();
    }
}

public sealed record SyntheticRenderPassMix(
    int Opaque = 1,
    int AlphaTested = 1,
    int Transparent = 1,
    int Liquid = 1,
    int Overlay = 1)
{
    internal void Validate()
    {
        if (Opaque < 0 || AlphaTested < 0 || Transparent < 0 || Liquid < 0 || Overlay < 0)
            throw new ArgumentOutOfRangeException(nameof(Opaque), "Render-pass counts must not be negative.");
        if (Opaque + AlphaTested + Transparent + Liquid + Overlay == 0)
            throw new ArgumentException("At least one render pass must be declared.");
    }
}

public sealed record SyntheticCamera(
    float[] Position,
    float YawDegrees,
    float PitchDegrees,
    float VerticalFovDegrees,
    float NearPlane,
    float FarPlane)
{
    public static SyntheticCamera Default { get; } = new([0f, 0f, 64f], 0f, -20f, 60f, 1f, 8192f);

    internal void Validate()
    {
        if (Position is null || Position.Length != 3 || Position.Any(value => !float.IsFinite(value)))
            throw new ArgumentException("Camera position must contain three finite values.", nameof(Position));
        if (!float.IsFinite(YawDegrees) || !float.IsFinite(PitchDegrees) || !float.IsFinite(VerticalFovDegrees)
            || !float.IsFinite(NearPlane) || !float.IsFinite(FarPlane)
            || VerticalFovDegrees <= 0f || NearPlane <= 0f || FarPlane <= NearPlane)
            throw new ArgumentException("Camera angles and clip planes are invalid.", nameof(SyntheticCamera));
    }
}

public sealed record SyntheticWorldNodeDescriptor(
    string Id,
    string? ParentId,
    WorldSceneNodeKind Kind,
    float[] LocalTransform,
    float[] LocalBoundsMin,
    float[] LocalBoundsMax,
    bool BoundsKnown,
    bool CanRejectSubtree,
    bool IsRenderable,
    bool IsQueryable,
    bool RequiresUpdate,
    string? AssetKey,
    WorldSceneRenderPass RenderPassMask,
    int? PortalGroup);

public sealed record SyntheticPortalLink(string SourceNodeId, string DestinationNodeId);

public sealed record SyntheticWorldWorkload(
    string Schema,
    string WorkloadClass,
    string FixtureName,
    int Seed,
    int ResidentRegionCount,
    int ChunksPerRegion,
    int WmoPlacements,
    int WmoGroupsPerPlacement,
    int M2Placements,
    int RepeatedAssetCount,
    int Pm4OverlayCount,
    int PortalLinkCount,
    SyntheticRenderPassMix RenderPassMix,
    SyntheticCamera Camera,
    IReadOnlyList<SyntheticWorldNodeDescriptor> Nodes,
    IReadOnlyList<SyntheticPortalLink> PortalLinks,
    string ManifestSha256)
{
    public const string CurrentSchema = "v1-synthetic-world-workload";
    public const string CurrentWorkloadClass = "synthetic_world_scene";

    public string ToJson(bool indented = true)
    {
        Validate();
        return JsonSerializer.Serialize(this, CreateJsonOptions(indented));
    }

    public string ComputeManifestSha256()
    {
        string canonicalJson = JsonSerializer.Serialize(this with { ManifestSha256 = string.Empty }, CreateJsonOptions(false));
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(canonicalJson));
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    public SyntheticWorldWorkload WithComputedManifestHash()
    {
        SyntheticWorldWorkload withoutHash = this with { ManifestSha256 = string.Empty };
        return withoutHash with { ManifestSha256 = withoutHash.ComputeManifestSha256() };
    }

    public static SyntheticWorldWorkload FromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            throw new ArgumentException("Synthetic workload JSON must not be empty.", nameof(json));

        SyntheticWorldWorkload workload = JsonSerializer.Deserialize<SyntheticWorldWorkload>(json, CreateJsonOptions(false))
            ?? throw new InvalidOperationException("Synthetic workload JSON did not produce a manifest.");
        workload.Validate();

        string expectedHash = workload.ComputeManifestSha256();
        if (!string.Equals(expectedHash, workload.ManifestSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Synthetic workload manifest hash does not match its contents.");

        return workload;
    }

    internal void Validate()
    {
        if (!string.Equals(Schema, CurrentSchema, StringComparison.Ordinal))
            throw new InvalidOperationException($"Unsupported synthetic workload schema '{Schema}'.");
        if (!string.Equals(WorkloadClass, CurrentWorkloadClass, StringComparison.Ordinal))
            throw new InvalidOperationException($"Unsupported workload class '{WorkloadClass}'.");
        if (string.IsNullOrWhiteSpace(FixtureName))
            throw new InvalidOperationException("Synthetic workload fixture name must not be empty.");
        if (ResidentRegionCount < 1 || ChunksPerRegion < 1)
            throw new InvalidOperationException("Synthetic workload region and chunk counts must be positive.");
        if (WmoPlacements < 0 || WmoGroupsPerPlacement < 0 || M2Placements < 0
            || RepeatedAssetCount < 0 || Pm4OverlayCount < 0 || PortalLinkCount < 0)
            throw new InvalidOperationException("Synthetic workload counts must not be negative.");
        if (M2Placements > 0 && RepeatedAssetCount == 0)
            throw new InvalidOperationException("Repeated asset count must be positive when M2 placements exist.");

        RenderPassMix.Validate();
        Camera.Validate();
        if (Nodes is null || Nodes.Count == 0)
            throw new InvalidOperationException("Synthetic workload must contain at least a root node.");
        if (PortalLinks is null)
            throw new InvalidOperationException("Synthetic workload portal links must not be null.");

        HashSet<string> ids = new(StringComparer.Ordinal);
        string? rootId = null;
        foreach (SyntheticWorldNodeDescriptor node in Nodes)
        {
            if (string.IsNullOrWhiteSpace(node.Id) || !ids.Add(node.Id))
                throw new InvalidOperationException($"Synthetic workload contains duplicate or empty node id '{node.Id}'.");
            if (node.ParentId is null)
            {
                if (rootId is not null)
                    throw new InvalidOperationException("Synthetic workload must contain exactly one root node.");
                rootId = node.Id;
            }
            else if (!ids.Contains(node.ParentId))
            {
                throw new InvalidOperationException($"Synthetic workload node '{node.Id}' references a parent that is not earlier in the manifest.");
            }

            ValidateArray(node.LocalTransform, 16, $"{node.Id}.LocalTransform");
            ValidateArray(node.LocalBoundsMin, 3, $"{node.Id}.LocalBoundsMin");
            ValidateArray(node.LocalBoundsMax, 3, $"{node.Id}.LocalBoundsMax");
            if (node.BoundsKnown && (node.LocalBoundsMin[0] > node.LocalBoundsMax[0]
                || node.LocalBoundsMin[1] > node.LocalBoundsMax[1]
                || node.LocalBoundsMin[2] > node.LocalBoundsMax[2]))
                throw new InvalidOperationException($"Synthetic workload node '{node.Id}' has unordered bounds.");
        }

        if (rootId is null)
            throw new InvalidOperationException("Synthetic workload must contain one root node.");
        foreach (SyntheticPortalLink link in PortalLinks)
        {
            if (!ids.Contains(link.SourceNodeId) || !ids.Contains(link.DestinationNodeId))
                throw new InvalidOperationException("Synthetic workload portal links must reference declared nodes.");
        }
    }

    private static JsonSerializerOptions CreateJsonOptions(bool indented) => new()
    {
        WriteIndented = indented,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Converters = { new JsonStringEnumConverter() }
    };

    private static void ValidateArray(float[]? values, int expectedLength, string name)
    {
        if (values is null || values.Length != expectedLength || values.Any(value => !float.IsFinite(value)))
            throw new InvalidOperationException($"Synthetic workload field '{name}' must contain {expectedLength} finite values.");
    }
}

public sealed record SyntheticWorldWorkloadBuildResult(
    WorldSceneGraph Graph,
    SyntheticWorldWorkload Manifest);
