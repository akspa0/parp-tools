using System.Numerics;

namespace WowViewer.Core.Runtime.World.Selection;

/// <summary>Kind of a pickable scene object (Spec 228 data model; Epic 251 U-01 E4).</summary>
public enum WorldSceneSelectionKind
{
    Wmo,
    Mdx,
    WmoDoodad,
}

/// <summary>What the selection evaluation is for.</summary>
public enum WorldSceneSelectionMode
{
    /// <summary>Click pick: every hit, ranked for selection and disambiguation.</summary>
    Click,

    /// <summary>Hover along the mouse ray: the single nearest visible scene object.</summary>
    HoverRay,
}

/// <summary>Outcome of one evaluation. <see cref="InvalidRequest"/> never carries a selection.</summary>
public enum WorldSceneSelectionStatus
{
    Hit,
    NoHit,
    InvalidRequest,
}

/// <summary>
/// Hover range policy: when <see cref="LimitRange"/> is set, only hits within
/// <see cref="MaxDistance"/> of the ray origin (or of <see cref="CameraPosition"/> for screen-space
/// hits) are eligible. With the limit off every hit is eligible, including non-finite distances.
/// </summary>
public readonly record struct WorldSceneSelectionPolicy(bool LimitRange, float MaxDistance, Vector3 CameraPosition)
{
    public static WorldSceneSelectionPolicy Unlimited { get; } = new(false, float.MaxValue, Vector3.Zero);

    /// <summary>A limited policy needs a non-negative, non-NaN range.</summary>
    public bool IsValid => !LimitRange || (!float.IsNaN(MaxDistance) && MaxDistance >= 0f);

    /// <summary>Screen-space checks also need a finite camera position when the range is limited.</summary>
    public bool IsValidForPositions => IsValid
        && (!LimitRange || (float.IsFinite(CameraPosition.X) && float.IsFinite(CameraPosition.Y) && float.IsFinite(CameraPosition.Z)));

    public bool IsDistanceAllowed(float distance) => !LimitRange || distance <= MaxDistance;

    public bool IsPositionAllowed(Vector3 worldPosition)
        => !LimitRange || Vector3.Distance(CameraPosition, worldPosition) <= MaxDistance;
}

/// <summary>One ray-hit scene object. Values and a snapshot-unique identifier only.</summary>
public readonly record struct WorldSceneSelectionCandidate(
    int Id,
    WorldSceneSelectionKind Kind,
    float RayDistance,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 SelectionPoint,
    float SelectionPointDistanceSq,
    bool SharesClickedChunk,
    int ChunkGridDistance);

/// <summary>A ray hit on a non-object target (liquid body): identifier and ray distance.</summary>
public readonly record struct WorldSceneRayTarget(int Id, float RayDistance);

/// <summary>A screen-space ("brush") hover measurement for one target.</summary>
public readonly record struct WorldSceneBrushCandidate(int Id, float ScreenDistanceSq, float Depth, Vector3 WorldPosition);

/// <summary>Inputs for one click or hover-ray evaluation.</summary>
public readonly record struct WorldSceneSelectionRequest(
    WorldSceneSelectionMode Mode,
    WorldSceneSelectionPolicy Policy,
    bool HasClickedChunk = false,
    bool WmosVisible = true,
    bool DoodadsVisible = true)
{
    public bool IsValid => Enum.IsDefined(Mode) && Policy.IsValid;
}

/// <summary>Immutable candidate list for one evaluation.</summary>
public sealed class WorldSceneSelectionSnapshot
{
    public WorldSceneSelectionSnapshot(IReadOnlyList<WorldSceneSelectionCandidate> candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        Candidates = candidates.ToArray();
        var ids = new HashSet<int>();
        IsValid = Candidates.All(c => c.Id >= 0 && ids.Add(c.Id));
    }

    public IReadOnlyList<WorldSceneSelectionCandidate> Candidates { get; }

    /// <summary>
    /// Identifiers must be non-negative and unique within a snapshot; a snapshot that breaks this is
    /// rejected rather than evaluated.
    /// </summary>
    public bool IsValid { get; }
}

/// <summary>
/// Result of a click or hover-ray evaluation. <see cref="RankedIds"/> is the full ordered list
/// (click) or the eligible hits nearest-first (hover); <see cref="BestId"/> is its first entry.
/// </summary>
public sealed record WorldSceneSelectionResult(WorldSceneSelectionStatus Status, IReadOnlyList<int> RankedIds)
{
    public static WorldSceneSelectionResult Invalid { get; } = new(WorldSceneSelectionStatus.InvalidRequest, []);

    public int BestId => RankedIds.Count > 0 ? RankedIds[0] : -1;
}

/// <summary>Which target a hover-ray evaluation ended on.</summary>
public enum WorldSceneHoverRayTarget
{
    None,
    SceneObject,
    LiquidBody,
}

/// <summary>
/// Hover along a ray after liquid bodies had their chance to be nearer. <see cref="Id"/> is a scene
/// candidate id or a liquid body id depending on <see cref="Target"/>.
/// </summary>
public readonly record struct WorldSceneHoverRayResult(WorldSceneSelectionStatus Status, WorldSceneHoverRayTarget Target, int Id, float Distance);

/// <summary>Best screen-space hover target and how many eligible targets were under the brush.</summary>
public readonly record struct WorldSceneBrushResult(
    WorldSceneSelectionStatus Status,
    int BestId,
    int EligibleCount,
    float BestScreenDistanceSq,
    float BestDepth);

/// <summary>Which hover source wins when the scene and the PM4 overlay both have a hit.</summary>
public enum WorldSceneHoverSource
{
    None,
    Scene,
    Pm4,
}
