namespace WowViewer.Core.Runtime.World.Visibility;

/// <summary>
/// Which rule let a WMO placement through to submission, or which rule rejected it.
/// </summary>
public enum WmoPlacementAdmissionRule
{
    /// <summary>Placement was admitted to the visible set.</summary>
    Admitted = 0,

    /// <summary>Hidden by the uniqueId hide filter before any distance work.</summary>
    RejectedHiddenByUniqueId,

    /// <summary>Outside the frustum and outside the forward vision cone.</summary>
    RejectedOffFrustumAndCone,

    /// <summary>Bounds distance exceeded the cone-scaled WMO cull distance.</summary>
    RejectedDistance,

    /// <summary>Center distance exceeded the absolute world-object view distance.</summary>
    RejectedMaxViewDistance,

    /// <summary>Projected screen height fell below the visibility profile threshold.</summary>
    RejectedProjectedSize,

    /// <summary>The asset is not resident yet, so the placement was queued instead of drawn.</summary>
    RejectedAssetNotReady,
}

/// <summary>
/// Which rule admitted a WMO group to submission inside an already-admitted placement.
/// </summary>
/// <remarks>
/// Spec 151 instrumentation. The Stormwind capture measured 7512 admitted groups and 80484 draw
/// calls with no counter saying which rule let them through; naming the rule is a precondition for
/// changing any admission logic.
/// </remarks>
public enum WmoGroupAdmissionRule
{
    /// <summary>Group was not admitted by any rule.</summary>
    None = 0,

    /// <summary>Runtime group visibility is switched off, so every group is admitted unconditionally.</summary>
    RuntimeVisibilityDisabled,

    /// <summary>The placement transform could not be inverted, so admission failed open.</summary>
    PlacementTransformInvalid,

    /// <summary>The portal evaluator returned its conservative fallback, which admits every group.</summary>
    PortalFallback,

    /// <summary>Portal traversal admitted the group and the frustum union did not.</summary>
    Portal,

    /// <summary>The post-portal frustum union admitted the group and portal traversal did not.</summary>
    Frustum,

    /// <summary>Portal traversal and the frustum union both admitted the group.</summary>
    PortalAndFrustum,

    /// <summary>
    /// The GPU-instanced opaque shell path, which never consults runtime group visibility and
    /// submits every manually visible group once per instance.
    /// </summary>
    GpuInstancedShell,
}

/// <summary>
/// Allocation-free per-frame accumulator for WMO admission accounting. Callers record one rule per
/// placement and one rule per group, then merge tallies with <see cref="Add"/>.
/// </summary>
public struct WmoAdmissionTally
{
    // Placement layer — which WMO instances entered the visible set.
    public int PlacementsConsidered;
    public int PlacementsAdmitted;
    public int PlacementsRejectedHidden;
    public int PlacementsRejectedOffFrustumAndCone;
    public int PlacementsRejectedDistance;
    public int PlacementsRejectedMaxViewDistance;
    public int PlacementsRejectedProjectedSize;
    public int PlacementsRejectedAssetNotReady;

    // Group layer — which groups inside admitted placements were submitted, and on whose authority.
    public int GroupPlacementEvaluations;
    public int GroupsConsidered;
    public int GroupsAdmitted;
    public int GroupsRejected;
    public int AdmittedByRuntimeVisibilityDisabled;
    public int AdmittedByPlacementTransformInvalid;
    public int AdmittedByPortalFallback;
    public int AdmittedByPortal;
    public int AdmittedByFrustum;
    public int AdmittedByPortalAndFrustum;
    public int AdmittedByGpuInstancedShell;
    public int PortalFallbackEvaluations;

    /// <summary>Largest admitted-group count contributed by a single placement evaluation.</summary>
    public int MaxGroupsAdmittedInOnePlacement;

    /// <summary>Model key of the placement that produced <see cref="MaxGroupsAdmittedInOnePlacement"/>.</summary>
    public string? WorstPlacementModelKey;

    /// <summary>First conservative-fallback reason seen this frame, as reported by the portal evaluator.</summary>
    public string? FirstPortalFallbackReason;

    public void RecordPlacement(WmoPlacementAdmissionRule rule)
    {
        PlacementsConsidered++;
        switch (rule)
        {
            case WmoPlacementAdmissionRule.Admitted:
                PlacementsAdmitted++;
                break;
            case WmoPlacementAdmissionRule.RejectedHiddenByUniqueId:
                PlacementsRejectedHidden++;
                break;
            case WmoPlacementAdmissionRule.RejectedOffFrustumAndCone:
                PlacementsRejectedOffFrustumAndCone++;
                break;
            case WmoPlacementAdmissionRule.RejectedDistance:
                PlacementsRejectedDistance++;
                break;
            case WmoPlacementAdmissionRule.RejectedMaxViewDistance:
                PlacementsRejectedMaxViewDistance++;
                break;
            case WmoPlacementAdmissionRule.RejectedProjectedSize:
                PlacementsRejectedProjectedSize++;
                break;
            case WmoPlacementAdmissionRule.RejectedAssetNotReady:
                PlacementsRejectedAssetNotReady++;
                break;
        }
    }

    public void RecordGroup(WmoGroupAdmissionRule rule)
    {
        GroupsConsidered++;
        switch (rule)
        {
            case WmoGroupAdmissionRule.None:
                GroupsRejected++;
                return;
            case WmoGroupAdmissionRule.RuntimeVisibilityDisabled:
                AdmittedByRuntimeVisibilityDisabled++;
                break;
            case WmoGroupAdmissionRule.PlacementTransformInvalid:
                AdmittedByPlacementTransformInvalid++;
                break;
            case WmoGroupAdmissionRule.PortalFallback:
                AdmittedByPortalFallback++;
                break;
            case WmoGroupAdmissionRule.Portal:
                AdmittedByPortal++;
                break;
            case WmoGroupAdmissionRule.Frustum:
                AdmittedByFrustum++;
                break;
            case WmoGroupAdmissionRule.PortalAndFrustum:
                AdmittedByPortalAndFrustum++;
                break;
            case WmoGroupAdmissionRule.GpuInstancedShell:
                AdmittedByGpuInstancedShell++;
                break;
        }

        GroupsAdmitted++;
    }

    /// <summary>
    /// Closes one placement's group evaluation. <paramref name="admittedInPlacement"/> is that
    /// placement's own admitted count, which is what identifies a single dominant offender.
    /// </summary>
    public void RecordGroupPlacementEvaluation(int admittedInPlacement, string? modelKey, string? portalFallbackReason)
    {
        GroupPlacementEvaluations++;
        if (portalFallbackReason is not null)
        {
            PortalFallbackEvaluations++;
            FirstPortalFallbackReason ??= portalFallbackReason;
        }

        if (admittedInPlacement > MaxGroupsAdmittedInOnePlacement)
        {
            MaxGroupsAdmittedInOnePlacement = admittedInPlacement;
            WorstPlacementModelKey = modelKey;
        }
    }

    public void Add(in WmoAdmissionTally other)
    {
        PlacementsConsidered += other.PlacementsConsidered;
        PlacementsAdmitted += other.PlacementsAdmitted;
        PlacementsRejectedHidden += other.PlacementsRejectedHidden;
        PlacementsRejectedOffFrustumAndCone += other.PlacementsRejectedOffFrustumAndCone;
        PlacementsRejectedDistance += other.PlacementsRejectedDistance;
        PlacementsRejectedMaxViewDistance += other.PlacementsRejectedMaxViewDistance;
        PlacementsRejectedProjectedSize += other.PlacementsRejectedProjectedSize;
        PlacementsRejectedAssetNotReady += other.PlacementsRejectedAssetNotReady;

        GroupPlacementEvaluations += other.GroupPlacementEvaluations;
        GroupsConsidered += other.GroupsConsidered;
        GroupsAdmitted += other.GroupsAdmitted;
        GroupsRejected += other.GroupsRejected;
        AdmittedByRuntimeVisibilityDisabled += other.AdmittedByRuntimeVisibilityDisabled;
        AdmittedByPlacementTransformInvalid += other.AdmittedByPlacementTransformInvalid;
        AdmittedByPortalFallback += other.AdmittedByPortalFallback;
        AdmittedByPortal += other.AdmittedByPortal;
        AdmittedByFrustum += other.AdmittedByFrustum;
        AdmittedByPortalAndFrustum += other.AdmittedByPortalAndFrustum;
        AdmittedByGpuInstancedShell += other.AdmittedByGpuInstancedShell;
        PortalFallbackEvaluations += other.PortalFallbackEvaluations;

        if (other.MaxGroupsAdmittedInOnePlacement > MaxGroupsAdmittedInOnePlacement)
        {
            MaxGroupsAdmittedInOnePlacement = other.MaxGroupsAdmittedInOnePlacement;
            WorstPlacementModelKey = other.WorstPlacementModelKey;
        }

        FirstPortalFallbackReason ??= other.FirstPortalFallbackReason;
    }

    public void Reset() => this = default;

    public readonly WmoAdmissionStats ToStats() => new()
    {
        PlacementsConsidered = PlacementsConsidered,
        PlacementsAdmitted = PlacementsAdmitted,
        PlacementsRejectedHidden = PlacementsRejectedHidden,
        PlacementsRejectedOffFrustumAndCone = PlacementsRejectedOffFrustumAndCone,
        PlacementsRejectedDistance = PlacementsRejectedDistance,
        PlacementsRejectedMaxViewDistance = PlacementsRejectedMaxViewDistance,
        PlacementsRejectedProjectedSize = PlacementsRejectedProjectedSize,
        PlacementsRejectedAssetNotReady = PlacementsRejectedAssetNotReady,
        GroupPlacementEvaluations = GroupPlacementEvaluations,
        GroupsConsidered = GroupsConsidered,
        GroupsAdmitted = GroupsAdmitted,
        GroupsRejected = GroupsRejected,
        AdmittedByRuntimeVisibilityDisabled = AdmittedByRuntimeVisibilityDisabled,
        AdmittedByPlacementTransformInvalid = AdmittedByPlacementTransformInvalid,
        AdmittedByPortalFallback = AdmittedByPortalFallback,
        AdmittedByPortal = AdmittedByPortal,
        AdmittedByFrustum = AdmittedByFrustum,
        AdmittedByPortalAndFrustum = AdmittedByPortalAndFrustum,
        AdmittedByGpuInstancedShell = AdmittedByGpuInstancedShell,
        PortalFallbackEvaluations = PortalFallbackEvaluations,
        MaxGroupsAdmittedInOnePlacement = MaxGroupsAdmittedInOnePlacement,
        WorstPlacementModelKey = WorstPlacementModelKey,
        FirstPortalFallbackReason = FirstPortalFallbackReason,
    };
}

/// <summary>
/// One frame's WMO admission accounting, in two layers: which placements entered the visible set,
/// and which groups inside those placements were submitted and on whose authority.
/// </summary>
public readonly record struct WmoAdmissionStats
{
    public int PlacementsConsidered { get; init; }
    public int PlacementsAdmitted { get; init; }
    public int PlacementsRejectedHidden { get; init; }
    public int PlacementsRejectedOffFrustumAndCone { get; init; }
    public int PlacementsRejectedDistance { get; init; }
    public int PlacementsRejectedMaxViewDistance { get; init; }
    public int PlacementsRejectedProjectedSize { get; init; }
    public int PlacementsRejectedAssetNotReady { get; init; }

    public int GroupPlacementEvaluations { get; init; }
    public int GroupsConsidered { get; init; }
    public int GroupsAdmitted { get; init; }
    public int GroupsRejected { get; init; }
    public int AdmittedByRuntimeVisibilityDisabled { get; init; }
    public int AdmittedByPlacementTransformInvalid { get; init; }
    public int AdmittedByPortalFallback { get; init; }
    public int AdmittedByPortal { get; init; }
    public int AdmittedByFrustum { get; init; }
    public int AdmittedByPortalAndFrustum { get; init; }
    public int AdmittedByGpuInstancedShell { get; init; }
    public int PortalFallbackEvaluations { get; init; }
    public int MaxGroupsAdmittedInOnePlacement { get; init; }
    public string? WorstPlacementModelKey { get; init; }
    public string? FirstPortalFallbackReason { get; init; }

    public static WmoAdmissionStats Empty { get; } = default;

    /// <summary>
    /// The rule that admitted the most groups this frame, which is the one worth acting on.
    /// Returns <see cref="WmoGroupAdmissionRule.None"/> when nothing was admitted.
    /// </summary>
    public WmoGroupAdmissionRule DominantGroupAdmissionRule
    {
        get
        {
            WmoGroupAdmissionRule rule = WmoGroupAdmissionRule.None;
            int best = 0;
            Consider(WmoGroupAdmissionRule.RuntimeVisibilityDisabled, AdmittedByRuntimeVisibilityDisabled);
            Consider(WmoGroupAdmissionRule.PlacementTransformInvalid, AdmittedByPlacementTransformInvalid);
            Consider(WmoGroupAdmissionRule.PortalFallback, AdmittedByPortalFallback);
            Consider(WmoGroupAdmissionRule.Portal, AdmittedByPortal);
            Consider(WmoGroupAdmissionRule.Frustum, AdmittedByFrustum);
            Consider(WmoGroupAdmissionRule.PortalAndFrustum, AdmittedByPortalAndFrustum);
            Consider(WmoGroupAdmissionRule.GpuInstancedShell, AdmittedByGpuInstancedShell);
            return rule;

            void Consider(WmoGroupAdmissionRule candidate, int count)
            {
                if (count <= best)
                    return;

                best = count;
                rule = candidate;
            }
        }
    }

    /// <summary>
    /// Mean groups admitted per placement evaluation. Separates "one enormous WMO" from
    /// "many ordinary WMOs" without needing a per-placement dump.
    /// </summary>
    public double MeanGroupsAdmittedPerPlacement
        => GroupPlacementEvaluations == 0 ? 0d : (double)GroupsAdmitted / GroupPlacementEvaluations;
}
