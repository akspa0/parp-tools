namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Per-frame scene-light workload (Epic 249 R-10a): how many emitted lights were collected and kept,
/// how many light queries ran and how many candidate lights they tested, and how the frame's opaque
/// WMO placements split between the instanced batch and the lit per-placement path.
/// </summary>
/// <param name="LightsCollected">Lights emitted by visible WMOs and loaded MDX/M2 this frame.</param>
/// <param name="LightsKept">Lights kept after validation and the view-frustum cull.</param>
/// <param name="QueryCount">Light queries issued (terrain, WMO, doodad and batch decisions).</param>
/// <param name="CandidatesTested">Candidate lights those queries tested exactly.</param>
/// <param name="WmoPlacementsBatched">Opaque WMO placements submitted through GPU instancing.</param>
/// <param name="WmoPlacementsLitFallback">Instancing-capable placements a scene light reaches.</param>
/// <param name="WmoPlacementsSelfLit">Of those, placements whose own model emits MOLT lights.</param>
public readonly record struct SceneLightingFrameStats(
    int LightsCollected,
    int LightsKept,
    int QueryCount,
    long CandidatesTested,
    int WmoPlacementsBatched,
    int WmoPlacementsLitFallback,
    int WmoPlacementsSelfLit);
