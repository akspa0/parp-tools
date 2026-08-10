using System.Numerics;

namespace WowViewer.Core.Runtime.World.SceneGraph;

public sealed record WorldSceneGraphSnapshot(
    int NodeCount,
    int RenderableCount,
    int QueryableCount,
    int UpdateRequiredCount,
    int NonRejectableCount,
    int MaxDepth,
    IReadOnlyDictionary<WorldSceneNodeKind, int> NodeKindCounts,
    IReadOnlyDictionary<WorldSceneRenderPass, int> RenderPassCounts,
    IReadOnlyList<string> NodeIds,
    Vector3 RootBoundsMin,
    Vector3 RootBoundsMax);
