# Spec 142 Phase 1 Quickstart

This slice proves the scene-graph and synthetic-workload contract only. It does not launch the
viewer, a GPU capture, a real-client load, training, or a long benchmark.

## Focused proof

From `I:\parp\parp-tools\wow-viewer`:

```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneGraphTests|FullyQualifiedName~SyntheticWorldWorkloadTests"
```

The graph/traversal foundation can be checked together with:

```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneGraphTests|FullyQualifiedName~SyntheticWorldWorkloadTests|FullyQualifiedName~WorldSceneTraversalTests"
```

## Full runtime-library proof

```powershell
dotnet build src/core/WowViewer.Core.Runtime/WowViewer.Core.Runtime.csproj -c Debug
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug
```

## Expected evidence

- Two builds from the same `fixture_name` and `seed` produce the same graph snapshot and node IDs.
- A nested WMO/M2/PM4 fixture has one reachable root, explicit parent ownership, and conservative
  bounds.
- Detaching a tile removes its complete subtree from lookup and enumeration.
- Invalid duplicate IDs, cycles, second-parent attachment, and non-finite rejectable bounds fail
  closed.
- A synthetic minimap/image record is not accepted as a synthetic world workload.
- A rejected synthetic region skips all descendant visibility tests and reports the skipped count.
- A node with unknown bounds is included without being passed to the visibility predicate.

## Not yet run by this phase

The four-scale performance ladder, current-vs-new traversal comparison, real-client parity capture,
GPU timing, and whole-map residency work remain later user-run gates. Do not interpret this test
pass as an FPS result.
