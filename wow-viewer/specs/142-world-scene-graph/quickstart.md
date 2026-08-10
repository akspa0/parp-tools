# Spec 142 Graph and Opt-In Traversal Quickstart

This slice proves the scene-graph, synthetic-workload, object-adapter, and opt-in traversal
contracts. It does not launch the viewer, a GPU capture, a real-client load, training, or a long
benchmark. The legacy viewer traversal remains the default.

## Focused proof

From `I:\parp\parp-tools\wow-viewer`:

```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneGraphTests|FullyQualifiedName~SyntheticWorldWorkloadTests|FullyQualifiedName~WorldSceneGraphObjectAdapterTests"
```

The graph/traversal foundation can be checked together with:

```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneGraphTests|FullyQualifiedName~SyntheticWorldWorkloadTests|FullyQualifiedName~WorldSceneTraversalTests|FullyQualifiedName~WorldSceneGraphObjectAdapterTests"
```

The viewer integration seam can be compile-checked without launching a client or capture:

```powershell
dotnet build src/viewer/WoWViewer/WoWViewer.csproj -c Debug
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
- Existing `WorldObjectInstance` placements adapt into stable tile/external graph buckets.
- Client-backed WMO group summaries mount beneath their placement node as nested `WmoGroup` nodes;
  missing or malformed group bounds remain fail-open.
- Unknown object bounds keep their bucket and map fail-open.
- The viewer project compiles with the opt-in `WorldScene.UseHierarchicalSceneTraversal` seam.

## Not yet run by this phase

The four-scale performance ladder, current-vs-new traversal comparison, portal/pass/query parity,
real-client capture, GPU timing, and whole-map residency work remain later user-run gates. Do not
interpret this test/build pass as an FPS result.
