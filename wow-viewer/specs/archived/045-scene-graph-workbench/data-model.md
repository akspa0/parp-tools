# Data Model: Scene Graph Workbench

## SceneGraphSnapshot

- **Purpose**: Immutable read-only hierarchy representing the currently loaded scene at one moment.
- **Fields**:
  - `RootNode` - the single root node for the active scene
  - `GeneratedAtUtc` - snapshot timestamp
  - `SceneKind` - e.g. world session, standalone asset
  - `DomainsPresent` - terrain/object/PM4 availability summary
  - `NodeCount` - total materialized node count
- **Relationships**:
  - owns a recursive tree of `SceneGraphNode`

## SceneGraphNode

- **Purpose**: One hierarchical node in the outliner.
- **Fields**:
  - `NodeId` - stable identity path
  - `Kind` - typed node category
  - `Label` - user-facing display label
  - `Summary` - compact status/count text
  - `Children` - child node list or lazy child descriptor
  - `IsSelectable` - whether the node maps to a live selection target
  - `SelectionTarget` - optional live scene selection mapping
  - `Metadata` - diagnostic key/value fields
  - `Availability` - loaded, unloaded, unavailable, deferred
- **Relationships**:
  - belongs to one parent node except the root
  - may reference one `SceneGraphSelectionTarget`

## SceneGraphNodeId

- **Purpose**: Stable typed path for a node.
- **Fields**:
  - `Segments` - ordered path segments such as `scene/world`, `terrain/tile:30_48`, `pm4/region:3262`
  - `CanonicalString` - normalized serialized form for lookup/filter/debugging
- **Rules**:
  - must be stable across refreshes when the underlying scene element is still the same logical element
  - must not depend on transient object references

## SceneGraphSelectionTarget

- **Purpose**: Bridge from a node to the viewer's existing live selection systems.
- **Fields**:
  - `TargetKind` - terrain chunk, placed object, PM4 object, PM4 sub-object, etc.
  - `TargetIdentity` - existing domain-specific lookup key
  - `CanFrameCamera` - whether camera focusing is supported
  - `CanHighlight` - whether viewport highlighting is supported

## SceneGraphDomainProvider

- **Purpose**: Domain-specific projector that contributes nodes to a scene graph snapshot.
- **Fields**:
  - `DomainKind` - terrain, placed objects, PM4
  - `DisplayName` - root label for the domain
  - `Availability` - whether the domain is available in the current scene
- **Rules**:
  - projects existing decoded/runtime data only
  - does not own live UI state

## ViewerSceneGraphFilterState

- **Purpose**: User-driven filtering/search state for the workbench.
- **Fields**:
  - `Query` - text filter
  - `KindFilters` - optional node-kind narrowing
  - `ShowUnavailable` - whether unavailable nodes remain visible
  - `AutoExpandMatches` - whether matching ancestors are expanded automatically

## NodeKind

- **Purpose**: Shared node classification vocabulary.
- **Representative values**:
  - `SceneRoot`
  - `TerrainRoot`
  - `Map`
  - `Tile`
  - `Chunk`
  - `ChunkLayer`
  - `Liquid`
  - `ObjectRoot`
  - `ObjectTile`
  - `WmoInstance`
  - `M2Instance`
  - `Pm4Root`
  - `Pm4Region`
  - `Pm4Object`
  - `Pm4SubObject`
  - `Pm4Surface`
  - `Pm4Anchor`

## Relationships Summary

- `SceneGraphSnapshot` owns exactly one `SceneGraphNode` root.
- `SceneGraphNode` can have zero or more child `SceneGraphNode` records.
- `SceneGraphNode` can optionally reference one `SceneGraphSelectionTarget`.
- `SceneGraphDomainProvider` contributes one or more domain root nodes beneath the scene root.
- `ViewerSceneGraphFilterState` does not mutate the underlying snapshot; it only changes visibility/presentation of nodes.
