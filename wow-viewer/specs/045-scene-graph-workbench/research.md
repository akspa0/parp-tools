# Research: Scene Graph Workbench

## Decision 1: Use A Read-Only Snapshot Contract Instead Of Rendering The Live Runtime Object Graph Directly

**Decision**: Build a scene-graph snapshot/projection layer that converts live terrain/object/PM4 state into stable read-only nodes before the UI renders them.

**Rationale**:

- The viewer needs stable node identities for expansion state, filtering, and reverse selection.
- Directly traversing live runtime objects in ImGui rendering code would tightly couple the tree to transient scene residency and make tests brittle.
- A snapshot contract can be unit-tested without a running GL viewer and can later back JSON export or debugging tools.

**Alternatives considered**:

- Direct UI traversal of `WorldScene` state: rejected because it mixes rendering/UI lifetime with scene-domain ownership.
- Separate ad hoc trees for terrain, objects, and PM4: rejected because the user explicitly wants one whole-scene outliner.

## Decision 2: Use Domain Providers For Terrain, Objects, And PM4 Under One Root

**Decision**: Introduce separate domain projectors/providers that each contribute child nodes under one shared scene root.

**Rationale**:

- Terrain, placed objects, and PM4 have different natural hierarchies and different ownership libraries.
- Domain providers keep each hierarchy model local while still producing one shared contract.
- This lets later standalone or asset-scene work reuse the same root contract without copy-pasting a second tree system.

**Alternatives considered**:

- One monolithic projector class in `WoWViewer`: rejected because it would pull PM4 and runtime ownership into shell code.
- PM4-only first implementation: rejected because the user asked for the whole scene graph, not another single-domain tool.

## Decision 3: Host The Workbench In The Dockable Viewer Shell, With A Right-Side Default Placement

**Decision**: The feature should live in the active `WoWViewer` shell as a right-sidebar surface and dockable panel, using the dockspace path restored by spec 044.

**Rationale**:

- The user explicitly asked for the right sidebar.
- Spec 044 already re-established docking as the intended shell owner.
- A dockable panel keeps the feature usable even if the long-term shell layout evolves beyond fixed sidebars.

**Alternatives considered**:

- Floating utility window only: rejected because it weakens discoverability and shell integration.
- Legacy fixed sidebar only: rejected because spec 044 moved the shell back toward docking.

## Decision 4: Keep The First Slice Read-Only, Selection-Capable, And Filterable

**Decision**: The first signed-off slice should support browse, expand, select, focus, and filter; editing and reparenting are explicitly deferred.

**Rationale**:

- The user's pain is inspection and understanding, not authoring.
- Read-only keeps the first slice bounded and much easier to validate against existing scene state.
- Selection sync already creates immediate value without reopening world-editing architecture.

**Alternatives considered**:

- Add visibility toggles and editing in the first slice: rejected as scope creep.
- Build a non-interactive report tree first: rejected because selection sync is a core usability requirement.

## Decision 5: Stable Node Identity Must Be Path-Based, Not Raw-Reference-Based

**Decision**: Scene graph nodes should carry a stable typed identity path assembled from semantic path segments such as map, tile, region id, CK24, uid, or chunk coordinate.

**Rationale**:

- Raw object references are not stable across refreshes, streaming, or reconstruction.
- Path-based identity makes reverse selection, expansion persistence, and snapshot export deterministic.
- PM4 and terrain both need identities that survive regeneration of the visible scene.

**Alternatives considered**:

- Runtime object hash codes: rejected because they are not stable across refreshes.
- Plain display labels as ids: rejected because labels collide and may change.

## Decision 6: PM4 Uses The Current Region -> Object -> Sub-object Hierarchy, With Newer Field Discoveries As Metadata

**Decision**: The PM4 branch should keep the current hierarchy rooted at `MSHD.Field04` region id, then CK24 object, then sub-object, while exposing `TypeFlags`, `GroupObjectId`, and related fields as node metadata rather than inventing a second ownership model.

**Rationale**:

- The current best PM4 ownership model is already documented and used elsewhere in the repo.
- Newer `TypeFlags` observations are valuable, but they do not yet justify replacing the ownership hierarchy.
- The workbench should clarify PM4, not destabilize it.

**Alternatives considered**:

- Rebuild the PM4 tree around `TypeFlags`: rejected because the field semantics are still partial.
- Show PM4 as a flat object list: rejected because it loses the point of a scene graph.

## Decision 7: Cross-Tile Or Shared Structures Need Canonical Owners Plus References, Not Blind Duplication

**Decision**: When a structure naturally spans multiple tiles or domains, the graph should pick one canonical owner node and surface the cross-links as metadata or subordinate reference nodes instead of duplicating a full subtree under every possible parent.

**Rationale**:

- Blind duplication destroys browseability and breaks stable selection identity.
- Canonical ownership keeps the outliner comprehensible while still exposing cross-tile reality.

**Alternatives considered**:

- Duplicate the same object under every tile: rejected because selection and filter results would become ambiguous.
- Hide cross-tile information entirely: rejected because that loses important scene truth.
