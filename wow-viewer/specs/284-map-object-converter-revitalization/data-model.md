# Data Model: Converter Revitalization

**Status**: Proposed contracts, not implemented. Namespace ownership is defined in [plan.md](plan.md).

## ClientEvidence / ClientSelection

- `ClientKey`: stable local identifier, independent of display name or role.
- `Root`, `MountedRoots`, `Product`, `BuildVersion`, `BuildKey`: configured identity, including
  base/overlay order where relevant. Exact build may remain unknown until resolved.
- `Origin`: loaded session / operator-saved known-good / explicit root.
- `Evidence[]`: observation type, observed value, file/content hash, confidence and conflict reason.
- `Fingerprint`: digest of canonical identity observations; content plan adds dependency hashes.
- `Availability`: available / missing / changed / ambiguous.
- `SavedDesignation`: preserve operator-known-good designation separately from conversion evidence.
- `Role`: source or target **in a request**, not a permanent property of a client.

Saved viewer entries are adapted from existing settings. The CLI can read those same saved records
through an explicit settings-file path and selection key; it never assumes a running viewer session.
Source/target reads are isolated from live editor state and client roots remain read-only.

## ConversionProfile / RouteDecision

- `ConverterKind`: Map / Model / Wmo.
- `ProfileId`, `ClientIdentity`, `FormatSignatures`, `ReaderCapabilities`, `WriterCapabilities`.
- `Limits`: measured group/index/vertex/material/layer/ID constraints, with evidence references.
- `RouteId`, `SourceProfile`, `TargetProfile`, ordered transform stages, capability version.
- `FeatureDisposition[]`: feature, preserve/transform/reduce/block, reason, loss code and evidence.
- `Status`: supported / needs explicit loss choice / experimental / ambiguous / unavailable.
- `EvidenceLevel`: planned / implemented / real-file-readback / viewer-witness / target-witness.

Do not encode upgrade/downgrade as a comparison of version strings. Model and WMO format families
may differ independently within one client build. A route may compose measured intermediate stages.

## ConversionRequest / PlannedConversion

- Schema version, converter kind, source/target snapshots and exact selected content.
- For Map: finalized v65 store identity/layout/digest, source revision/tile selection, native-field
  coverage requirements and typed placement-table digests. Source backend may be MPQ/loose/CASC;
  CASC install/product/build and path/FDID evidence are retained through ingestion.
- Map ID/name, canonical tile set, or source asset locator (path or FileDataID).
- Target profile, preset and resolved domain options; generation/static fallback explicit.
- Dependency policy: include converted closure or use proven target dependencies, with inventory.
- Output root, namespace, collision policy, ID allocation range and reservation manifest digest.
- Required/accepted loss codes, planned outputs/dependencies and source-input hashes.
- `RequestDigest`: canonical inputs/options/profile/tool-version digest, excluding elapsed time/run ID.

Defaults are resolved once before digesting. Execution rejects changed inputs or newly discovered
unaccepted losses and requires re-plan. Run metadata is separate from deterministic content outputs.
Map source selection resolves a matching v65 store or prepares ingestion through existing harvester
tooling. Missing v65 data cannot silently fall back to direct-client terrain decoding. Python owns
store I/O; the C# map service consumes its typed reverse stream. Object geometry is separately resolved.

## PlacementIdentityGraph / PlacementIdMap

- Source metadata comes from v65's typed placement/reference groups, with original integer bits
  extracted before any float32 projection. Re-itemization never mutates source v65 IDs.
- `SourceRecordKey`: source fingerprint + map + tile/global owner + placement kind + source row.
- `SourceInstanceKey`: explicit grouping of records proven to represent one physical instance.
- `OriginalUniqueIdRaw`, canonical asset identity, full transform, flags, source bounds/provenance.
- `ReferenceEdges[]`: referring record/chunk, source semantics (index/uniqueID), destination identity.
- `OutputInstanceKey`, `ParentInstanceKey` for actual generated one-to-many replacements.
- `TargetUniqueId`, target scope and canonical order key; original ID remains immutable.
- `ReservedIds`, target representable range, reservation inventory coverage and digest.

Invariants:

1. Every distinct output instance has one allocation record and one in-range unreserved ID.
2. All records referencing that instance use the same allocation; different instances cannot collide.
3. M2 and WMO share the conservative output-map namespace. Group/material/local doodad indices do not.
4. Duplicate original IDs cannot collapse records without matching instance evidence. Ambiguous
   identity blocks conversion or requires an explicit identity decision recorded in the plan.
5. Allocation is independent of traversal/thread order. Changes to scope/reservations create a new
   request; replaying the same request reproduces its allocation.
6. Rewriting updates both placement fields and target-specific references, including Alpha's
   preserved uniqueID arrays before writer index reconstruction.

## DependencyRecord / TopologyMap

Map materialization records name the canonical writer/version, fresh-document versus height-patch
mode, native-versus-derived WDL policy and source field→target DTO→writer/readback disposition.
Source/target layer-opacity reports join through explicit layer mappings, including reduced/zero/
occluded layers, metadata changes and quantization/spatial error. Reports never mutate source v65.

- Source content identity: path/FDID, hash, source client, asset kind and owning dependency edge.
- Target path, target-normalized collision key, format/profile, generated/converted/reused status.
- Companions: textures, skins, animations, WMO groups, doodad assets and referenced tables as needed.
- Missing/unresolved/cyclic edges and selected policy; no placeholder counts as resolved.
- WMO source group → output groups, remapped root references and semantic dispositions.
- Geometry source face/material/component → output partition lineage where topology changes.

## ConversionMesh / MeshCandidate

- Local basis, units, pivot, transforms, positions, normals, UV channels, indices and material groups.
- Connected components, protected seams, source visual/collision provenance and optional skin data.
- Source hash, processing parameters/algorithm version, target profile and output content hashes.
- Before/after vertices/triangles/groups/bounds, measured geometric error and material/seam changes.
- Candidate state: proposed → generated → readback-validated → review-pending → accepted/rejected.
- Reasons and actor/time for explicit acceptance/rejection; substitution references accepted content.

No automatic promotion from generated to accepted. Static replacements label discarded animation,
collision or interior semantics. A debug proxy cannot satisfy visual-generation acceptance.

## ConversionRun / ConversionReport

- Run ID, request digest, stage, structured progress/cancellation, paths, hashes and durations.
- Per-map/tile/asset outcome and feature/loss diagnostics; ID/path/topology manifest locations.
- Source/target evidence snapshots and separate verification receipts for code/readback/viewer/target.
- Execution state: planned → running → validating → completed; alternatively blocked/failed/cancelled.
- Publication state: staging / validated-output / partial-retained. Completed means the requested
  output was validated under accepted losses; it does not imply visual or target-client acceptance.

Required dependencies failing prevent a full-map completed result. Batch partial success is explicit
and nonzero; detailed outcomes retain successful maps without mislabelling the batch as complete.
