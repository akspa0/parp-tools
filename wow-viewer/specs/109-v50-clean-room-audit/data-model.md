# Data Model: V50 Clean-Room Dataset and Repository Reset

## ArtifactRecord

- `artifact_id`: deterministic identity of the record
- `kind`: dataset, checkpoint, model, prior archive, manifest, report, tmp tree, cache, client copy,
  command surface, environment, or continuity document
- `resolved_path`: audit-time absolute path
- `observed_bytes`: measured size for generated filesystem artifacts
- `content_identity`: metadata-tree or content hash at the achieved proof level
- `owner`: canonical workflow or `unknown`
- `proof_level`: inventory, contract, sampled, full, or quality
- `trust_state`: unverified, rejected, verified, or promoted
- `disposition`: keep, quarantine, verify, migrate, or remove-candidate
- `dependencies`: artifact IDs that consume or protect this artifact
- `evidence_paths`: reports supporting the state

## ClientBuildEvidence

- `client_library_id`: operator-defined logical library identity
- `build_id`: normalized client build
- `root_argument`: recorded only in local run evidence, never hardcoded in source
- `executable_identity`: size and hash where available
- `archive_catalog_identity`: ordered archive/listing fingerprint
- `required_paths`: build-critical paths checked
- `reader_identity`: existing C# harvester/build identity
- `verification_time` and `result`

## DatasetStoreManifest

- `model_family` and `release`
- `schema` and `store_id`
- `build_id`
- `producer_identity` and normalized build parameters
- `client_build_evidence_id`
- `index_identity`, row count, map coverage, and partition contract
- `signals`: collection of DatasetSignal records
- `row_lineage_identity`
- `finalization_state`: incomplete or complete
- `unavailable_signals`: explicit reasons, never silent fills

## DatasetSignal

- `name`, `dtype`, `row_shape`, fill policy, semantic range, and authoritative source
- `required` flag
- `content_identity`
- `coverage_count`
- `verification_scope` and report identity
- `migration_policy`: copy-if-verified, fresh-only, derived-after-build, or unavailable

## RowLineage

- `store_row`, build, map, tile coordinates, and source group
- source artifact/row identity for migrated data
- per-signal action: copied, freshly extracted, derived, or unavailable
- per-signal source and destination hashes
- split/group identifiers used by curriculum manifests

## VerificationResult

- artifact, signal, and optional row scope
- check name and version
- expected and observed values
- status: pass, fail, skipped, or unavailable
- proof level achieved
- evidence output and timestamp

## MigrationLedger

- source and destination store identities
- ordered RowLineage records
- copied/fresh/omitted counts per signal
- resume checkpoint and completion state
- final destination identities

## CurriculumManifest

- manifest identity and release
- referenced canonical store IDs
- selected row IDs and source groups
- train/validation/test partition
- selection reasons and policy identity
- no copied array payloads

## CleanupPlan

- plan ID, creation time, inventory identity, and approved generated roots
- protected artifact IDs
- ordered cleanup targets with resolved paths, kinds, identities, observed bytes, replacement proof,
  dependency result, and approval state
- expected recovered bytes
- dry-run result
- apply confirmation hash

### Cleanup state transitions

`discovered -> quarantined -> candidate -> approved -> deleted -> verified-absent`

Any failed dependency, identity, path-boundary, or replacement check returns the target to
`quarantined`. Client libraries and the active v50 release never enter `candidate`.

## V50ReleaseManifest

- release identity
- complete DatasetStoreManifest IDs
- CurriculumManifest IDs
- compatible checkpoint/prior identities when they exist
- audit and migration report identities
- cleanup plan/report identities
- release state: building, verified, promoted, or retired
