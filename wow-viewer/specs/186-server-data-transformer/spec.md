# Feature Specification: Server data transformer and world content browsing

**Feature Branch**: `186-server-data-transformer`

**Created**: 2026-08-23

**Status**: Draft

**Input**: User description: "I don't really want to be the one saving the archive, just building the
tooling around letting someone go find all the sql db's and use the tooling to index it all into
better, more forward-thinking model based tooling... Might as well make the transformer for the data,
since it's all the stuff missing from the viewer for seeing npc's that we know existed, and spells
and all that really dense text data."

## Context

The viewer can render a world in exquisite detail and knows essentially nothing about who lived in
it. It can put you on the exact tile where a camp stood; it cannot tell you which creatures stood
there, what they were called, what they cast, what they dropped, or what the quest giver said. That
information exists — split across two places, in two very different states of preservation.

### Two halves that have never been joined here

| half | holds | state in this repo |
|---|---|---|
| **Client tables** (DBC/DB2) | Spells, items, display info, the game's own reference data | **Already reachable.** DBCD and 1,320 WoWDBDefs definitions are vendored and wired into both core and viewer. |
| **Server databases** (fan project SQL) | Spawns, quest chains, loot tables, creature behaviour, and the dense text | **One dialect, one era, one feature.** `AlphaCoreDbReader` parses alpha-core dump text with no MySQL; `SqlWorldPopulationService` turns it into map-filtered spawn records. |

Neither half is sufficient alone. A spell's mechanics live in the client tables; *which creature
casts it* lives in a server database. A creature's model lives in the client tables; *where it stood
and what it said* lives in a server database, and only because a person put it there.

**This spec builds the transformer that joins them** and makes the result browsable in the viewer.

### The project ships tooling, not a collection

This is explicitly **not** an archive that this project curates, hosts, or redistributes. The operator
goes and finds the databases they care about and points the tooling at them. What the project owes
them is a transformer good enough that nothing is lost on the way in, and an index good enough that
the result is usable by both a person and a model.

That boundary is deliberate and it decides several requirements. Because the operator supplies the
sources, the tool's job is fidelity and attribution rather than curation — and licensing is the
operator's question about their own sources, not a gate on this spec.

### Why losslessness is a tool property, not a nicety

The server half is the part that erodes. Client tables are preserved many times over; a fan project's
SQL is one repository away from vanishing, and what it contains is twenty years of undocumented work
— someone who worked out that a creature's Z coordinate was being read wrong and fixed it by hand,
someone who reconstructed a quest chain from forum posts.

A transformer that quietly drops a column it does not model is therefore not merely lossy, it
destroys the specific thing the operator was trying to keep. Three consequences, all testable:

- **Lossless**: every source field survives, including fields this project does not interpret, proven
  by reconstructing the record and comparing it to the source.
- **Never normalise**: two projects disagreeing about one creature is a *record of two people solving
  the same problem differently*. No ingest may resolve that by silently picking a winner.
- **Attributed**: every record resolves to the project it came from, because the operator needs to
  know whose work they are looking at.

### Structured, not flattened to text

The obvious shortcut — flatten the database to prose and hand it to a model — is the one approach
that guarantees detail loss, and it loses exactly the dense structured content that motivated the
work. The store keeps records as records. A model is then handed exact, complete rows and asked a
question about them, rather than asked to have absorbed them.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Point the tooling at a server database and lose nothing (Priority: P1)

An operator obtains a fan server's database dump, points the transformer at it, and gets a
structured, attributed store — with no database engine installed at any point, and every source field
preserved whether or not this project understands it.

**Why this priority**: It is the transformer. Everything else in this spec reads what it produces, and
its fidelity is the one property that cannot be added later.

**Independent Test**: Ingest a dump, reconstruct records from the store, and compare field by field
against the source. Any difference is a failure.

**Acceptance Scenarios**:

1. **Given** a server database dump, **When** ingestion runs, **Then** a structured store is produced
   with no database engine installed or running at any stage.
2. **Given** an ingested record, **When** it is reconstructed and compared to its source, **Then**
   every field matches, including fields the entity model does not interpret.
3. **Given** a source field with no place in the model, **When** ingestion runs, **Then** it is
   preserved and reported as unmapped — never summarised, flattened to prose, or dropped.
4. **Given** an ingested store, **When** any record is inspected, **Then** it resolves to the project
   it came from, the game version it describes, and the ingestion that produced it.
5. **Given** an unchanged source already ingested, **When** ingestion runs again, **Then** it costs no
   reprocessing and produces an identical store.

---

### User Story 2 - See who lived here (Priority: P1)

A user standing in a location in the viewer can see the creatures and objects that belonged there,
open any of them, and read what is actually known about it — its name, its abilities, what it
dropped, what it said.

**Why this priority**: This is the gap the user named. It is also the first thing that makes the
transformer visibly worth having, and it needs no simulation whatsoever — the data is enough.

**Independent Test**: Load a map with an ingested source, select a known creature, and confirm its
client-table and server-database facts are shown together and are correct against both sources.

**Acceptance Scenarios**:

1. **Given** an ingested source and a loaded map, **When** the user views a populated location,
   **Then** the creatures and objects recorded there are listed.
2. **Given** a selected creature, **When** its detail is shown, **Then** facts joined from client
   tables and from the server database are shown together, each labelled with where it came from.
3. **Given** an entity referencing content the client tables lack, **When** it is shown, **Then** the
   missing reference is reported as missing rather than rendered blank or invented.
4. **Given** dense text content, **When** it is displayed, **Then** it is shown as stored, without
   truncation or rewriting.

---

### User Story 3 - Browse and search the dense content directly (Priority: P2)

A user searches the ingested content — spells, items, quests, creatures, text — and navigates between
related records by their real relationships.

**Why this priority**: The content is worth exploring on its own, not only where it happens to be
placed in a world. This is also what makes the store useful to someone who is not standing anywhere.

**Acceptance Scenarios**:

1. **Given** an ingested store, **When** the user searches for a term, **Then** matching records are
   returned across content kinds with their source labelled.
2. **Given** a record that references another, **When** the user follows the reference, **Then** they
   arrive at the referenced record, or are told it is absent.
3. **Given** a record the model does not fully interpret, **When** it is opened, **Then** its
   unmapped fields are still visible.

---

### User Story 4 - Compare what different projects did (Priority: P2)

An operator with two or more ingested sources can see where they disagree about the same entity, with
each project credited for what it holds.

**Why this priority**: The differences are the most interesting thing in the collection and the least
recoverable elsewhere. It is P2 because it needs two sources ingested.

**Acceptance Scenarios**:

1. **Given** two sources describing the same entity, **When** they are compared, **Then** their
   differences are reported and attributed to each project.
2. **Given** two sources ingested into one store, **When** either is inspected, **Then** neither has
   been altered, overwritten, or normalised by the other's ingest.
3. **Given** an entity only one source contains, **When** comparison runs, **Then** its absence
   elsewhere is reported as absence, not as an error.

---

### User Story 5 - Add a dialect without changing the core (Priority: P2)

Someone with a database the tooling has never seen can teach it that dialect, without modifying the
entity model or the store.

**Why this priority**: The operator, not this project, decides which databases matter. A transformer
that only understands the dialects we happened to test is not the tooling that was asked for.

**Acceptance Scenarios**:

1. **Given** an unrecognised dialect, **When** support for it is added, **Then** no change to the
   entity model or store format is required.
2. **Given** a dialect definition, **When** ingestion runs, **Then** unmapped fields are reported so
   the definition can be improved incrementally.
3. **Given** a dialect that cannot be recognised at all, **When** it is offered, **Then** the tooling
   says so plainly rather than ingesting a partial or wrong result.

---

### User Story 6 - Serve the store to external and model tooling (Priority: P3)

An external tool — including a locally served model — can query the store, retrieve exact records,
and get back precisely what is stored.

**Why this priority**: The stated end goal is model-based tooling over this data. It is P3 because the
store and its content surfaces must exist and be trusted first.

**Acceptance Scenarios**:

1. **Given** the store, **When** an external tool queries it, **Then** it receives the same records
   the viewer shows, unmodified.
2. **Given** a retrieval query, **When** it runs against a fixed index, **Then** it returns the same
   result every time, and the index version is reported with the result.
3. **Given** a model consuming the store, **When** it returns an answer, **Then** any stored value in
   that answer is traceable to the record it came from.

---

### Edge Cases

- A dump that is truncated, partially corrupted, or mid-migration.
- A dialect that stores the same concept in a structurally different way, where the entity model must
  hold both without privileging either.
- A server database referencing client content for a version the operator does not have.
- Two sources for the same project at different points in its history.
- Text content in an encoding the dump does not declare.
- A source so large that ingesting it whole is impractical in one pass.
- A record whose identifier collides with a different record in another source.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Ingestion MUST read server database sources directly, with no database engine required
  at ingestion or at any later point.
- **FR-002**: Ingestion MUST be lossless — every source field preserved, including uninterpreted ones
  — and a stored record MUST be reconstructable for comparison against its source.
- **FR-003**: An unmapped source field MUST be preserved and reported, never summarised, flattened to
  prose, or dropped.
- **FR-004**: Every stored record MUST resolve to its originating project, the game version it
  describes, and the ingestion that produced it.
- **FR-005**: Ingesting one source MUST NOT alter, overwrite, or normalise away another source's
  records; no merge may silently pick a winner.
- **FR-006**: The system MUST report differences between two sources for the same entity, attributed
  to each.
- **FR-007**: Re-ingesting an unchanged source MUST cost no reprocessing and yield an identical store.
- **FR-008**: The store MUST use this project's existing datastore conventions and codec defaults, and
  MUST NOT introduce a parallel store or competing defaults.
- **FR-009**: The system MUST join server-database content with client-table content and MUST label
  which half each displayed fact came from.
- **FR-010**: A reference that cannot be resolved MUST be reported as unresolved; the system MUST NOT
  render it blank or substitute a plausible value.
- **FR-011**: Stored text MUST be displayed and served as stored, without truncation or rewriting.
- **FR-012**: Support for an additional dialect MUST be addable without changing the entity model or
  the store format.
- **FR-013**: A source whose dialect cannot be recognised MUST be refused with a clear reason rather
  than partially ingested.
- **FR-014**: The store MUST be queryable by external tooling, returning the same records the viewer
  shows, unmodified.
- **FR-015**: Retrieval against a fixed index MUST be reproducible, and the index version MUST be
  reported with results.
- **FR-016**: No model may produce, paraphrase, complete, or substitute any stored value that is
  presented as content; models may rank, search, and summarise *alongside* records, never in place of
  them.
- **FR-017**: The project MUST NOT require, bundle, or redistribute any third-party server database;
  sources are supplied by the operator.
- **FR-018**: Ingestion MUST NOT require the operator's source to be complete or well-formed to make
  progress; partial sources MUST yield a partial store with the gaps reported.

### Key Entities

- **Source**: an operator-supplied server database, belonging to a project, describing a game version.
- **Dialect definition**: the description of how a source's structure maps to the entity model.
- **Content record**: any ingested thing — creature, object, spell, item, quest, text — with its
  fields, its unmapped remainder, and its attribution.
- **Client table**: the game's own reference data, already reachable in this project.
- **Join**: the correspondence between a server-database record and the client-table content it
  references.
- **Index**: the versioned artifact that makes retrieval over the store reproducible.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of source fields reconstruct from the store and compare equal to the original,
  including fields the entity model does not interpret.
- **SC-002**: No database engine is installed, running, or required at any point.
- **SC-003**: 100% of stored records resolve to their originating project and version.
- **SC-004**: Ingesting a second source leaves the first source's records byte-identical.
- **SC-005**: For an entity present in two sources, the system reports their differences attributed to
  each source.
- **SC-006**: Re-ingesting an unchanged source performs zero reprocessing.
- **SC-007**: At least two dialects ingest through one entity model, and a third can be added without
  changing the entity model or store format.
- **SC-008**: A user selects a creature in the viewer and sees facts from both client tables and the
  server database, each labelled with its origin.
- **SC-009**: Unresolvable references are reported as unresolved in 100% of cases; zero are rendered
  blank or filled with a substitute.
- **SC-010**: Stored text is served byte-identical to what was ingested.
- **SC-011**: The same retrieval query against the same index version returns identical results across
  repeated runs.
- **SC-012**: Zero stored values presented as content originate from a model, verified by inspection
  of the serving path.

## Assumptions

- The operator supplies the sources and owns any question about their licensing and redistribution.
  This project ships the transformer.
- Client-table access is already solved in this project and is consumed, not rebuilt.
- The existing alpha-core reader is the proof that a database engine is unnecessary, and is the
  starting point for the general ingest rather than a component preserved unchanged.
- MaNGOS-family dialects are the expected second target given the projects the user named, but the
  specific second dialect is settled in planning against real dumps.
- Model-based tooling over the store is the intended consumer; the spec requires the store be
  reproducibly queryable, not that any particular model be used.
- Long ingests and any model-serving runs are user-run.

## Out of Scope

- Collecting, hosting, curating, or redistributing any third-party server database.
- Simulating the world — that is spec 187, which consumes this store.
- Building the general-purpose datastore that specs 179–183 own; this spec consumes it.
- Reimplementing or porting any server project's source code.
- Training or fine-tuning a model.
- Writing into Blizzard container formats.
