# Feature Specification: CASC Data Source (Local Install + Remote CDN)

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "We may have to add CASC support (we likely have the libraries already, via Warcraft.net), so we can open clients from Remote (CDN), and then support clients above version 5.0.1, to read the assets properly. We ultimately can learn from existing implementations like https://github.com/Kruithne/wow.export which handles all casc version data, as well as what we need to make it happen in the renderer. Lots of research work ahead, but it will be worth it, use speckit. Pin this to the version 0.6 release."

## Context

The viewer reads game data from loose files and MPQ archives only. Every World of Warcraft client
from 6.0 (Warlords of Draenor) onward stores its data in **CASC**, so none of those clients can be
opened today. The data can also be fetched on demand from Blizzard's content CDN without a local
install.

This spec is the storage layer only: getting bytes out of CASC. Understanding the modern file
formats those bytes contain is [Spec 239](../239-modern-client-assets/spec.md). Spec 237 (ADT/v22
terrain) depends on both, because its textures and models are expected to resolve against a modern
client.

**Repo facts (measured 2026-09-16):**

- `IDataSource` already mentions "CASC storage" in its doc comment, but no implementation exists.
- **Warcraft.NET has no CASC support**; it is a file-format library only. The assumption that CASC
  comes with it is incorrect.
- `libs/Marlamin/WoWTools.Minimaps` references `CascLib` and `TACT.NET` as nested folders, but both
  are **empty** (uninitialized nested submodules). Populated copies exist only inside the read-only
  `gillijimproject_refactor`, which wow-viewer may not reference (Constitution I).
- The CASC library Marlamin currently maintains and uses in `wow.tools.local` is **TACTSharp**
  (`wowdev/TACTSharp`, MIT, active as of 2026-08). Marlamin's `CascLib` fork is unlicensed and has
  been idle since 2026-03.
- `libs/*` entries are git submodule links, but the repo-root `.gitmodules` has **no entries** for
  them, so a fresh clone can't initialize them.
- The core has **zero FileDataID support**. Modern clients address files by numeric id, so the data
  source must support id-addressed reads from day one.
- A community listfile is already vendored at `libs/wowdev/wow-listfile`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Open a locally installed modern client (Priority: P1)

The operator points the viewer at a local World of Warcraft installation, picks one of the installed
products (for example retail, classic, PTR), and the viewer can list and read its files by path and
by FileDataID, exactly as it does for MPQ clients today.

**Why this priority**: This is the fastest route to real data, and it is the baseline every later
story and Spec 239 build on.

**Independent Test**: Open a local install, read a known file by path and by id, and confirm its
bytes hash-match the same file obtained from an independent tool.

**Acceptance Scenarios**:

1. **Given** a folder containing a CASC install, **When** the operator opens it, **Then** the installed products and their build versions are listed.
2. **Given** an opened product, **When** a file is requested by virtual path, **Then** its bytes are returned if the listfile maps that path to an id present in the build.
3. **Given** an opened product, **When** a file is requested by FileDataID, **Then** its bytes are returned whether or not the listfile knows a name for it.
4. **Given** a file that isn't in this build, **When** it is requested, **Then** "not present" is reported without an error.

---

### User Story 2 - Stream a client from the CDN without installing it (Priority: P1)

The operator chooses a product and region, sees the available builds, picks one, and the viewer
streams only the files it actually needs from the CDN, keeping them in a local cache so a second
visit is fast and works offline.

**Why this priority**: This was the headline of the request. It removes the need to install (or
own disk space for) every client build under study.

**Independent Test**: With no local install, open a build remotely, load a map in the viewer,
close, disconnect from the network, reopen the same build, and load the same map from cache.

**Acceptance Scenarios**:

1. **Given** network access, **When** the operator selects a product and region, **Then** the current build versions for that product are listed.
2. **Given** a selected remote build, **When** the viewer requests a file, **Then** only the data needed for that file is downloaded, and it is cached on disk.
3. **Given** a build whose files were cached earlier, **When** the network is unavailable, **Then** cached files still load and uncached files report "unavailable offline".
4. **Given** a slow or failing CDN host, **When** a download fails, **Then** alternate configured hosts are tried and the failure is reported per file without freezing the viewer.

---

### User Story 3 - Open a specific historical build (Priority: P2)

The operator opens a build that is no longer "current" on the product version service (for example
an older 7.x or 8.x build) by supplying its build identity, through a configured CDN or archive
mirror that still serves it.

**Why this priority**: The project studies many eras, and the build that matters is rarely the
current one. It is P2 because it depends on external mirrors holding the data.

**Independent Test**: Open a non-current build by identity from a configured mirror and read one
known file by id.

**Acceptance Scenarios**:

1. **Given** a build identity and a mirror that serves it, **When** the operator opens it, **Then** it behaves exactly as a current remote build.
2. **Given** a build identity that no configured host serves, **When** the operator opens it, **Then** the hosts tried and their responses are reported.

---

### User Story 4 - Encrypted files are handled honestly (Priority: P2)

Some files in modern builds are encrypted. Files whose keys are known decrypt transparently; files
whose keys are not known are reported as "encrypted, key unavailable" and are never returned as
garbage bytes.

**Why this priority**: Returning undecrypted bytes as if they were valid would silently corrupt
every reader and renderer downstream.

**Independent Test**: Request a file known to be encrypted with a published key and one with an
unpublished key; the first decodes to a valid file header, the second reports the key as missing.

**Acceptance Scenarios**:

1. **Given** a known key set, **When** an encrypted file with a known key is read, **Then** the decrypted bytes are returned.
2. **Given** an encrypted file with an unknown key, **When** it is read, **Then** a "key unavailable" result names the missing key id and no bytes are returned.
3. **Given** a refreshed key set, **When** the operator updates it, **Then** previously unavailable files become readable without reopening the application.

---

### User Story 5 - Every result is traceable to its build (Priority: P2)

Every file read, evidence receipt and log line produced from CASC data records the product, build
version and build configuration identity it came from.

**Why this priority**: Constitution III/VI require build fingerprints, and remote builds change
under the same product name over time.

**Independent Test**: Produce a receipt from a CASC-backed command and confirm it names the product,
version string and build configuration hash.

**Acceptance Scenarios**:

1. **Given** any opened CASC build, **When** its identity is queried, **Then** the product, version and build configuration hash are returned.

---

### User Story 6 - Existing MPQ and loose-file clients are unaffected (Priority: P1)

**Independent Test**: Open an existing 0.5.3 and 3.3.5 client after the change; load time and
rendered output are unchanged, and the existing test suite passes.

### Edge Cases

- The local install has several products (retail, classic, PTR) sharing one data folder: each is listed and opened separately.
- The local install is partially downloaded (the launcher was interrupted): unavailable files report "not present locally". The data source does **not** silently fall back to the CDN unless the operator enabled hybrid mode.
- The listfile maps a path to an id that the build lacks, or the build has ids with no listfile name: both are normal and must not error.
- A file has several locale/content-flag variants: the operator's configured locale wins, and the choice is recorded.
- The cache disk is full, or the cache directory is not writable: reads still work (network-only), with a warning.
- A cached object is corrupt (hash mismatch): it is discarded and re-downloaded, never served.
- The root manifest format differs by era (the pre-8.2 root, the 8.2+ root and later changes): each supported era is proven on a real build of that era.
- The build is older than any supported root format: it is reported as unsupported with the detected format, not misread.
- Many concurrent reads from viewer streaming threads: reads are thread-safe, and duplicate in-flight downloads of the same object are coalesced.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST open a local CASC installation, enumerate its products and builds, and open one selected product.
- **FR-002**: The system MUST open a remote build from the CDN, selected either from the product's current version list or by explicit build identity.
- **FR-003**: The system MUST read files by FileDataID and by virtual path, resolving paths through the listfile.
- **FR-004**: The system MUST report a missing file, an unknown key and an unavailable-offline file as distinct, non-exceptional results.
- **FR-005**: Remote reads MUST download only what the requested file needs and MUST cache it on disk, verified by content hash before use.
- **FR-006**: The cache location, CDN hosts/mirrors, region, locale and key-set source MUST be operator configuration; none may be hardcoded (Constitution VI).
- **FR-007**: The system MUST decrypt files whose keys are in the configured key set, and MUST NOT return undecrypted bytes for files whose keys are unknown.
- **FR-008**: The opened build MUST expose its identity (product, version, build configuration hash, CDN configuration hash) for receipts and logs.
- **FR-009**: The CASC source MUST plug into the viewer's existing data-source abstraction so every existing reader and renderer can consume it unchanged, and MUST add id-addressed reads to that abstraction without breaking MPQ or loose-file sources.
- **FR-010**: Reads MUST be safe under the viewer's concurrent streaming load.
- **FR-011**: The inspect CLI MUST be able to list products/builds, show build identity, read a file by path or id to disk, and report cache statistics.
- **FR-012**: CASC containers MUST remain read-only inputs. The system MUST NOT write, repack or emit a CASC/TACT container (Constitution VII). The download cache holds verbatim input objects for local reuse only.
- **FR-013**: Every root-manifest era claimed as supported MUST be proven on a real build of that era, with the proof recorded.
- **FR-014**: The CASC library dependency MUST be vendored inside `wow-viewer` so that a fresh clone can build it (Constitution I).

### Key Entities

- **Product**: A distributable client line (retail, classic variants, PTR, beta), identified by its product code.
- **Build**: One version of a product, identified by version string plus build configuration and CDN configuration hashes.
- **Storage source**: Where a build's data comes from (a local install, a remote CDN, or both in hybrid mode).
- **File reference**: A FileDataID, optionally named by the listfile, with locale/content-flag variants.
- **Key set**: The known decryption keys by key id, with their origin.
- **Cache**: The on-disk store of verbatim downloaded objects, keyed by content hash, with size and hit statistics.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For at least 1,000 randomly sampled FileDataIDs from one local build, 100% of the bytes returned hash-match an independent reference tool's extraction of the same build.
- **SC-002**: The same sample read from the CDN build of the same version is byte-identical to the local read.
- **SC-003**: At least one real build from each supported root-manifest era opens and passes SC-001 on a smaller sample (at least 200 ids).
- **SC-004**: A map the viewer can already render (from a client that exists both as MPQ and CASC, where one exists; otherwise a map that loads in the viewer's minimum rendering path) opens from a remote build in under 2 minutes cold, and in under 20 seconds when reopened from cache.
- **SC-005**: With the network disabled, 100% of previously cached files load and 0 uncached files return bytes.
- **SC-006**: 0 encrypted files with unknown keys return bytes; 100% of files with known keys decode to a valid header for their file type.
- **SC-007**: The existing test suite passes, and load time for an existing MPQ client changes by no more than 5%.

## Assumptions

- **TACTSharp** is the preferred CASC library (Marlamin-maintained, MIT, used by `wow.tools.local`). The alternative, `CascLib`, stays a fallback, subject to license confirmation. The final choice is made in plan research, and can be confirmed directly with Marlamin, who is in contact with the operator.
- wow.export (MIT, JavaScript) is a **reference** for behavior (remote/local source handling, cache layout, key handling, era differences). Its code is not ported wholesale.
- The operator has lawful access to the builds being opened (Data Policy: bring your own data). The cache stays local and is never distributed.
- The community listfile and a community key set are acceptable inputs, refreshed by the operator.
- **No client newer than 5.0.1 is installed (2026-09-16)**, and installing the live client needs the operator to free disk space first. Remote CDN streaming (US2) is the first route to real modern data. US1 (local install) is implemented alongside it but validated only once an install exists. SC-001 runs against a remote build until then, and SC-002 waits for an install.
- Build-to-build patching, the install manifest, and background pre-download of whole builds are out of scope. Only on-demand reads are in scope.
- Understanding modern file formats (FileDataID-referencing WDT/ADT/M2/WMO, DB2) is Spec 239, not this spec.
