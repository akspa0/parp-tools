# Research: CASC Data Source

Survey date: 2026-09-16.

## Findings (measured)

| Finding | Evidence |
|---|---|
| No CASC implementation in wow-viewer | `IDataSource.cs` mentions CASC in a doc comment only; the implementations are `LooseFileDataSource` and `MpqDataSource` |
| Warcraft.NET has no CASC/TACT/BLTE code | `libs/ModernWoWTools/Warcraft.NET/Warcraft.NET/Files` holds ADT/BLP/M2/SKIN/WDL/WDT/WMO format types only |
| Vendored `CascLib` / `TACT.NET` folders are empty | `libs/Marlamin/WoWTools.Minimaps/{CascLib,TACT.NET}` are uninitialized nested submodules |
| Populated CascLib exists only in read-only reference trees | `gillijimproject_refactor/lib/{wow.tools.local,MapUpconverter}/CascLib`: not referenceable (Constitution I, read-only rule) |
| `libs/*` are gitlinks with no `.gitmodules` entries | `git ls-files -s` shows mode 160000 for 6 libs; the root `.gitmodules` lists none of them |
| wow.tools.local (Marlamin) uses TACTSharp | `wow.tools.local.csproj` has a ProjectReference to `TACTSharp\TACTSharp.csproj` |
| Library activity | TACTSharp: MIT, pushed 2026-08-25. Marlamin/CascLib: no license, pushed 2026-03-07. wow.export: MIT, JS, pushed 2026-06-22 |
| Zero FileDataID support in core | 0 files match `MAID`/`SFID`/`TXID`/`GFID`/`MODI`/`FileDataId` |

---

### R1: CASC library

- **Decision**: TACTSharp, vendored as a submodule.
- **Rationale**: It is maintained by the people who document the format, it is MIT-licensed, and it is what Marlamin's own current tooling depends on. It handles local, online and encrypted data.
- **Alternatives**: Marlamin/CascLib (older CASCExplorer lineage; no license file, so it can't be vendored until that is resolved). Porting wow.export's `src/js/casc/*` (MIT, but JS, and a rewrite duplicates maintained code). An in-house reader (rejected; see Complexity Tracking).
- **Ask Marlamin**: whether TACTSharp is the recommended embedding target for a third-party viewer, and whether its public API is stable enough to pin.

### R2: Root manifest era coverage

- **Question**: Does TACTSharp read every root format from 6.0 through current (the legacy root, the 8.2 "TSFM" root, and later root revisions, including TVFS-based products)?
- **Plan**: Phase 0 opens one real build per era and records a pass/fail table. Where TACTSharp lacks an era, wow.export's `casc-source.js`/`casc-source-local.js`/`casc-source-remote.js` are the behavioral reference for a contribution upstream (preferred) or a local adapter.

### R3: Id-addressed reads in existing abstractions

- **Decision**: Add a separate `IFileDataIdReader` interface in Core.IO, and default interface members on `IDataSource`. MPQ and loose sources return not-present.
- **Rationale**: This is non-breaking for the ~every-consumer surface, and it lets Spec 239 readers depend on ids without casting to CASC types.

### R4: Result types

- **Decision**: `FileReadResult` separates NotPresent, KeyUnavailable(keyId), OfflineUnavailable and Failed(reason).
- **Rationale**: FR-004 and US4. Today `ReadFile` returns null for every failure, which would hide encryption and cache states.

### R5: Hybrid mode

- **Decision**: Off by default. When enabled, a local miss is fetched from the CDN for the **same build identity** only.
- **Rationale**: A silent remote fallback would mix builds or hide an incomplete install (spec edge case).

### R6: Historical builds

- **Question**: Blizzard's CDN does not reliably serve old build configurations. Which mirrors serve historical builds, and in what URL layout?
- **Plan**: Treat hosts as an ordered, configurable list. Record which hosts serve which surveyed builds. Ask Marlamin which community mirrors are appropriate to configure.

### R7: Cache and Constitution VII

- **Decision**: The cache holds verbatim objects keyed by content hash, is verified on read, and is never exported. It is documented as a read-side input cache.
- **Resolved 2026-09-16**: the operator confirmed caches are fine; the project already uses caches.

### R8: Independent reference for byte verification (SC-001)

- **Decision**: Compare against files extracted by an independent tool from the same build: wow.export, or wow.tools.local run by the operator. Both can stream from the CDN, so **no local install is needed for the reference**. Record the tool name and version in the receipt.
- **Detector power**: Deliberately alter one byte in a reference file; `casc verify` must report exactly that id as a mismatch.

### R9: Listfile

- **Decision**: Reuse `libs/wowdev/wow-listfile` (already vendored). Refresh is an operator action. Ids unknown to the listfile stay browsable by id.

### R10: What to learn from wow.export (reference, not port)

| wow.export path | What to learn |
|---|---|
| `src/js/casc/casc-source-local.js`, `casc-source-remote.js`, `casc-source.js` | Local vs remote parity; how root, encoding and index are layered |
| `src/js/casc/build-cache.js`, `cdn-resolver.js` | Cache layout, host ranking/failover |
| `src/js/casc/version-config.js`, `cdn-config.js`, `realmlist.js` | Product/version discovery |
| `src/js/casc/tact-keys.js`, `salsa20.js`, `blte-reader.js` | Key refresh, encrypted block handling |
| `src/js/casc/listfile.js`, `locale-flags.js`, `content-flags.js` | Listfile merging, locale/content-flag selection |
