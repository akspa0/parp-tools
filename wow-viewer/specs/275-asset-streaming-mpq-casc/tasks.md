# Tasks: Spec 275 High-Performance Asset Streaming for MPQ and CASC

**Spec**: [Spec 275: High-Performance Asset Streaming for MPQ and CASC](spec.md)  
**Plan**: [Technical Design Plan](plan.md)  
**Status**: Authoring / Pending Phase 1 Execution  
**Receipt**: `evidence/receipt-spec275.md` (to be authored upon verification)  

---

## Phase 1: In-Memory WMO Parsing & Write Containment

- [ ] **T001**: Add in-memory parsing overload `public WmoV14Data ParseWmoV14(byte[] wmoData)` to `WmoV14ToV17Converter.cs` in `WowViewer.Core.IO.Converters`, parsing directly from a `MemoryStream` / `BinaryReader`.
- [ ] **T002**: Refactor `WorldAssetManager.LoadWmoDataModel` to eliminate `Path.Combine(Path.GetTempPath(), $"wmo_{Guid.NewGuid():N}.tmp")` and `File.WriteAllBytes`, calling the in-memory `ParseWmoV14(data)` overload directly.
- [ ] **T003**: Author unit tests in `WowViewer.Core.Tests` verifying byte-level parity between file-based and in-memory WMO v14 parsing.
- [ ] **Gate A**: Run `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` and verify 100% pass with 0 errors.

---

## Phase 2: MPQ Lookahead Dependency Prefetching

- [ ] **T004**: Implement `MpqAssetDependencyExtractor.cs` in `WoWViewer.Terrain` to inspect raw M2 and WMO root bytes:
  - For M2s: extract referenced `.skin` paths and `.blp` texture filenames.
  - For WMOs: read `GroupCount` from `MOHD` header and generate `[base]_[000..GroupCount-1].wmo` group paths, plus texture names from `MOTX`.
- [ ] **T005**: Wire dependency extraction into `WorldAssetManager.PrefetchModelBytes` and `MpqDataSource.PrefetchFile` so that dependent group files, skins, and textures are queued into background archive prefetchers ahead of time.
- [ ] **T006**: Author unit tests verifying dependency extraction on sample M2 and WMO byte chunks.
- [ ] **Gate B**: Build solution with 0 errors and verify prefetch cache hit rate improvements in MPQ mode.

---

## Phase 3: Off-Thread Model & Mesh Decoding Worker

- [ ] **T007**: Define `PreparedModelPackage.cs` data contract in `WoWViewer.Terrain` to hold CPU-decoded models (`M2StaticRenderModel` / `WmoV14Data`), bounds, and resolved paths.
- [ ] **T008**: Implement `WorldAssetDecodeWorker.cs` in `WoWViewer.Terrain` with a bounded worker pool reading from a `Channel<AssetDecodeRequest>` and enqueuing completed packages into a `ConcurrentQueue<PreparedModelPackage>`.
- [ ] **T009**: Update `WorldAssetManager.cs` to submit deferred M2 and WMO loads to `WorldAssetDecodeWorker`.
- [ ] **T010**: In `WorldAssetManager.ProcessPendingLoads`, consume completed packages from the worker queue and perform fast-path OpenGL buffer allocation (`glGenVertexArrays`, `glBufferData`) within the frame budget.
- [ ] **Gate C**: Run `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` and verify all tests pass.

---

## Phase 4: Dynamic Frustum & Camera-Distance Priority Queue

- [ ] **T011**: Implement `WorldAssetPriorityQueue.cs` in `WoWViewer.Terrain` supporting scored item insertion:
  $$\text{Score} = \text{DistanceSq} \times (\text{InFrustum} ? 0.25f : 1.0f)$$
- [ ] **T012**: Connect camera position and frustum updates from `WorldScene.Update` / `FlushPendingVisibleMdxLoads` into `WorldAssetPriorityQueue`, pruning out-of-range candidates beyond maximum streaming distance ($> 800\text{ yds}$).
- [ ] **T013**: Author unit tests for `WorldAssetPriorityQueue` verifying distance ordering, frustum boosting, and duplicate suppression.
- [ ] **Gate D**: Verify priority queue tests pass with 0 errors.

---

## Phase 5: WMO Material Texture Off-Thread Streaming & Deduplication

- [ ] **T014**: Refactor `WmoMaterialManager.cs` to decode BLP material textures off-thread into `BlpCompressedTexture` packets (DXT1, DXT3, DXT5).
- [ ] **T015**: On the render thread, upload compressed textures directly via `glCompressedTexImage2D` with pre-computed mipmaps, removing `ImageSharp` RGBA expansion and `glGenerateMipmap` from the render thread.
- [ ] **T016**: Add cross-WMO texture deduplication so identical materials across multiple buildings share the same GPU texture handle.
- [ ] **Gate E**: Run viewer on complex WMO scenes (e.g. Ironforge / Stormwind) and verify textures render properly without material fallback errors.

---

## Phase 6: Integration, Telemetry & Operator Verification

- [ ] **T017**: Register streaming telemetry metrics in `RuntimeStats` (worker queue depth, completed packages, upload time per frame, cache hits).
- [ ] **T018**: Full solution build: `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- [ ] **T019**: Run all test suites across the solution: `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- [ ] **T020**: Prepare operator verification commands for high-speed flight across MPQ (Classic 1.12 / 3.3.5) and CASC (1.60.1 / 4.3.4) maps.
- [ ] **T021**: Author verification receipt in `evidence/receipt-spec275.md`.
