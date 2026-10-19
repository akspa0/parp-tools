# Tasks: Spec 278 — World Dialogue, NPC Proximity Barks, Quest Flavor Inspector, and Quick Population Dashboard

## Phase 1: Core Data Models & Extractor
- [ ] T001: Implement `DialogueModels.cs` in `WowViewer.Core/Population/` (`DialogueChatType`, `DialogueTextRecord`, `QuestTemplateRecord`, `EntityDialogueFlavorData`).
- [ ] T002: Implement `DialogueFlavorExtractor.cs` in `WowViewer.Core.IO/Population/` parsing TrinityCore 3.3.5a tables (`creature_text`, `broadcast_text`, `npc_text`, `creature_queststarter`, `gameobject_queststarter`, `quest_template`).
- [ ] T003: Implement Alpha-Core 0.5.3 parsing in `DialogueFlavorExtractor.cs` (`npc_text`, `broadcast_text`, `creature_quest_starter`, `gameobject_quest_starter`, `quest_template`) strictly isolated from TrinityCore data.
- [ ] T004: Wire dialogue extraction into `WowViewer.Tool.PopulationExtractor` or standalone runner.

## Phase 2: Dataset Extraction & Version Isolation
- [ ] T005: Extract `dialogue_flavor.json` for 3.3.5a into `data/population/3.3.5a/dialogue_flavor.json`.
- [ ] T006: Extract `dialogue_flavor.json` for 0.5.3 into `data/population/0.5.3/dialogue_flavor.json` sourcing exclusively from `alpha-core`.
- [ ] T007: Add unit tests verifying dataset deserialization, quest lookup, and version isolation in `WowViewer.Core.Tests`.

## Phase 3: Quick Page Population Dashboard
- [ ] T008: Move/embed `_sqlSpawnStreaming.DrawPopulationSubTabContent()` into `WorkbenchPanelsService.DrawQuickControlsContent()` under a prominent SectionHeader.
- [ ] T009: Verify Quick page controls: version selection, auto-load on map change, reload button, AOI radius slider, and canonical map name status display.

## Phase 4: Fake WoW Chat Window & NPC Proximity Barks
- [ ] T010: Implement `FakeChatWindowService.cs` in `WoWViewer/Workbench/Services/Dialogue/` with authentic WoW chat styling, channel coloring, tabs, and auto-scrolling log.
- [ ] T011: Implement `WorldDialogueService.cs` in `WoWViewer/Workbench/Services/Dialogue/` with camera proximity check ($\le 40$ yards), per-creature cooldown (25s), and global throttle (3s).
- [ ] T012: Wire `WorldDialogueService` and `FakeChatWindowService` into `ViewerApp` / `ViewerApp_Host` following God-Class freeze conventions.
- [ ] T013: Expose chat window and proximity bark settings in Quick tab (`Enable Chat Overlay`, `Proximity Range`, `Clear Chat`).

## Phase 5: Inspector Enrichment (Quests & Gossip)
- [ ] T014: In `InspectorPayloadsService.cs`, query `WorldDialogueService` when inspecting an NPC or GameObject spawn with an associated entry ID.
- [ ] T015: Render **"Quests Offered"** section showing title, level, objectives, and flavor text.
- [ ] T016: Render **"Dialogue & Gossip"** section showing greeting text and barks.

## Phase 6: Research Documentation & Format Deepening
- [ ] T017: Author `research.md` documenting multi-version WoW server database schemas (vmangos 1.12.1, cmangos-tbc 2.4.3, TrinityCore 4.3.4).
- [ ] T018: Document remaining decoding specification for `WL*` 80-ushort block attributes and `.lit` runtime lighting pipeline boundaries.

## Phase 7: Verification & Receipts
- [ ] T019: Build solution with `dotnet build WowViewer.slnx -c Debug` (0 errors).
- [ ] T020: Run unit tests with `dotnet test WowViewer.slnx -c Debug`.
- [ ] T021: Update `wow-viewer/specs/STATUS.md` with Spec 278 ledger entry.
