# Technical Plan: Spec 278 — World Dialogue, NPC Proximity Barks, Quest Flavor Inspector, and Quick Population Dashboard

## 1. Architectural Overview & Boundaries
Following `AGENTS.md` §4, §10, and §11:
- Core data models and format readers reside in `WowViewer.Core` and `WowViewer.Core.IO`.
- No new fields or methods added to `ViewerApp` or `WorldScene` (God-Class Freeze).
- Dialogue, proximity detection, chat window rendering, and inspector enrichment are implemented via dedicated, owned services injected through `IViewerAppHost`.
- 0.5.3 Alpha data strictly sources from `alpha-core` (`world.sql`), completely isolated from 3.3.5a TrinityCore data.

```
┌─────────────────────────────────────────────────────────────┐
│                    Viewer Presentation                      │
│                                                             │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │ FakeChatWindowService │       │  Quick Tab Population │  │
│  │ (WoW Chat Overlay)    │       │  (SqlSpawnStreaming)  │  │
│  └──────────▲────────────┘       └───────────▲───────────┘  │
│             │                                │              │
│  ┌──────────┴────────────┐       ┌───────────┴───────────┐  │
│  │ WorldDialogueService  │       │ InspectorPayloadsSvc  │  │
│  │ (Proximity & Cooldown)│       │ (Quests & Gossip Sec) │  │
│  └──────────▲────────────┘       └───────────▲───────────┘  │
└─────────────┼────────────────────────────────┼──────────────┘
              │                                │
┌─────────────┴────────────────────────────────┴──────────────┐
│                  WowViewer.Core / IO                        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ EntityDialogueFlavorData / DialogueTextRecord         │  │
│  │ QuestTemplateRecord / WorldDialogueService            │  │
│  └───────────────────────────▲───────────────────────────┘  │
│                              │                              │
│  ┌───────────────────────────┴───────────────────────────┐  │
│  │ DialogueFlavorExtractor                               │  │
│  │ ├─ TrinityCoreSqlReader (3.3.5a: creature_text/quest) │  │
│  │ └─ AlphaCoreSqlReader   (0.5.3: npc_text/quest)       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. File Changes & New Components

### A. Core Library (`WowViewer.Core`)
1. **`src/core/WowViewer.Core/Population/DialogueModels.cs`**:
   - `DialogueChatType`: Enum (`Say = 12`, `Yell = 14`, `Emote = 16`, `Whisper = 18`, `System = 0`).
   - `DialogueTextRecord`: Represents one spoken line or emote (`EntryId`, `Text`, `Type`, `Probability`, `Comment`).
   - `QuestTemplateRecord`: Represents quest metadata (`QuestId`, `Title`, `QuestLevel`, `MinLevel`, `Objectives`, `Details`, `OfferRewardText`).
   - `EntityDialogueFlavorData`: Serializable map of creature barks, gossip, quest starters, and quest templates.

### B. IO & Extractor (`WowViewer.Core.IO` & Tooling)
2. **`src/core/WowViewer.Core.IO/Population/DialogueFlavorExtractor.cs`**:
   - Parser for 3.3.5a TrinityCore tables: `creature_text`, `broadcast_text`, `npc_text`, `creature_queststarter`, `gameobject_queststarter`, `quest_template`, `quest_details`.
   - Parser for 0.5.3 Alpha-Core tables: `npc_text`, `broadcast_text`, `creature_quest_starter`, `gameobject_quest_starter`, `quest_template`.
   - Enforces strict version isolation.
3. **`tools/population/WowViewer.Tool.PopulationExtractor/Program.cs`**:
   - Add `--extract-dialogue` option to export `dialogue_flavor.json` alongside spawn partitions.
   - Shipped pre-extracted datasets: `data/population/3.3.5a/dialogue_flavor.json` and `data/population/0.5.3/dialogue_flavor.json`.

### C. Viewer Shell (`WoWViewer`)
4. **`src/viewer/WoWViewer/Workbench/Services/Dialogue/WorldDialogueService.cs`**:
   - Loads and caches `dialogue_flavor.json` according to the active population dataset version.
   - Frame tick: checks distance between camera and active spawns in scene.
   - Throttling state machine: per-creature cooldown (default 25s) and global interval (default 3s).
   - Randomly selects appropriate bark and triggers chat message.
   - Exposes lookup methods for inspector: `GetQuestsForCreature(int entry)`, `GetQuestsForGameObject(int entry)`, `GetGossipForCreature(int entry)`.
5. **`src/viewer/WoWViewer/Workbench/Services/Dialogue/FakeChatWindowService.cs`**:
   - Renders authentic WoW chat overlay:
     - Semi-transparent background, movable, collapsable.
     - Channel color coding: Say (`#FFFFFF`), Yell (`#FF4040`), Emote (`#FF8040`), Whisper (`#FF80FF`), System (`#FFFF00`).
     - Auto-scroll, clear log, proximity toggle, range slider.
6. **`src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs`**:
   - In `DrawQuickControlsContent()`: add prominent section header for **"World Population (NPCs & Game Objects)"** embedding `_sqlSpawnStreaming.DrawPopulationSubTabContent()`.
   - Add **"WoW Chat & NPC Barks"** controls to Quick page.
7. **`src/viewer/WoWViewer/Workbench/Services/InspectorPayloads/InspectorPayloadsService.cs`**:
   - In `BuildMdxInspector`: if selected instance has `inst.SpawnEntryId`, query `WorldDialogueService` and append **"Quests Offered"** and **"Dialogue & Gossip"** sections.

### D. Documentation & Specification
8. **`wow-viewer/specs/278-world-dialogue-quest-flavor-and-quick-population/research.md`**:
   - Multi-version WoW server database blueprint (vmangos 1.12.1, cmangos-tbc 2.4.3, TrinityCore 4.3.4).
   - WL* 80-ushort attribute table breakdown and .lit lighting pipeline boundary analysis.

---

## 3. Phase Roadmap
- **Phase 1**: Author Core dialogue & quest models (`DialogueModels.cs`) and extractor (`DialogueFlavorExtractor.cs`).
- **Phase 2**: Generate `dialogue_flavor.json` for 3.3.5a and 0.5.3 (Alpha-Core only).
- **Phase 3**: Relocate Population controls to Quick page in `WorkbenchPanelsService.cs`.
- **Phase 4**: Implement `WorldDialogueService` and `FakeChatWindowService` (overlay, proximity barks, cooldowns).
- **Phase 5**: Enrich `InspectorPayloadsService` with Quests Offered and Gossip sections for selected spawns.
- **Phase 6**: Research documentation for multi-version server DBs and WL*/LIT format hardening.
- **Phase 7**: Validation, tests, and build check.
