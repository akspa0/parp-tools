# Spec 278: World Dialogue, NPC Proximity Barks, Quest Flavor Inspector, and Quick Population Dashboard

## 1. Problem Statement & Motivation
With Spec 276 and recent population scale/texture fixes, the viewer accurately streams hundreds of thousands of authentic creatures and game objects into the 3D scene across 0.5.3 Alpha and 3.3.5a Wrath of the Lich King. However:
1. **The world feels static and unalive**: NPCs stand silently in the scene without the ambient dialogue, emotes, yells, and greeting "barks" characteristic of the World of Warcraft experience.
2. **Population controls are buried**: Population streaming controls currently live under `Scene > Placements > Population (SQL)` in the Editor workspace, making them hard to discover for general viewer operators. Moving these controls directly to the **Quick** workbench page makes population features immediately accessible upon opening the app.
3. **Spawn metadata lacks server-side context**: While the inspector displays entry ID and model path, it does not reveal the rich server-side context available in emulator databases—such as quests offered by NPCs and game objects, quest descriptions, and gossip greeting texts.
4. **Multi-version database isolation**: 0.5.3 data must strictly source from Alpha-Core databases without falling back or mixing in 3.3.5a TrinityCore tables. Other eras (1.12.1 Vanilla, 2.4.3 TBC, 4.3.4 Cataclysm) have well-maintained open-source databases (e.g. `vmangos`, `cmangos-tbc`, `TrinityCore 4.3.4`) that can be integrated following the same pipeline.
5. **Format decoding status**: Early terrain liquid files (`WL*`) and Alpha lighting files (`.lit`) have functioning parsers but possess unmapped block fields (such as the 80 `ushort` attributes per `WlBlock`) and unproven runtime color/fog ownership.

---

## 2. User Stories & Acceptance Criteria

### User Story 1: Quick Population Dashboard (US-1)
*As a viewer operator, I want population controls prominently hosted on the Quick workbench page so that I can immediately toggle, stream, and configure authentic world spawns without digging into editor placement menus.*

- **AC-1.1**: The Quick workbench tab (`WorkbenchTab.Quick`) features a dedicated, collapsible section: **"World Population (NPCs & Game Objects)"**.
- **AC-1.2**: Operators can switch dataset versions (`Auto`, `3.3.5a TrinityCore`, `0.5.3 Alpha-Core`, `Custom`), toggle auto-streaming, adjust AOI radius, toggle NPC/GO categories, and trigger reload directly from the Quick page.
- **AC-1.3**: Status text displays the active canonical map name and live streamed spawn counts (e.g., `Loaded 221/221 spawns for Coilfang: The Underbog (546)`).

### User Story 2: Authentic WoW Chat Window Overlay & Proximity Barks (US-2)
*As a viewer operator exploring the world, I want nearby NPCs to speak and bark ambient dialogue into an authentic WoW-style chat window when the camera approaches within ~40 yards, bringing the world to life.*

- **AC-2.1**: The viewer renders a lightweight, authentic WoW-style chat frame overlay in the bottom-left of the viewport with semi-transparent dark background, styled text channels, and auto-scrolling log.
- **AC-2.2**: When the camera approaches within $\le 40$ yards (configurable) of an NPC spawn that has registered dialogue text, the NPC barks a line into the chat window (e.g., `[Guard] says: For the Alliance!`, `[Vendor] says: Warm greetings, friend.`).
- **AC-2.3**: Bark throttling applies a per-creature cooldown (minimum 20 seconds before the same NPC speaks again) and a global cadence limiter (minimum 3 seconds between barks) to prevent chat flood.
- **AC-2.4**: Chat window toggle and proximity bark settings (enable/disable, range slider, clear log button) are accessible via the Quick page and chat frame controls.

### User Story 3: Quest & Gossip Flavor in Inspector (US-3)
*As a lore and content researcher, when I select an NPC or Game Object in the 3D scene, I want to see the quests it offers and its gossip/flavor text displayed in the Inspector.*

- **AC-3.1**: Inspecting an NPC or Game Object with an associated template entry queries the active version's dialogue and quest starter tables.
- **AC-3.2**: A dedicated section **"Quests Offered"** displays each quest title, quest level, objectives summary, and flavor text (`LogDescription`, `QuestDescription`, or `Details`).
- **AC-3.3**: A dedicated section **"Dialogue & Gossip"** displays greeting texts (`npc_text` / `broadcast_text`) and possible creature barks.
- **AC-3.4**: If no quests or gossip are associated with the selected template, the sections cleanly display `None recorded` or collapse.

### User Story 4: Strict Version Isolation & Multi-Version Database Architecture (US-4)
*As a systems engineer, I need 0.5.3 Alpha data to strictly source from Alpha-Core databases without mixing in 3.3.5a tables, while establishing clear integration blueprints for other historical server databases.*

- **AC-4.1**: 0.5.3 Alpha dialogue and quest data are extracted exclusively from `alpha-core` (`world.sql`), never falling back to TrinityCore 3.3.5a SQL or JSON.
- **AC-4.2**: 3.3.5a dialogue and quest data are extracted from TrinityCore TDB (`TDB_full_world_335...`).
- **AC-4.3**: Database schemas and extraction contracts are documented for target historical versions: 1.12.1 (`vmangos`), 2.4.3 (`cmangos-tbc`), and 4.3.4 (`TrinityCore 4.3.4`).

### User Story 5: WL* and LIT Format Clarification & Hardening (US-5)
*As a format researcher, I need remaining decoding gaps in WL* (liquid) and LIT (lighting) files documented with clear schema specifications and safety boundaries.*

- **AC-5.1**: Document the specification for `WlBlock.Data` (the 80 `ushort` attribute fields: 16 corner depths, 32 quad flags, and 32 UV/liquid flow parameters).
- **AC-5.2**: Document the runtime lighting boundary for `.lit` files (difference between parser summary/spatial lookup and runtime OpenGL skybox/ambient color ownership).

---

## 3. Constraints & Architectural Boundaries
- **No new members in WorldScene or ViewerApp** (§10 God-Class Freeze). All new UI, streaming, and chat features live in dedicated service classes (`FakeChatWindowService`, `WorldDialogueService`, etc.) receiving dependencies through `IViewerAppHost`.
- **Core Library First** (§4): Data extraction and dialogue/quest models live in `WowViewer.Core` and `WowViewer.Core.IO`. UI components live in `WoWViewer`.
- **UI Standardization** (§11): All workbench controls use `SharedUiWidgets` styling.
- **Strict Version Separation**: Never mix 0.5.3 Alpha and 3.3.5a Wrath data.
