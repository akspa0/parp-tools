# MdxViewer — WoW World Viewer

A high-performance .NET 9 / OpenGL 3.3 world viewer for **World of Warcraft Alpha 0.5.3**, **0.6.0**, and **Lich King 3.3.5** game data. Renders terrain, WMO world objects, MDX/M2 models, liquids, and DBC-driven overlays with high fidelity.

## Key Features

### Terrain & World
- **Alpha 0.5.3 WDT** — Monolithic WDT with 256 MCNK chunks per tile.
- **0.6.0 Split ADTs** — Per-tile ADT files with reversed FourCC, MCNK header offsets, packed MCLQ instances.
- **0.6.0 WMO-Only Maps** — Instance dungeons/battlegrounds via WDT-level MWMO+MODF.
- **3.3.5 Split ADTs** — Standard root/obj/tex ADT files from MPQ (partial — loading freeze under investigation).
- **AOI Streaming** — 9×9 tile area-of-interest with directional lookahead, persistent tile cache, and throttled MPQ reads.
- **MCSH Shadows** — 64×64 shadow bitmasks applied across all terrain layers.

### Liquids
- **MCLQ** (0.5.3/0.6.0) — Per-vertex sloped heights for waterfalls, absolute world Z, packed multi-instance format.
- **MH2O** (3.3.5) — Per-tile liquid with sub-rect heightmaps and visibility masks.
- **MLIQ** (WMO) — WMO group liquids with `matId`-based type detection (Water, Ocean, Magma, Slime).

### WMO Rendering (v14, v16 & v17)
- **4-Pass Transparency** — Opaque → Doodads → Liquids → Transparent.
- **100% Doodad Load Rate** — Case-insensitive MPQ search with `.mdx`/`.mdl` extension swapping.
- **Doodad Culling** — Distance (500u) + cap (64) + nearest-first sort for performance.
- **Doodad Sets** — Full support for switching between internal WMO doodad configurations.

### MDX/M2 Models
- **Two-Pass Rendering** — Opaque pass followed by depth-sorted transparent pass.
- **Blend Modes 0-6** — All 7 standard WoW blend modes (Opaque, AlphaKey, Alpha, Additive, etc.).
- **Alpha Cutout** — Hard discard for tree canopies in opaque pass.
- **DBCD Texture Resolution** — Resolves textures using CreatureModelData, CreatureDisplayInfo, CreatureDisplayInfoExtra, and ItemDisplayInfo DBC tables.

### Minimap & Overlays
- **Minimap** — Camera-centered with BLP tile textures, scroll-wheel zoom, and double-click teleport.
- **Area POIs** — DBC-driven point-of-interest markers as 3D pins and minimap dots.
- **Taxi Paths** — Flight path visualization from TaxiPath/TaxiPathNode DBC data.
- **Area Names** — Live area name display from AreaTable.dbc with MapID validation.
- **Zone Lighting** — DBC-driven ambient/fog/sky colors from Light.dbc + LightData.dbc.
- **SQL World Population** — Optional NPC/GameObject placement injection from alpha-core SQL dumps.

## Controls

| Input | Action |
|-------|--------|
| **WASD** | Move camera (North/South/East/West) |
| **Q / E** | Move camera Vertical (Down / Up) |
| **Right Drag** | Look around (Yaw / Pitch) |
| **Scroll Wheel** | Adjust camera speed (viewport) / Minimap zoom (minimap) |
| **Double-Click Minimap** | Teleport camera to clicked location |
| **Day/Night Slider** | Adjust world lighting (Terrain/WMO) |

## Requirements

- **Runtime**: .NET 10.0 SDK
- **GPU**: OpenGL 3.3+ capable hardware
- **OS**: Windows x64 (other platforms untested)

## Building from Source

### 1. Clone the repo
```bash
git clone https://github.com/akspa0/parp-tools.git
cd parp-tools/gillijimproject_refactor
git submodule update --init --recursive
```

This initializes the alpha-core dependency at `external/alpha-core` used by SQL spawn injection.

### 2. Bootstrap library dependencies
External libraries (SereniaBLPLib, DBCD, WoWDBDefs) are not git submodules — they are cloned on demand by the bootstrap script.

**PowerShell (Windows):**
```powershell
./setup-libs.ps1
```

**Bash (Linux/macOS):**
```bash
chmod +x setup-libs.sh
./setup-libs.sh
```

Re-run with `-Force` (PowerShell) or `--force` (bash) to re-clone all libraries.

### 3. Build & Run
```bash
cd src/MdxViewer
dotnet build
dotnet run -- path/to/game/directory
```

The viewer auto-detects the WoW build version from the game path and loads the appropriate terrain adapter.

## SQL World Population (alpha-core)

MdxViewer can inject NPC and GameObject spawns from alpha-core SQL files.

- Expected root: `external/alpha-core`
- Required files:
  - `etc/databases/world/world.sql`
  - `etc/databases/dbc/dbc.sql`

In the viewer:

1. Open **World Objects → SQL World Population**.
2. Set SQL root (or click **Use Submodule Path**).
3. Click **Load SQL Spawns (Current Map)**.
4. Click a SQL GameObject in the scene to get **SQL GameObject Animation** controls in the right sidebar (play/pause, sequence, keyframe stepping, frame scrub).

### Pre-built Releases
Download self-contained binaries from [Releases](../../releases) — no .NET SDK or library setup required.

## Architecture

The viewer is built on a modular adapter pattern:

### Core Abstractions
- **IDataSource** — MPQ archives (`MpqDataSource`) or loose files.
- **ITerrainAdapter** — Unified interface for terrain loading:
  - `AlphaTerrainAdapter` — Alpha 0.5.3 monolithic WDT
  - `StandardTerrainAdapter` — 0.6.0 / 3.3.5 split ADTs + WMO-only maps

### Rendering Pipeline
- **TerrainManager** — AOI streaming with persistent cache, MPQ throttling (`SemaphoreSlim(4)`), directional priority loading.
- **WorldScene** — Placement transforms, instance management, frustum culling.
- **WmoRenderer** — WMO GPU rendering (4-pass, doodad culling, MLIQ liquid).
- **ModelRenderer** — MDX GPU rendering (two-pass, blend modes, alpha cutout).
- **LiquidRenderer** — MCLQ/MLIQ/MH2O liquid mesh rendering.

### Services
- **WorldAssetManager** — Centralized caching for WMO and MDX/M2 geometry.
- **ReplaceableTextureResolver** — DBCD-backed dynamic texture ID → BLP path resolution.
- **LightService** — Zone-based lighting from Light.dbc + LightData.dbc.
- **AreaTableService** — MapID-aware area name lookups.

## Supported Formats

| Format | Version | Status |
|--------|---------|--------|
| **WDT** | 0.5.3 / 0.6.0 / 3.3.5 | ✅ Fully supported |
| **ADT** | 0.5.3 / 0.6.0 / 3.3.5 | ✅ (3.3.5 has loading freeze) |
| **WMO** | v14, v16, v17 | ✅ Fully supported |
| **MDX** | v1300+ | ✅ Supported (animation WIP) |
| **M2** | v264+ | 🔧 Partial support |
| **DBC** | 0.5.3 / 0.6.0 / 3.3.5 | ✅ AreaTable, AreaPOI, TaxiPath, Light, Map |
| **GLB** | Export | ✅ MDX/WMO export |

## Coordinate System (Ghidra Verified)

- **WoW Coords**: X=North, Y=West, Z=Up (Right-handed, Direct3D CW winding).
- **Renderer Coords**: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`, `rendererZ = wowZ`.
- **WMO-Only Maps**: Raw WoW world coords (no MapOrigin conversion).
- **GPU Fix**: Reverse triangle winding at upload (CW→CCW) + 180° Z rotation in placement transforms.
