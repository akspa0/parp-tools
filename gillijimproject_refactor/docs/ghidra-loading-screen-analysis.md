# Ghidra Reverse Engineering Guide: WoW 0.5.3 Alpha Loading Screen System

## Objective

Reverse-engineer the loading screen system in the World of Warcraft 0.5.3.3368 Alpha client binary (`WoWClient.exe`) to understand:

1. How the client selects which loading screen BLP to display for a given map
2. How the loading progress bar is rendered (position, size, colors, texture)
3. How loading progress percentage is calculated and updated
4. What UI elements are drawn on the loading screen (logo, text, bar frame, tips)
5. The exact render pipeline used during the loading phase (before the world is shown)

The goal is to clone the loading screen behavior in a custom OpenGL viewer that reads the same game data.

---

## Target Binary

- **Executable**: `WoWClient.exe` (Alpha 0.5.3, build 3368)
- **Architecture**: x86 (32-bit PE)
- **Engine**: Custom WoW engine (pre-release, no Lua UI system yet)
- **Rendering API**: Direct3D 8 (possibly D3D7 in this early build)
- **String encoding**: ASCII for file paths, localized strings in DBC

---

## Known Data Structures

### LoadingScreens.dbc

For build 0.5.3.3368, the layout is:

```
$id$ID<32>
Name        // string — human-readable name like "Azeroth"
FileName    // string — BLP path like "Interface\Glues\Loading\LoadScreenAzeroth.blp"
```

- **3 columns**: ID (int32), Name (string), FileName (string)
- The `FileName` field contains the full virtual path to the loading screen BLP texture
- BLP files are stored in the MPQ archives (typically `interface.MPQ` or `misc.MPQ`)

### Map.dbc

For build 0.5.3.3368 (layout range `0.9.0.3807-0.12.0.3988`), the relevant fields are:

```
$id$ID<32>
Directory           // string — map folder name like "Azeroth", "Kalimdor"
InstanceType<32>
PVP<32>
MapName_lang        // localized string — display name
AreaTableID<32>
MapDescription0_lang
MapDescription1_lang
LoadingScreenID<32> // → foreign key into LoadingScreens.dbc
```

The `LoadingScreenID` field links each map to its loading screen BLP via `LoadingScreens.dbc`.

### Loading Screen BLP Files

Expected paths (based on later client knowledge, verify in 0.5.3 MPQ):

```
Interface\Glues\Loading\LoadScreenAzeroth.blp
Interface\Glues\Loading\LoadScreenKalimdor.blp
Interface\Glues\Loading\LoadScreenDeadmines.blp
Interface\Glues\Loading\LoadScreenWailingCaverns.blp
... etc
```

These are 1024×768 (or 800×600) BLP textures displayed fullscreen during map loading.

### UI Textures (Progress Bar)

Look for these paths in the MPQ or string references:

```
Interface\Glues\Loading\LoadingBarBorder.blp
Interface\Glues\Loading\LoadingBar.blp
Interface\Glues\Loading\LoadingBarFill.blp
```

If these don't exist in 0.5.3, the progress bar may be rendered procedurally (colored quads).

---

## Search Strategy

### Phase 1: Find Loading Screen String References

1. **Open WoW.exe in Ghidra** and let auto-analysis complete
2. **Search → For Strings** — look for:
   - `"LoadingScreens"` — DBC table name used when loading the DBC
   - `"LoadScreen"` — partial match for BLP paths
   - `"Interface\\Glues\\Loading"` — the loading screen texture directory
   - `"LoadingBar"` — progress bar texture references
   - `"Loading..."` — any loading status text
   - `"Map.dbc"` or `"Map"` — where Map.dbc is loaded and LoadingScreenID is read

3. **For each string hit**, use **References → Find References to** to locate the code that uses it

### Phase 2: Trace the DBC Loading Chain

The client loads DBC files using a pattern like:

```c
// Pseudocode — the actual function signature varies
DBCStorage* LoadDBC(const char* dbcName, ...);
// e.g., LoadDBC("LoadingScreens");
```

1. Find the function that loads `"LoadingScreens"` DBC
2. Trace how it stores the result (likely a global `DBCStorage*` pointer)
3. Find where `LoadingScreenID` is read from the `Map.dbc` storage
4. Follow the code path: `MapID → Map.dbc row → LoadingScreenID → LoadingScreens.dbc row → FileName`

### Phase 3: Find the Loading Screen Render Function

Look for a function that:

1. Takes a BLP path or texture handle as input
2. Calls Direct3D `SetTexture` / `DrawPrimitive` to render a fullscreen quad
3. Is called during map transitions (not during normal gameplay rendering)

**Key indicators**:
- Calls to `IDirect3DDevice8::SetTexture` or `IDirect3DDevice8::DrawPrimitive`
- References to screen dimensions (800, 600, 1024, 768)
- A function that sets up an orthographic projection matrix (2D rendering mode)
- `Present()` / `EndScene()` calls — the loading screen must present frames during loading

### Phase 4: Find the Progress Bar Logic

The progress bar is the most important element to clone. Look for:

1. **A global float or int variable** that tracks loading progress (0.0→1.0 or 0→100)
2. **A function that updates this variable** — called from various loading stages:
   - ADT/WDT file loading
   - WMO loading
   - MDX/M2 loading
   - Texture loading
   - DBC loading
3. **A render function** that draws the bar proportional to the progress value

**Search for progress bar rendering**:
- Look for `DrawPrimitive` calls with vertex data that changes width based on a variable
- Look for a colored rectangle (likely gold/yellow fill, dark border) drawn at the bottom of the screen
- The bar is typically positioned at approximately:
  - **X**: centered, ~20% margin on each side
  - **Y**: bottom ~15% of screen
  - **Width**: proportional to progress (0% → 0px, 100% → full bar width)
  - **Height**: ~20-30 pixels

### Phase 5: Identify the Loading State Machine

The client has a state machine that controls transitions:

```
GLUE_SCREEN → LOADING_SCREEN → WORLD_RENDER
```

Look for:
- An enum or state variable with values like `STATE_LOADING`, `STATE_WORLD`, `STATE_GLUE`
- A main loop `switch` statement that dispatches to different render functions based on state
- The transition from loading → world (when does the loading screen disappear?)

---

## Specific Ghidra Techniques

### Finding DBC Accessor Functions

DBC rows are typically accessed via:

```c
// Pattern: get row by ID
void* GetDBCRow(DBCStorage* storage, int id);

// Pattern: get field from row
int GetInt(void* row, int fieldIndex);
const char* GetString(void* row, int fieldIndex);
```

To find these:
1. Locate a known DBC string reference (e.g., `"LoadingScreens"`)
2. The function that uses this string likely calls a generic `LoadDBC()` function
3. Follow the returned storage pointer to find `GetRow()` / `GetField()` accessor calls
4. The `LoadingScreenID` field is at column index ~8 in Map.dbc (count from 0)
5. The `FileName` field is at column index 2 in LoadingScreens.dbc

### Finding the Render Loop

1. Search for `IDirect3DDevice8::Present` (or the vtable offset for Present)
2. This is called once per frame — trace backwards to find the main render dispatch
3. During loading, the render function should:
   - Clear the screen
   - Draw the loading screen BLP as a fullscreen quad
   - Draw the progress bar
   - Draw any text (map name, tips)
   - Call Present()

### Finding Texture Loading

BLP textures are loaded via:

```c
TextureHandle LoadBLP(const char* path);
```

Search for string references to `.blp` or `Interface\\Glues` to find this function.

---

## Expected Findings to Document

After analysis, document the following for implementation:

### 1. Loading Screen Selection
```
Input:  MapID (from WDT/ADT being loaded)
Output: BLP virtual path for the loading screen texture

Steps:
1. Look up MapID in Map.dbc → get LoadingScreenID
2. Look up LoadingScreenID in LoadingScreens.dbc → get FileName
3. Load FileName as BLP texture from MPQ
```

### 2. Progress Bar Geometry
```
Bar position:   (x, y) in screen-space pixels or normalized coordinates
Bar dimensions: (width, height)
Bar colors:     fill color (RGBA), border color (RGBA), background color (RGBA)
Bar textures:   if textured, the BLP paths for border/fill
```

### 3. Progress Stages
```
Document what loading stages exist and their weight in the progress bar:
- Stage 1: WDT loading          (0% → X%)
- Stage 2: ADT tile loading     (X% → Y%)
- Stage 3: WMO loading          (Y% → Z%)
- Stage 4: MDX/M2 loading       (Z% → W%)
- Stage 5: Texture loading       (W% → 100%)
```

### 4. Text Rendering
```
- Is the map name displayed? Where? What font?
- Are loading tips shown? Where do they come from (DBC? hardcoded?)
- Is there a "Loading..." text? Position and style?
```

### 5. Timing and Transitions
```
- Is there a minimum display time for the loading screen?
- Is there a fade-in/fade-out transition?
- At what point does the loading screen disappear and the world render begin?
```

---

## Alpha 0.5.3 Specific Notes

- This is a **very early build** — the UI system may be minimal or hardcoded
- There is **no Lua/XML UI framework** yet (that came in later Alpha/Beta builds)
- The loading screen is likely rendered entirely in C++ with Direct3D calls
- The progress bar may be a simple colored quad, not a textured element
- There may be **no loading tips** — those were added later
- The `LoadingScreens.dbc` only has `Name` and `FileName` — no widescreen variants
- The resolution is likely fixed at **800×600** or **1024×768**

---

## Tools Required

- **Ghidra** (latest version with x86 decompiler)
- **Ghidra MCP server** (for programmatic access to decompilation results)
- The `WoW.exe` binary from the 0.5.3.3368 Alpha client
- MPQ archives from the Alpha client (to verify BLP paths and extract textures)

## MCP Server Commands to Use

If you have access to the Ghidra MCP server, use these commands:

```
# Search for loading-related functions
mcp2_search_functions_by_name("Loading")
mcp2_search_functions_by_name("LoadScreen")
mcp2_search_functions_by_name("Progress")

# Find string references
mcp2_list_strings(filter="LoadingScreen")
mcp2_list_strings(filter="LoadScreen")
mcp2_list_strings(filter="Interface\\Glues")
mcp2_list_strings(filter="LoadingBar")
mcp2_list_strings(filter=".dbc")

# Once you find relevant functions, decompile them
mcp2_decompile_function("FunctionName")

# Check cross-references to understand call chains
mcp2_get_function_xrefs("FunctionName")
```

---

## Deliverables

After completing the analysis, produce:

1. **Loading screen selection pseudocode** — exact DBC lookup chain
2. **Progress bar render specification** — position, size, colors, textures
3. **Progress calculation formula** — what stages exist and their weights
4. **State machine diagram** — loading screen lifecycle (show → update → hide)
5. **Any hardcoded constants** — screen positions, colors, timing values
6. **BLP paths** — all loading-related textures found in string references

This information will be used to implement an authentic loading screen in a custom OpenGL world viewer that reads the same 0.5.3 Alpha game data.
