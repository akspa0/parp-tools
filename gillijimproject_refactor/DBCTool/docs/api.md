# DBCTool Remap File API (`remap.json`)

This document describes the structure and semantics of the `.remap.json` files generated and consumed by `DBCTool`. These files provide a deterministic way to map AreaIDs from one game client build to another (e.g., from version `0.5.3` to `3.3.5`).

## File Structure

The JSON file consists of five top-level objects:

```json
{
  "meta": { ... },
  "aliases": { ... },
  "explicit_map": [ ... ],
  "ignore_area_numbers": [ ... ],
  "options": { ... }
}
```

---

### 1. `meta` Object

Contains metadata about the file's generation context.

-   `src_alias` (string): The short alias for the source build (e.g., `"0.5.3"`).
-   `src_build` (string): The canonical, full build version of the source client (e.g., `"0.5.3.3368"`).
-   `tgt_build` (string): The canonical, full build version of the target client (e.g., `"3.3.5.12340"`).
-   `generated_at` (string): The UTC timestamp (ISO 8601 format) when the file was generated.

**Example:**
```json
"meta": {
  "src_alias": "0.5.3",
  "src_build": "0.5.3.3368",
  "tgt_build": "3.3.5.12340",
  "generated_at": "2025-09-12T20:13:41.9670375Z"
}
```

---

### 2. `aliases` Object

Provides a dictionary for mapping a normalized, lowercase area name to a list of alternative names. This is used during the name-matching process to handle known name changes or variations between builds.

-   **Key** (string): The normalized, canonical name.
-   **Value** (array of strings): A list of other names that should be treated as equivalent to the key.

**Example:**
```json
"aliases": {
  "demonic stronghold": [
    "dreadmaul hold"
  ],
  "shadowfang": [
    "shadowfang keep"
  ]
}
```

---

### 3. `explicit_map` Array

An array of objects that defines a direct, forced mapping from a source area to a target area. These mappings take precedence over any automated matching logic.

-   `src_areaNumber` (integer): The unique Area ID from the source build's `AreaTable.dbc`.
-   `tgt_areaID` (integer): The corresponding unique Area ID in the target build's `AreaTable.dbc`.
-   `note` (string, optional): A comment explaining why the mapping was made (e.g., `"name+map"`, `"fuzzy+map"`, `"manual_override"`).

**Example:**
```json
"explicit_map": [
  {
    "src_areaNumber": 1048576,
    "tgt_areaID": 1,
    "note": "name+map"
  }
]
```

---

### 4. `ignore_area_numbers` Array

An array of integers representing source `AreaNumber` values that should be completely excluded from the matching process. This is typically used for test areas, unused entries, or other placeholders that have no valid equivalent in the target build.

**Example:**
```json
"ignore_area_numbers": [
  1310729, // Collin's Test
  1310730  // Test
]
```

---

### 5. `options` Object

Contains boolean flags that control the behavior of the matching algorithm when the remap file is applied.

-   `disallow_do_not_use_targets` (boolean): If `true`, the tool will not map a source area to any target area whose name contains the phrase "DO NOT USE". This is the default behavior to avoid mapping to deprecated or invalid areas.

**Example:**
```json
"options": {
  "disallow_do_not_use_targets": true
}
```
