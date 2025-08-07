# PM4 File Hierarchy (High-Level)

The diagram below illustrates the relationships between the major PM4 chunks involved in building reconstruction.  It focuses on the *correct* data path we have identified: **MPRL → MSLK → MSUR (+ MSVT/MSPV/MSCN vertices)**.

```mermaid
flowchart TD
    %% --- Tile Context ---
    subgraph TILE[PM4 Tile]
        direction LR
        MPRL["MPRL\nPlacements\n(2,493 entries)"]
        MSLK["MSLK\nLinkage\n(12,820 entries)"]
        MSUR["MSUR\nSurface Groups\n(~6k groups)"]
        MSVT[("MSVT\nVertices")]
        MSPV[("MSPV\nVertices")]
        MSCN[("MSCN\nConn. Vertices")]
    end

    %% --- Primary Relationships ---
    MPRL -->|Unknown4 = ParentId| MSLK
    MSLK -->|SurfaceRefIndex| MSUR
    MSUR -->|Faces use indices| MSVT
    MSUR --> MSPV
    MSUR --> MSCN

    %% --- Building-scale Assembly ---
    subgraph BUILDING_GROUP[Example Building (Unknown4 = 0x1A3C)]
        direction TB
        B_MPRL["Placement\nMPRL idx 172"]
        B_MSLK1["MSLK ParentId=0x1A3C\nentry 1"]
        B_MSLK2["MSLK ParentId=0x1A3C\nentry 2"]
        B_SURF1["MSUR key 456"]
        B_SURF2["MSUR key 789"]
    end

    %% --- Example Links ---
    B_MPRL -- Unknown4 --> B_MSLK1 & B_MSLK2
    B_MSLK1 -- SurfaceRefIndex --> B_SURF1
    B_MSLK2 -- SurfaceRefIndex --> B_SURF2

    %% --- Cross-tile Note ---
    classDef note fill:#fff5b1,stroke:#b59f00,stroke-dasharray: 4 2;
    CROSS_NOTE[("Cross-tile links: some MSLK entries reference vertices in adjacent tiles → need global loader")]
    class CROSS_NOTE note
    MSLK --- CROSS_NOTE
```

## Legend
* **MPRL.Unknown4**  – unique building identifier (root of hierarchy).
* **MSLK.ParentIndex** – matches `Unknown4`; groups surface fragments for that building.
* **MSLK.SurfaceRefIndex** – points into `MSUR` surface array.
* **MSUR** – stores face indices referencing global vertex pools (`MSVT`, `MSPV`, `MSCN`).
* Dashed call-out highlights *cross-tile* references responsible for out-of-bounds indices.

---

### Reading the Graph
1. Start at an `MPRL` placement (left).  Its **Unknown4** field is the building key.
2. Find all `MSLK` rows whose **ParentIndex** equals that key.
3. Each of those links provides a **SurfaceRefIndex** into `MSUR` which contains the faces.
4. Faces use vertex indices that may come from regular (`MSVT`), parametric (`MSPV`), or connecting (`MSCN`) pools — potentially spanning tiles.

> **Why duplicates today?**  We are mistakenly treating *every* `MSLK` entry as a separate export unit instead of aggregating by `Unknown4`; hence thousands of near-identical OBJ files. The refactor will group as shown in the *Building Group* subgraph.
