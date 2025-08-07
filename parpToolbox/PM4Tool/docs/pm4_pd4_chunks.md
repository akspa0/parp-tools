# PM4 and PD4 File Format Chunks

This document describes the known data chunks found within PM4 (Path Marker v4) and PD4 (Path Data v4) files, used for navigation and rendering data in World of Warcraft.

The files consist of a series of chunks, each identified by a 4-character code (magic number).

## Common Chunks (Found in both PM4 and PD4)

*   **`MVER` (Version)**
    *   Contains a single integer representing the file format version. Expected value is typically 4 for PM4/PD4.
    *   `uint Version`

*   **`MSHD` (Mesh Header)**
    *   Provides header information for the mesh data, often containing counts or offsets for other chunks.
    *   Structure: `MSHDChunk`
        *   `uint Flags`: Unknown flags.
        *   `uint OffsetMSVT`: Offset to the MSVT chunk from the beginning of the MSHD chunk data.
        *   `uint SizeMSVT`: Size of the MSVT chunk data.
        *   `uint OffsetMSVI`: Offset to the MSVI chunk from the beginning of the MSHD chunk data.
        *   `uint SizeMSVI`: Size of the MSVI chunk data.
        *   `uint OffsetMSUR`: Offset to the MSUR chunk from the beginning of the MSHD chunk data.
        *   `uint SizeMSUR`: Size of the MSUR chunk data.
        *   `uint OffsetMDOS`: Offset to the MDOS chunk from the beginning of the MSHD chunk data.
        *   `uint SizeMDOS`: Size of the MDOS chunk data.
        *   `uint OffsetMDSF`: Offset to the MDSF chunk from the beginning of the MSHD chunk data.
        *   `uint SizeMDSF`: Size of the MDSF chunk data.

*   **`MSVT` (Mesh Vertices)**
    *   Contains the raw vertex data for the mesh. Each vertex likely represents a point in 3D space.
    *   Structure: Array of `MSVTVertex`
        *   `float X`: X-coordinate.
        *   `float Y`: Y-coordinate.
        *   `float Z`: Z-coordinate.
    *   **Note:** When used for rendering (e.g., exporting to `.obj`), coordinates often need transformation. Based on current findings, the transformation `(Y, X, Z)` seems appropriate for OBJ export from `MSVT`.

*   **`MSVI` (Mesh Vertex Indices)**
    *   Contains indices into the `MSVT` chunk, defining the order in which vertices should be connected to form faces (triangles, quads, etc.).
    *   Structure: Array of `uint` representing 0-based indices into the `MSVT` vertex array.

*   **`MSUR` (Mesh Surfaces/Submeshes)**
    *   Defines surfaces or submeshes, referencing ranges within the `MSVI` chunk to specify which vertices/indices belong to this surface. May also contain flags or links to other data like destructible object states.
    *   Structure: Array of `MSUREntry`
        *   `ushort FlagsOrUnknown_0x00`: Flags or unknown data. Often observed as `0x12`.
        *   `ushort Unknown_0x02`: Unknown data. Observed values include 1, 2, 3. Might relate to material or surface type.
        *   `uint MsviFirstIndex`: The starting 0-based index within the `MSVI` array for this surface.
        *   `int IndexCount`: The number of indices (from `MsviFirstIndex`) belonging to this surface.
        *   `uint MdosIndex_DEPRECATED`: **Previously thought to be a direct index into MDOS, but this is incorrect.** Use `MDSF` for linking.

*   **`MDOS` (Mesh Destructible Object States)**
    *   Defines different states for destructible objects within the mesh. Often linked to specific surfaces via the `MDSF` chunk.
    *   Structure: Array of `MDOSEntry`
        *   `uint m_destructible_building_index`: An ID, likely referencing a `DestructibleBuilding.dbd` entry or similar identifier. `0` often indicates a non-destructible or default part.
        *   `byte destruction_state`: The state of destruction (e.g., 0 = intact, 1+ = different levels of destruction).
        *   `byte unknown_0x05`, `unknown_0x06`, `unknown_0x07`: Unknown padding or data.

*   **`MDSF` (Mesh Destructible Object State Face Mapping)**
    *   Provides the mapping between `MSUR` surfaces and `MDOS` states. This is the crucial link for determining which state applies to which surface.
    *   Structure: Array of `MDSFEntry`
        *   `uint msur_index`: The 0-based index of the `MSUR` entry this mapping applies to.
        *   `uint mdos_index`: The 0-based index of the `MDOS` entry linked to the specified `MSUR` entry.

*   **`MDBH` (Mesh Destructible Building Hash?)**
    *   Purpose not fully confirmed. Might contain hashes or identifiers related to the `DestructibleBuilding.dbd` entries referenced in `MDOS`. Seems to contain unique `uint` values.
    *   Structure: Array of `uint`.

## Path Chunks (Primarily PM4)

These chunks define the navigation paths and nodes.

*   **`MPRL` (Path Relations/Links?)**
    *   Defines relationships or links between path nodes. Contains indices referencing other path nodes.
    *   Structure: Array of `MPRLEntry`
        *   `int linked_node_index`: Index of a linked node within the `MSPV` (Path Vertices) array.
        *   `ushort unknown_0x04`: Unknown data.
        *   `ushort unknown_0x06`: Unknown data.

*   **`MPRR` (Path Relation References?)**
    *   References entries in the `MPRL` chunk, possibly grouping related links.
    *   Structure: Array of `uint` representing 0-based indices into the `MPRL` array.

*   **`MSPV` (Mesh/Path Vertices?)**
    *   Defines the nodes or vertices of the navigation path graph. Contains information about node type, links, and position references.
    *   Structure: Array of `MSPVAEntry` (Size: 12 bytes)
        *   `int Unknown_0x00`: Potentially the index of the first related `MPRR` entry or a count.
        *   `int NumPathRelations`: Number of `MPRR` entries associated with this node.
        *   `ushort Unknown_0x08`: Unknown flags or data. Observed values include 0x00, 0x01, 0x02. May represent node type (e.g., anchor, link).
        *   `ushort PositionAnchorIndex`: Index used to link this node to a 3D position, likely via `MSLK` and `MSPI`.

*   **`MSPI` (Mesh/Path Position Indices?)**
    *   Contains indices, likely referencing entries in the `MSVI` (Mesh Vertex Indices) chunk. Used as part of the chain to link path nodes (`MSPV`) to actual vertex positions (`MSVT`).
    *   Structure: Array of `uint` representing 0-based indices into the `MSVI` array.

*   **`MSLK` (Mesh/Path Links?)**
    *   Acts as an intermediate lookup table in the chain linking path nodes (`MSPV`) to vertex positions (`MSVT`).
    *   An `MSPV` entry's `PositionAnchorIndex` is used as a 0-based index into this `MSLK` array.
    *   The `uint` value *stored* at `MSLK[PositionAnchorIndex]` is the 0-based index needed to look up an entry in the `MSPI` array.
    *   Structure: Array of `uint`.

## Unknown/Other Chunks (PD4 Specific?)

*   **`MCRC` (Mesh CRC?)**
    *   Likely contains CRC checksums for data integrity verification, possibly for the entire file or specific chunks. Observed in PD4.
    *   Structure: Array of `uint`.

*   **`MSCN` (Mesh Scene?)**
    *   Purpose unknown. Observed in PD4. Content structure needs investigation. May relate to scene setup, lighting, or other metadata.
    *   Structure: Currently unknown. Appears to contain `float` values.

---

**Linking Path Nodes (MSPV) to 3D Coordinates (MSVT):**

The chain appears to be:
`MSPV Entry` -> `PositionAnchorIndex` -> `MSLK[PositionAnchorIndex]` -> `MSPI Index` -> `MSPI[MSPI Index]` -> `MSVI Index` -> `MSVI[MSVI Index]` -> `MSVT Index` -> `MSVT[MSVT Index]` -> `(X, Y, Z)` 