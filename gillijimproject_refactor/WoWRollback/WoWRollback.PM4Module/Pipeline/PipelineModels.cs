// PM4 Pipeline Data Models
// Clean data structures for the PM4 to ADT conversion pipeline
// Part of the PM4 Clean Reimplementation

using System.Numerics;

namespace WoWRollback.PM4Module.Pipeline;

#region Pipeline Configuration

/// <summary>
/// Configuration for the PM4 pipeline execution.
/// </summary>
public record PipelineConfig(
    /// <summary>Directory containing PM4 files</summary>
    string Pm4Directory,
    
    /// <summary>Directory to write patched ADTs</summary>
    string OutputDirectory,
    
    /// <summary>Path to WMO library cache JSON</summary>
    string? WmoLibraryPath = null,
    
    /// <summary>Path to M2 library cache JSON (future)</summary>
    string? M2LibraryPath = null,
    
    /// <summary>Directory containing Museum ADTs to use as base</summary>
    string? MuseumAdtDirectory = null,
    
    /// <summary>Enable M2 matching (experimental)</summary>
    bool EnableM2Matching = false,
    
    /// <summary>Size tolerance for WMO matching (0.15 = 15%)</summary>
    float SizeTolerance = 0.15f,
    
    /// <summary>Only process this specific tile (e.g., "30_22")</summary>
    string? SingleTile = null,
    
    /// <summary>Export CSV files for debugging</summary>
    bool ExportCsv = true,
    
    /// <summary>Export Debug WMO models for manual verification</summary>
    bool ExportDebugWmos = false,
    
    /// <summary>Dry run - don't write ADT files</summary>
    bool DryRun = false,
    
    /// <summary>Use global PM4 reader for cross-tile object support (recommended)</summary>
    bool UseGlobalReader = true
);

#endregion

#region PM4 Extraction Models

/// <summary>
/// A WMO candidate extracted from PM4 data, grouped by CK24.
/// </summary>
public record Pm4WmoCandidate(
    /// <summary>CK24 value (24-bit object grouping key from MSUR.PackedParams)</summary>
    uint CK24,
    
    /// <summary>Instance ID (derived from MdosIndex or sequence)</summary>
    int InstanceId,

    /// <summary>Tile X coordinate (0-63)</summary>
    int TileX,
    
    /// <summary>Tile Y coordinate (0-63)</summary>
    int TileY,
    
    /// <summary>Bounding box minimum (world coordinates)</summary>
    Vector3 BoundsMin,
    
    /// <summary>Bounding box maximum (world coordinates)</summary>
    Vector3 BoundsMax,
    
    /// <summary>Dominant wall angle in degrees (for rotation matching)</summary>
    float DominantAngle,
    
    /// <summary>Number of surfaces in this object</summary>
    int SurfaceCount,
    
    /// <summary>Total vertex count</summary>
    int VertexCount,
    
    /// <summary>CK24 Type byte (Byte2) - 0x40=interior, 0x80=exterior</summary>
    byte TypeFlags,
    
    /// <summary>MPRL-derived rotation in degrees (0-360), null if not available</summary>
    float? MprlRotationDegrees = null,
    
    /// <summary>MPRL-derived placement position (XYZ corrected), null if not available</summary>
    Vector3? MprlPosition = null,
    
    /// <summary>Complete debug geometry (Mesh only) for export</summary>
    List<Vector3>? DebugGeometry = null,
    
    /// <summary>Faces for debug geometry (triangles)</summary>
    List<int[]>? DebugFaces = null,
    
    /// <summary>MSCN (Scene Node) vertices for debug visualization</summary>
    List<Vector3>? DebugMscnVertices = null
)
{
    /// <summary>Calculate bounding box center</summary>
    public Vector3 Center => (BoundsMin + BoundsMax) / 2f;
    
    /// <summary>Calculate bounding box size</summary>
    public Vector3 Size => BoundsMax - BoundsMin;
    
    /// <summary>Get tile identifier string</summary>
    public string TileId => $"{TileX}_{TileY}";
    
    /// <summary>Whether this candidate has MPRL-derived rotation data</summary>
    public bool HasMprlRotation => MprlRotationDegrees.HasValue;
}

/// <summary>
/// An M2 candidate extracted from PM4 MSCN data (future implementation).
/// </summary>
public record Pm4M2Candidate(
    /// <summary>Position in world coordinates</summary>
    Vector3 Position,
    
    /// <summary>Tile X coordinate</summary>
    int TileX,
    
    /// <summary>Tile Y coordinate</summary>
    int TileY,
    
    /// <summary>Index within the MSCN chunk</summary>
    int MscnIndex
);

#endregion

#region WMO Library Models

/// <summary>
/// Cached WMO geometry statistics for matching.
/// </summary>
public record WmoLibraryEntry(
    /// <summary>Full path to WMO file (e.g., "World/wmo/...")</summary>
    string Path,
    
    /// <summary>Bounding box minimum</summary>
    Vector3 BoundsMin,
    
    /// <summary>Bounding box maximum</summary>
    Vector3 BoundsMax,
    
    /// <summary>Dominant wall angle in degrees</summary>
    float DominantAngle,
    
    /// <summary>Total surface count (walkable surfaces)</summary>
    int SurfaceCount,
    
    /// <summary>Total vertex count</summary>
    int VertexCount
)
{
    /// <summary>Calculate bounding box center</summary>
    public Vector3 Center => (BoundsMin + BoundsMax) / 2f;
    
    /// <summary>Calculate bounding box size (Width, Height, Depth)</summary>
    public Vector3 Size => BoundsMax - BoundsMin;
}

#endregion

#region Matching Models

/// <summary>
/// Result of matching a PM4 candidate to a WMO.
/// </summary>
public record WmoMatch(
    /// <summary>The matched WMO path</summary>
    string WmoPath,
    
    /// <summary>Position in ADT world coordinates (XZY swapped)</summary>
    Vector3 Position,
    
    /// <summary>Rotation in degrees (Pitch, Yaw, Roll) - XYZ not swapped</summary>
    Vector3 Rotation,
    
    /// <summary>Scale factor (1.0 = normal)</summary>
    float Scale,
    
    /// <summary>Confidence score (0.0 - 1.0)</summary>
    float ConfidenceScore,
    
    /// <summary>The PM4 candidate that was matched</summary>
    Pm4WmoCandidate SourceCandidate,
    
    /// <summary>The WMO library entry that matched</summary>
    WmoLibraryEntry MatchedEntry
);

/// <summary>
/// Result of matching a PM4 candidate to an M2 (future).
/// </summary>
public record M2Match(
    /// <summary>The matched M2 path</summary>
    string M2Path,
    
    /// <summary>Position in ADT world coordinates</summary>
    Vector3 Position,
    
    /// <summary>Rotation in degrees</summary>
    Vector3 Rotation,
    
    /// <summary>Scale factor</summary>
    float Scale
);

#endregion

#region ADT Entry Models

/// <summary>
/// MODF chunk entry for WMO placement in ADT.
/// Matches the 64-byte SMMapObjDef structure.
/// </summary>
public record ModfEntry(
    /// <summary>Index into MWMO name list</summary>
    int NameIndex,
    
    /// <summary>Unique ID (must be globally unique across all ADTs)</summary>
    uint UniqueId,
    
    /// <summary>Position in world coords - stored as XZY in ADT</summary>
    Vector3 Position,
    
    /// <summary>Rotation in degrees - stored as XYZ in ADT (no swap)</summary>
    Vector3 Rotation,
    
    /// <summary>Bounding box minimum</summary>
    Vector3 BoundsMin,
    
    /// <summary>Bounding box maximum</summary>
    Vector3 BoundsMax,
    
    /// <summary>Flags (e.g., 0x0001 = destroyable)</summary>
    ushort Flags = 0,
    
    /// <summary>Doodad set index</summary>
    ushort DoodadSet = 0,
    
    /// <summary>Name set index</summary>
    ushort NameSet = 0,
    
    /// <summary>Scale (3.3.5 uses 0, Legion+ uses scale/1024)</summary>
    ushort Scale = 0
);

/// <summary>
/// MDDF chunk entry for M2/doodad placement in ADT.
/// Matches the 36-byte SMDoodadDef structure.
/// </summary>
public record MddfEntry(
    /// <summary>Index into MMDX name list</summary>
    int NameIndex,
    
    /// <summary>Unique ID</summary>
    uint UniqueId,
    
    /// <summary>Position in world coords</summary>
    Vector3 Position,
    
    /// <summary>Rotation in degrees</summary>
    Vector3 Rotation,
    
    /// <summary>Scale (0-65535, where 1024 = 1.0)</summary>
    ushort Scale = 1024,
    
    /// <summary>Flags</summary>
    ushort Flags = 0
);

#endregion

#region Pipeline Results

/// <summary>
/// Result of processing a single ADT tile.
/// </summary>
public record TileResult(
    /// <summary>Tile identifier (e.g., "30_22")</summary>
    string TileId,
    
    /// <summary>Number of WMOs placed</summary>
    int WmosPlaced,
    
    /// <summary>Number of M2s placed</summary>
    int M2sPlaced,
    
    /// <summary>Whether the tile was successfully patched</summary>
    bool Success,
    
    /// <summary>Error message if failed</summary>
    string? Error = null
);

/// <summary>
/// Overall pipeline execution result.
/// </summary>
public record PipelineResult(
    /// <summary>Number of tiles processed</summary>
    int TilesProcessed,
    
    /// <summary>Total WMOs placed across all tiles</summary>
    int TotalWmosPlaced,
    
    /// <summary>Total M2s placed across all tiles</summary>
    int TotalM2sPlaced,
    
    /// <summary>Number of tiles that failed</summary>
    int FailedTiles,
    
    /// <summary>Per-tile results</summary>
    List<TileResult> TileResults,
    
    /// <summary>Global errors</summary>
    List<string> Errors,
    
    /// <summary>Execution time</summary>
    TimeSpan Duration
)
{
    /// <summary>Overall success (no errors)</summary>
    public bool Success => Errors.Count == 0 && FailedTiles == 0;
}

/// <summary>
/// Result of patching a single ADT file.
/// </summary>
public record PatchResult(
    /// <summary>Path to the output ADT file</summary>
    string OutputPath,
    
    /// <summary>Whether patching succeeded</summary>
    bool Success,
    
    /// <summary>Number of MODF entries written</summary>
    int ModfCount,
    
    /// <summary>Number of MDDF entries written</summary>
    int MddfCount,
    
    /// <summary>Error message if failed</summary>
    string? Error = null
);

#endregion
