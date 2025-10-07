namespace WoWRollback.DbcModule;

/// <summary>
/// Result of DBC area table dump operation containing paths to generated CSV files.
/// </summary>
public sealed record DbcDumpResult(
    string SrcCsvPath,
    string TgtCsvPath,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Result of crosswalk generation containing paths to mapping artifacts.
/// </summary>
public sealed record CrosswalkResult(
    string MapsJsonPath,
    string CrosswalkV2Dir,
    string CrosswalkV3Dir,
    bool Success,
    string? ErrorMessage = null);
