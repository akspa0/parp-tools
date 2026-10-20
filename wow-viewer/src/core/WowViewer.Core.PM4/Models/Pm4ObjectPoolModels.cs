namespace WowViewer.Core.PM4.Models;

/// <summary>Operator-authored expectations, never decoded facts or confirmed matches.</summary>
public sealed record Pm4ObjectPoolManifest(
    int Version, string MapName, IReadOnlyList<Pm4ObjectPool> Pools,
    IReadOnlyList<Pm4ObjectPoolAssignment> Assignments,
    IReadOnlyList<Pm4DesignKitAlias>? HintAliases = null,
    IReadOnlyList<Pm4DesignKitSourceHint>? SourceHints = null);

public sealed record Pm4DesignKitAlias(string Token, string Zone, string DesignKit);
public sealed record Pm4DesignKitSourceHint(
    string SourcePath, IReadOnlyList<string> MatchedAliases,
    IReadOnlyList<string> SuggestedZones, IReadOnlyList<string> SuggestedDesignKits,
    string TileAssociation, bool OperatorConfirmed);

public sealed record Pm4ObjectPool(
    string Id, string DesignKit, IReadOnlyList<string> ObjectClasses,
    IReadOnlyList<string> AssetKinds, IReadOnlyList<string> AssetPaths,
    IReadOnlyList<string> PathPrefixes, string Note);

public sealed record Pm4ObjectPoolAssignment(
    string PoolId, int MinTileX, int MinTileY, int MaxTileX, int MaxTileY, string Note);

public sealed record Pm4ObjectPoolSelection(
    string MapName, int TileX, int TileY, bool Restricted,
    IReadOnlyList<string> ActivePools, int InputCandidates, int EligibleCandidates,
    SurfaceCorrelationDatabase Database);
