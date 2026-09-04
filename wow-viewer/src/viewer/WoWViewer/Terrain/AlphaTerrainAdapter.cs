using System.Collections.Concurrent;
using System.Numerics;
using GillijimProject.WowFiles.Alpha;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WowViewer.Core.Audio;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

/// <summary>
/// Bridges Alpha WDT/ADT/MCNK parsed data into GPU-ready <see cref="TerrainChunkData"/>.
/// Handles the Alpha-specific non-interleaved vertex layout and coordinate system.
/// Reuses existing gillijimproject-csharp parsers (WdtAlpha, AdtAlpha, McnkAlpha).
/// </summary>
/// <summary>
/// Parsed MDDF placement entry (MDX/M2 doodad placement in world).
/// </summary>
public struct MddfPlacement
{
    public int NameIndex;   // Index into WDT MDNM name table
    public int UniqueId;    // For dedup across tiles
    public Vector3 Position;
    public Vector3 Rotation; // Degrees
    public float Scale;      // 1024 = 1.0 in Alpha
}

/// <summary>
/// Parsed MODF placement entry (WMO placement in world).
/// </summary>
public struct ModfPlacement
{
    public int NameIndex;   // Index into WDT MONM name table
    public int UniqueId;    // For dedup across tiles
    public Vector3 Position;
    public Vector3 Rotation; // Degrees
    public Vector3 BoundsMin;
    public Vector3 BoundsMax;
    public ushort Flags;
}

/// <summary>
/// Result of loading a single tile — terrain chunks + per-tile placements.
/// </summary>
public class TileLoadResult
{
    public List<TerrainChunkData> Chunks { get; init; } = new();
    public List<MddfPlacement> MddfPlacements { get; init; } = new();
    public List<ModfPlacement> ModfPlacements { get; init; } = new();
    public List<TerrainSoundEmitter> SoundEmitters { get; init; } = new();
}

public sealed record TerrainSoundEmitter(
    int TileX,
    int TileY,
    int ChunkX,
    int ChunkY,
    uint SoundPointId,
    uint SoundNameId,
    Vector3 RawPosition,
    Vector3 Position,
    float MinDistance,
    float MaxDistance,
    float CutoffDistance,
    uint StartTime,
    uint EndTime,
    uint Mode,
    byte[] RawEntry,
    byte LoopCountMin = 0,
    byte LoopCountMax = 0,
    ushort GroupSilenceMin = 0,
    ushort GroupSilenceMax = 0,
    ushort PlayInstancesMin = 0,
    ushort PlayInstancesMax = 0,
    ushort InterSoundGapMin = 0,
    ushort InterSoundGapMax = 0,
    AudioTriggerKind TriggerKind = AudioTriggerKind.Mcse,
    uint McnkFlags = 0,
    int LiquidFamily = -1,
    int SoundWaterSubtype = 0,
    string CoordinateProfile = "TerrainSoundEmitter.RawPosition -> renderer world");

public class AlphaTerrainAdapter : ITerrainAdapter
{
    private readonly string _wdtPath;
    private readonly WdtAlpha _wdt;
    private readonly List<int> _existingTiles;
    private readonly List<int> _adtOffsets;

    /// <summary>Texture names referenced across all loaded tiles (MTEX).</summary>
    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();

    private readonly List<string> _mdxModelNames;
    private readonly List<string> _wmoModelNames;

    /// <summary>
    /// MDX model name table from the base WDT MDNM, extended with phase-only model paths as they
    /// are composed.
    /// </summary>
    public IReadOnlyList<string> MdxModelNames => _mdxModelNames;

    /// <summary>
    /// WMO model name table from the base WDT MONM, extended with phase-only model paths as they
    /// are composed.
    /// </summary>
    public IReadOnlyList<string> WmoModelNames => _wmoModelNames;

    /// <summary>Collected MDDF placements from all loaded tiles (deduplicated by uniqueId).</summary>
    public List<MddfPlacement> MddfPlacements { get; } = new();

    /// <summary>Collected MODF placements from all loaded tiles (deduplicated by uniqueId).</summary>
    public List<ModfPlacement> ModfPlacements { get; } = new();

    private readonly object _placementLock = new();

    /// <summary>WorldPosition of every loaded chunk (for diagnostics).</summary>
    public List<Vector3> LastLoadedChunkPositions { get; } = new();

    /// <summary>True if this is a WMO-only map (no terrain tiles).</summary>
    public bool IsWmoBased { get; }

    /// <summary>
    /// The phase overlay stack. Each layer is a second alpha WDT, read through its own adapter.
    /// </summary>
    private readonly List<PhaseLayerSettings> _phaseLayers = [];

    /// <summary>Adapters for phase maps, cached per map name. A cached null is a cached failure.</summary>
    private readonly Dictionary<string, AlphaTerrainAdapter?> _phaseAdapters = new(StringComparer.OrdinalIgnoreCase);

    // Tile loads run repeatedly while streaming. Keep phase admission diagnostics actionable
    // without emitting one line for every visible tile. Keys include the current layer settings,
    // so editing a layer naturally produces a new diagnostic on the next reload.
    private readonly ConcurrentDictionary<string, byte> _phaseDiagnosticOnce = new(StringComparer.Ordinal);

    /// <summary>
    /// Resolves a phase map name to a readable alpha WDT path.
    /// </summary>
    /// <remarks>
    /// REQUIRED for phase layers to work, because <see cref="_wdtPath"/> is NOT
    /// <c>World/Maps/&lt;map&gt;/&lt;map&gt;.wdt</c>. The viewer extracts archive-backed WDTs to a
    /// flat cache directory and hands the adapter that path, so probing for a sibling map directory
    /// finds nothing -- which is exactly how phase layers on alpha maps silently did nothing. The
    /// host supplies this so phase maps are resolved through the same source the base map came from.
    /// </remarks>
    public Func<string, string?>? PhaseWdtPathResolver { get; set; }

    /// <inheritdoc />
    public IList<PhaseLayerSettings> PhaseLayers => _phaseLayers;

    /// <inheritdoc />
    public string? OverlayMapName
    {
        get => _phaseLayers.FirstOrDefault(static layer => layer.Enabled)?.MapName;
        set
        {
            _phaseLayers.Clear();
            if (!string.IsNullOrWhiteSpace(value))
                _phaseLayers.Add(new PhaseLayerSettings { MapName = value.Trim() });
        }
    }

    public AlphaTerrainAdapter(string wdtPath)
    {
        _wdtPath = wdtPath;
        _wdt = new WdtAlpha(wdtPath);
        _existingTiles = _wdt.GetExistingAdtsNumbers();
        _adtOffsets = _wdt.GetAdtOffsetsInMain();
        _mdxModelNames = _wdt.GetMdnmFileNames();
        _wmoModelNames = _wdt.GetMonmFileNames();
        IsWmoBased = _wdt.IsWmoBased || _existingTiles.Count == 0;

        // Collect WDT-level MODF placement whenever available
        var wdtModf = _wdt.GetWdtModfRaw();
        if (wdtModf.Length > 0)
        {
            CollectModfPlacements(wdtModf);
            ViewerLog.Trace($"[TerrainAdapter] WDT MODF: {ModfPlacements.Count} WMO placements from WDT header");
        }

        ViewerLog.Trace($"[TerrainAdapter] WDT loaded: {_existingTiles.Count} tiles, {MdxModelNames.Count} MDX names, {WmoModelNames.Count} WMO names, wmoBased={IsWmoBased}");

        // Placements are now collected per-tile in LoadTileWithPlacements() for lazy loading.
        // No upfront PreScan needed — placements stream in as tiles enter AOI.
    }

    /// <summary>
    /// Pre-scan all ADTs to collect MDDF/MODF placements without loading terrain geometry.
    /// This allows the asset manifest to be built before any tiles are loaded.
    /// </summary>
    public void PreScanPlacements()
    {
        int scanned = 0;
        foreach (int tileIdx in _existingTiles)
        {
            if (tileIdx < 0 || tileIdx >= _adtOffsets.Count || _adtOffsets[tileIdx] == 0) continue;
            try
            {
                var adt = new AdtAlpha(_wdtPath, _adtOffsets[tileIdx], tileIdx);
                CollectMddfPlacements(adt.GetMddfRaw());
                CollectModfPlacements(adt.GetModfRaw());
                scanned++;
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[TerrainAdapter] PreScan error tile {tileIdx}: {ex.Message}");
            }
        }
        ViewerLog.Trace($"[TerrainAdapter] PreScan: {scanned} tiles → {MddfPlacements.Count} MDDF, {ModfPlacements.Count} MODF placements");
    }

    /// <summary>
    /// Returns the list of existing tile numbers (index = y*64+x).
    /// </summary>
    public IReadOnlyList<int> ExistingTiles => _existingTiles;

    /// <summary>
    /// Check if a tile exists at the given grid coordinates.
    /// </summary>
    public bool TileExists(int tileX, int tileY)
    {
        // Alpha WDT MAIN is row-major: index = tileX*64+tileY (where tileX is row and tileY is col)
        if (TileExistsInOwnWdt(tileX, tileY))
            return true;

        // A phase layer may supply a tile the base map does not have.
        foreach (PhaseLayerSettings layer in _phaseLayers)
        {
            if (!layer.Enabled || string.IsNullOrWhiteSpace(layer.MapName) || layer.Channels == PhaseDataChannel.None)
                continue;

            AlphaTerrainAdapter? phaseAdapter = ResolvePhaseAdapter(layer.MapName);
            if (phaseAdapter != null && phaseAdapter.TileExistsInOwnWdt(tileX - layer.TileOffsetX, tileY - layer.TileOffsetY))
                return true;
        }

        return false;
    }

    public bool TryGetPlacementSourceData(int tileX, int tileY, out string sourcePath, out byte[] sourceBytes)
    {
        sourcePath = string.Empty;
        sourceBytes = Array.Empty<byte>();
        return false;
    }

    public bool TryGetPlacementWritablePath(int tileX, int tileY, out string? fullPath)
    {
        fullPath = null;
        return false;
    }

    /// <summary>
    /// Load all 256 chunks for a given tile, returning GPU-ready chunk data.
    /// Uses AdtAlpha + McnkAlpha parsers from gillijimproject-csharp.
    /// </summary>
    public List<TerrainChunkData> LoadTile(int tileX, int tileY)
    {
        var result = LoadTileWithPlacements(tileX, tileY);
        return result.Chunks;
    }

    /// <summary>
    /// Load a tile and return terrain chunks + per-tile MDDF/MODF placements.
    /// Placements are collected into the returned TileLoadResult AND into the global lists.
    /// </summary>
    /// <summary>
    /// Load the tile and compose every active phase layer onto it.
    /// </summary>
    /// <remarks>
    /// Alpha WDTs keep their terrain inside the WDT rather than in loose ADTs, so a phase layer here
    /// is a second alpha WDT read through its own adapter. Composition then reuses the same
    /// <see cref="PhaseCompositionPolicy"/> rules as the split-ADT path, so the two eras cannot drift
    /// apart on what a channel means.
    /// </remarks>
    public TileLoadResult LoadTileWithPlacements(int tileX, int tileY)
    {
        TileLoadResult result = LoadTileCore(tileX, tileY);

        if (_phaseLayers.Count > 0)
        {
            string stackKey = string.Join("|", _phaseLayers.Select(layer =>
                $"{layer.MapName}:{layer.Enabled}:{(int)layer.Channels}:{layer.TileOffsetX}:{layer.TileOffsetY}"));
            if (_phaseDiagnosticOnce.TryAdd($"stack:{stackKey}", 0))
            {
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"[AlphaADT] Phase admission: base='{Path.GetFileNameWithoutExtension(_wdtPath)}' "
                    + $"layers={_phaseLayers.Count}; first requested tile=({tileX},{tileY}); "
                    + $"stack={stackKey}");
            }
        }

        foreach (PhaseLayerSettings layer in _phaseLayers)
        {
            if (!layer.Enabled || string.IsNullOrWhiteSpace(layer.MapName) || layer.Channels == PhaseDataChannel.None)
            {
                string rejectedKey = $"rejected:{layer.MapName}:{layer.Enabled}:{(int)layer.Channels}";
                if (_phaseDiagnosticOnce.TryAdd(rejectedKey, 0))
                {
                    ViewerLog.Important(ViewerLog.Category.Terrain,
                        $"[AlphaADT] Phase skipped before resolution: map='{layer.MapName}' "
                        + $"enabled={layer.Enabled} channels={PhaseCompositionPolicy.Describe(layer.Channels)}.");
                }
                continue;
            }

            AlphaTerrainAdapter? phaseAdapter = ResolvePhaseAdapter(layer.MapName);
            if (phaseAdapter == null)
            {
                string unresolvedKey = $"unresolved:{layer.MapName}";
                if (_phaseDiagnosticOnce.TryAdd(unresolvedKey, 0))
                {
                    ViewerLog.Important(ViewerLog.Category.Terrain,
                        $"[AlphaADT] Phase admitted but no adapter: map='{layer.MapName}'. "
                        + "See the preceding resolver diagnostic for every attempted WDT path.");
                }
                continue;
            }

            int sourceTileX = tileX - layer.TileOffsetX;
            int sourceTileY = tileY - layer.TileOffsetY;
            if (layer.HasTileOffset)
            {
                (float worldDx, float worldDy) = PhaseCompositionPolicy.TileOffsetToWorldTranslation(
                    layer.TileOffsetX,
                    layer.TileOffsetY,
                    WoWConstants.TileSize);
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"[AlphaADT] Phase offset mapping '{layer.MapName}': targetTile=({tileX},{tileY}) "
                    + $"offset=({layer.TileOffsetX},{layer.TileOffsetY}) "
                    + $"sourceTile=target-offset=({sourceTileX},{sourceTileY}) "
                    + $"placementDelta=({worldDx:F1},{worldDy:F1})");
            }
            if (!phaseAdapter.TileExistsInOwnWdt(sourceTileX, sourceTileY))
            {
                string missingTileKey = $"tile-miss:{layer.MapName}:{layer.TileOffsetX}:{layer.TileOffsetY}";
                if (_phaseDiagnosticOnce.TryAdd(missingTileKey, 0))
                {
                    ViewerLog.Important(ViewerLog.Category.Terrain,
                        $"[AlphaADT] Phase WDT opened but has no source tile for map='{layer.MapName}': "
                        + $"target=({tileX},{tileY}) source=({sourceTileX},{sourceTileY}) "
                        + $"offset=({layer.TileOffsetX},{layer.TileOffsetY}) ownTiles={phaseAdapter._existingTiles.Count}. "
                        + "This is an overlap/offset issue, not a phase-stack propagation failure.");
                }
                continue;
            }

            TileLoadResult phase = phaseAdapter.LoadTileCore(sourceTileX, sourceTileY);
            phaseAdapter.TileTextures.TryGetValue((sourceTileX, sourceTileY), out List<string>? phaseTextures);
            RemapPhasePlacementNameIndices(phase, phaseAdapter);
            MergePhaseTile(
                result,
                phase,
                phaseTextures ?? new List<string>(),
                phaseAdapter.MdxModelNames.Count,
                phaseAdapter.WmoModelNames.Count,
                layer,
                tileX,
                tileY);
        }

        return result;
    }

    /// <summary>
    /// Converts a phase WDT's local MDNM/MONM indices into this base adapter's combined name tables.
    /// </summary>
    /// <remarks>
    /// MDDF and MODF store an index, not a path. Each Alpha WDT owns its own name tables, so passing
    /// a phase's raw index through to the base adapter makes the placement resolve whichever unrelated
    /// entry happens to occupy the same slot in the base table.
    /// </remarks>
    private void RemapPhasePlacementNameIndices(TileLoadResult phase, AlphaTerrainAdapter phaseAdapter)
    {
        int remappedMddf = RemapPlacementNameIndices(
            phase.MddfPlacements,
            phaseAdapter.MdxModelNames,
            _mdxModelNames,
            "MDDF",
            static (placement, index) =>
            {
                placement.NameIndex = index;
                return placement;
            });
        int remappedModf = RemapPlacementNameIndices(
            phase.ModfPlacements,
            phaseAdapter.WmoModelNames,
            _wmoModelNames,
            "MODF",
            static (placement, index) =>
            {
                placement.NameIndex = index;
                return placement;
            });

        ViewerLog.Trace(
            $"[AlphaADT] Phase placement name remap: MDDF={remappedMddf}/{phase.MddfPlacements.Count} "
            + $"MODF={remappedModf}/{phase.ModfPlacements.Count} "
            + $"baseNames=(mdx:{_mdxModelNames.Count},wmo:{_wmoModelNames.Count})");
    }

    private static int RemapPlacementNameIndices<TPlacement>(
        List<TPlacement> placements,
        IReadOnlyList<string> phaseNames,
        List<string> baseNames,
        string placementKind,
        Func<TPlacement, int, TPlacement> withNameIndex)
        where TPlacement : struct
    {
        int remapped = 0;
        for (int i = 0; i < placements.Count; i++)
        {
            int phaseNameIndex = placements[i] switch
            {
                MddfPlacement mddf => mddf.NameIndex,
                ModfPlacement modf => modf.NameIndex,
                _ => -1,
            };
            if ((uint)phaseNameIndex >= (uint)phaseNames.Count)
                continue;

            string phasePath = phaseNames[phaseNameIndex];
            int baseNameIndex = FindOrAddName(baseNames, phasePath);
            placements[i] = withNameIndex(placements[i], baseNameIndex);
            remapped++;

            if (i < 3)
            {
                ViewerLog.Trace(
                    $"[AlphaADT] Phase {placementKind} name map: phaseIndex={phaseNameIndex} "
                    + $"path='{phasePath}' -> baseIndex={baseNameIndex}");
            }
        }

        return remapped;
    }

    private static int FindOrAddName(List<string> names, string path)
    {
        for (int i = 0; i < names.Count; i++)
        {
            if (string.Equals(names[i], path, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        names.Add(path);
        return names.Count - 1;
    }

    /// <summary>Resolve (and cache) the adapter for a phase map's own alpha WDT.</summary>
    private AlphaTerrainAdapter? ResolvePhaseAdapter(string mapName)
    {
        if (_phaseAdapters.TryGetValue(mapName, out AlphaTerrainAdapter? cached))
            return cached;

        AlphaTerrainAdapter? adapter = null;
        var attempted = new List<string>();
        try
        {
            // Preferred: ask the host, which can read the map out of the same archive the base map
            // came from. The sibling probe below only works for genuinely loose map directories.
            string? resolved = PhaseWdtPathResolver?.Invoke(mapName);
            if (!string.IsNullOrWhiteSpace(resolved))
                attempted.Add(resolved!);

            if (string.IsNullOrWhiteSpace(resolved) || !File.Exists(resolved))
            {
                string? mapsRoot = Path.GetDirectoryName(Path.GetDirectoryName(_wdtPath));
                if (!string.IsNullOrEmpty(mapsRoot))
                {
                    string candidate = Path.Combine(mapsRoot, mapName, mapName + ".wdt");
                    attempted.Add(candidate);
                    if (File.Exists(candidate))
                        resolved = candidate;
                }
            }

            if (!string.IsNullOrWhiteSpace(resolved) && File.Exists(resolved))
            {
                adapter = new AlphaTerrainAdapter(resolved!) { PhaseWdtPathResolver = PhaseWdtPathResolver };
                string occupiedTilePreview = string.Join(", ", adapter._existingTiles
                    .Take(16)
                    .Select(static index => $"({index / 64},{index % 64})"));
                if (adapter._existingTiles.Count > 16)
                    occupiedTilePreview += $", … +{adapter._existingTiles.Count - 16}";
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"[AlphaADT] Phase map '{mapName}' opened from '{resolved}' ({adapter._existingTiles.Count} tiles): "
                    + $"occupied tile coordinates={occupiedTilePreview}.");
            }
            else
            {
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"[AlphaADT] Phase map '{mapName}' could not be resolved to a WDT; the layer contributes nothing. "
                    + $"Tried: {(attempted.Count == 0 ? "<no candidates>" : string.Join(", ", attempted))}");
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[AlphaADT] Phase map '{mapName}' failed to open: {ex.Message}");
        }

        _phaseAdapters[mapName] = adapter;
        return adapter;
    }

    /// <summary>Tiles present in this adapter's own WDT, ignoring any phase layers it carries.</summary>
    private bool TileExistsInOwnWdt(int tileX, int tileY)
    {
        int idx = tileX * 64 + tileY;
        return idx >= 0 && idx < _adtOffsets.Count && _adtOffsets[idx] != 0;
    }

    /// <summary>Compose one phase layer's tile onto the base tile, channel by channel.</summary>
    private void MergePhaseTile(
        TileLoadResult parent,
        TileLoadResult phase,
        IReadOnlyList<string> phaseTextures,
        int phaseMdxNameCount,
        int phaseWmoNameCount,
        PhaseLayerSettings layer,
        int tileX,
        int tileY)
    {
        int baseMddfCount = parent.MddfPlacements.Count;
        int baseModfCount = parent.ModfPlacements.Count;
        TileTextures.TryGetValue((tileX, tileY), out List<string>? baseTextures);
        var mergedTextures = new List<string>(baseTextures ?? new List<string>());
        var textureIndices = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < mergedTextures.Count; i++)
            textureIndices.TryAdd(mergedTextures[i], i);

        int patched = 0;
        int added = 0;
        int skipped = 0;
        PhaseDataChannel contributed = PhaseDataChannel.None;

        foreach (TerrainChunkData phaseChunk in phase.Chunks)
        {
            TerrainChunkData? parentChunk = parent.Chunks.FirstOrDefault(candidate =>
                candidate.ChunkX == phaseChunk.ChunkX && candidate.ChunkY == phaseChunk.ChunkY);

            PhaseDataChannel present = PhaseChunkMerger.DescribePresence(phaseChunk);
            PhaseDataChannel take = PhaseCompositionPolicy.ResolveChannelsToTake(
                layer.Channels, present, layer.OnlyTakeWhatThePhaseCarries);

            if ((take & PhaseDataChannel.TextureLayers) != 0 || parentChunk == null)
                RemapPhaseTextureIndices(phaseChunk, phaseTextures, mergedTextures, textureIndices);

            if (parentChunk == null)
            {
                if (present == PhaseDataChannel.None)
                {
                    skipped++;
                    continue;
                }

                parent.Chunks.Add(phaseChunk);
                added++;
                contributed |= present;
                continue;
            }

            if (take == PhaseDataChannel.None)
            {
                skipped++;
                continue;
            }

            int parentIndex = parent.Chunks.IndexOf(parentChunk);
            parent.Chunks[parentIndex] = PhaseChunkMerger.Merge(parentChunk, phaseChunk, take);
            patched++;
            contributed |= take;
        }

        if (layer.HasTileOffset)
            TranslatePhasePlacements(phase, layer);

        bool phaseReplacesDoodads = PhaseCompositionPolicy.PhaseOwnsPlacements(
            layer.Channels,
            PhaseDataChannel.Doodads,
            phase.MddfPlacements.Count);
        if (phaseReplacesDoodads)
        {
            parent.MddfPlacements.Clear();
            parent.MddfPlacements.AddRange(phase.MddfPlacements);
            contributed |= PhaseDataChannel.Doodads;
        }

        bool phaseReplacesWorldObjects = PhaseCompositionPolicy.PhaseOwnsPlacements(
            layer.Channels,
            PhaseDataChannel.WorldObjects,
            phase.ModfPlacements.Count);
        if (phaseReplacesWorldObjects)
        {
            parent.ModfPlacements.Clear();
            parent.ModfPlacements.AddRange(phase.ModfPlacements);
            contributed |= PhaseDataChannel.WorldObjects;
        }

        TileTextures[(tileX, tileY)] = mergedTextures;

        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"[AlphaADT] Phase patch ({tileX},{tileY}) from '{layer.MapName}': "
            + $"phaseChunks={phase.Chunks.Count} patched={patched} added={added} skippedEmpty={skipped} "
            + $"channels={PhaseCompositionPolicy.Describe(contributed)} "
            + $"doodads=base:{baseMddfCount}/phase:{phase.MddfPlacements.Count}/result:{parent.MddfPlacements.Count}"
            + $"/{(phaseReplacesDoodads ? "replaced" : "preserved")} "
            + $"wmos=base:{baseModfCount}/phase:{phase.ModfPlacements.Count}/result:{parent.ModfPlacements.Count}"
            + $"/{(phaseReplacesWorldObjects ? "replaced" : "preserved")} "
            + $"names=base(mdx:{MdxModelNames.Count},wmo:{WmoModelNames.Count})"
            + $"/phase(mdx:{phaseMdxNameCount},wmo:{phaseWmoNameCount})"
            + (layer.HasTileOffset ? $" tileOffset=({layer.TileOffsetX},{layer.TileOffsetY})" : string.Empty));
    }

    private static void TranslatePhasePlacements(TileLoadResult phase, PhaseLayerSettings layer)
    {
        // One tile of offset = one ADT in the 64x64 grid = 533.33 yds. Despite its name,
        // WoWConstants.ChunkSize IS that span (cornerX = MapOrigin - tileX * ChunkSize below);
        // WoWConstants.TileSize is 16 ADTs and would overshoot placements 16x. The 2026-09-03
        // "TileSize fix" was a magnitude error and is reverted here; only the sign/label fix stands.
        (float dx, float dy) = PhaseCompositionPolicy.TileOffsetToWorldTranslation(
            layer.TileOffsetX, layer.TileOffsetY, WoWConstants.ChunkSize);

        for (int i = 0; i < phase.MddfPlacements.Count; i++)
        {
            MddfPlacement placement = phase.MddfPlacements[i];
            placement.Position = new Vector3(placement.Position.X + dx, placement.Position.Y + dy, placement.Position.Z);
            phase.MddfPlacements[i] = placement;
        }

        for (int i = 0; i < phase.ModfPlacements.Count; i++)
        {
            ModfPlacement placement = phase.ModfPlacements[i];
            placement.Position = new Vector3(placement.Position.X + dx, placement.Position.Y + dy, placement.Position.Z);
            phase.ModfPlacements[i] = placement;
        }
    }

    private static void RemapPhaseTextureIndices(
        TerrainChunkData phaseChunk,
        IReadOnlyList<string> phaseTextures,
        List<string> mergedTextures,
        Dictionary<string, int> textureIndices)
    {
        for (int layerIndex = 0; layerIndex < phaseChunk.Layers.Length; layerIndex++)
        {
            TerrainLayer chunkLayer = phaseChunk.Layers[layerIndex];
            if ((uint)chunkLayer.TextureIndex >= (uint)phaseTextures.Count)
                continue;

            string textureName = phaseTextures[chunkLayer.TextureIndex];
            if (!textureIndices.TryGetValue(textureName, out int mergedIndex))
            {
                mergedIndex = mergedTextures.Count;
                mergedTextures.Add(textureName);
                textureIndices.Add(textureName, mergedIndex);
            }

            chunkLayer.TextureIndex = mergedIndex;
            phaseChunk.Layers[layerIndex] = chunkLayer;
        }
    }

    private TileLoadResult LoadTileCore(int tileX, int tileY)
    {
        // Alpha WDT MAIN is row-major: index = tileX*64+tileY (where tileX is row and tileY is col)
        int tileIdx = tileX * 64 + tileY;
        if (tileIdx < 0 || tileIdx >= _adtOffsets.Count || _adtOffsets[tileIdx] == 0)
            return new TileLoadResult();

        // Use the existing AdtAlpha parser to get MCIN offsets and MTEX
        var adt = new AdtAlpha(_wdtPath, _adtOffsets[tileIdx], tileIdx);
        var mtexNames = adt.GetMtexTextureNames();
        TileTextures.TryAdd((tileX, tileY), mtexNames);

        var chunks = new List<TerrainChunkData>(256);
        var soundEmitters = new List<TerrainSoundEmitter>();

        // Use AdtAlpha's internal MCIN to get MCNK offsets (same pattern as ToAdtLk)
        var offsets = adt.GetMcnkOffsets();
        using var fs = File.OpenRead(_wdtPath);

        for (int i = 0; i < 256 && i < offsets.Count; i++)
        {
            int off = offsets[i];
            if (off <= 0) continue;

            try
            {
                var mcnk = new McnkAlpha(fs, off, headerSize: 0, adtNum: tileIdx);
                if (TryReadAlphaMcnkPayload(fs, off, out byte[]? mcnkPayload))
                {
                    AdtMcseData mcse = AdtMcseReader.ReadAlpha053Mcnk(mcnkPayload);
                    foreach (AdtMcseEmitter emitter in mcse.Emitters)
                    {
                        soundEmitters.Add(new TerrainSoundEmitter(
                            tileX,
                            tileY,
                            mcnk.IndexX,
                            mcnk.IndexY,
                            emitter.SoundPointId,
                            emitter.SoundNameId,
                            emitter.Position,
                            ConvertSoundPosition(
                                emitter.Position,
                                tileX,
                                tileY,
                                mcnk.IndexX,
                                mcnk.IndexY),
                            emitter.MinDistance,
                            emitter.MaxDistance,
                            emitter.CutoffDistance,
                            emitter.StartTime,
                            emitter.EndTime,
                            emitter.Mode,
                            emitter.RawEntry,
                            emitter.LoopCountMin,
                            emitter.LoopCountMax,
                            emitter.GroupSilenceMin,
                            emitter.GroupSilenceMax,
                            emitter.PlayInstancesMin,
                            emitter.PlayInstancesMax,
                            emitter.InterSoundGapMin,
                            emitter.InterSoundGapMax,
                            CoordinateProfile: "MCSE chunk-local -> owning chunk -> renderer world"));
                    }
                }
                var chunkData = ExtractChunkData(mcnk, tileX, tileY, tileIdx);
                if (chunkData != null)
                {
                    chunks.Add(chunkData);
                    LegacyLiquidSoundEmitterFactory.Append(soundEmitters, chunkData);
                }
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[TerrainAdapter] Error reading chunk {i} of tile ({tileX},{tileY}): {ex.Message}");
            }
        }

        // Collect MDDF/MODF placement entries from this ADT into per-tile lists (no dedup — always parse all)
        var tileMddf = new List<MddfPlacement>();
        var tileModf = new List<ModfPlacement>();
        ParseMddfEntries(adt.GetMddfRaw(), tileMddf);
        ParseModfEntries(adt.GetModfRaw(), tileModf);

        // Also add to global lists with dedup for backwards compat
        lock (_placementLock)
        {
            foreach (var p in tileMddf)
                MddfPlacements.Add(p);
            foreach (var p in tileModf)
                ModfPlacements.Add(p);
        }

        // Diagnostic: print tile corner position for first tile loaded
        if (LastLoadedChunkPositions.Count <= 256)
        {
            float cornerX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize;
            float cornerY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize;
            ViewerLog.Trace($"[TerrainAdapter] Tile ({tileX},{tileY}) corner=({cornerX:F1}, {cornerY:F1})  wowY={cornerX:F1}  wowX={cornerY:F1}");
        }

        ViewerLog.Trace($"[TerrainAdapter] Tile ({tileX},{tileY}): {chunks.Count} chunks, {mtexNames.Count} textures, {tileMddf.Count} MDDF, {tileModf.Count} MODF");
        return new TileLoadResult
        {
            Chunks = chunks,
            MddfPlacements = tileMddf,
            ModfPlacements = tileModf,
            SoundEmitters = soundEmitters
        };
    }

    private static bool TryReadAlphaMcnkPayload(FileStream stream, int chunkOffset, out byte[]? payload)
    {
        payload = null;
        if (chunkOffset < 0 || chunkOffset + 8 > stream.Length)
            return false;

        Span<byte> header = stackalloc byte[8];
        stream.Seek(chunkOffset, SeekOrigin.Begin);
        if (stream.Read(header) != header.Length)
            return false;

        int size = BitConverter.ToInt32(header.Slice(4, 4));
        if (size < 0 || chunkOffset + 8L + size > stream.Length)
            return false;

        payload = new byte[size];
        stream.Seek(chunkOffset + 8L, SeekOrigin.Begin);
        return stream.Read(payload, 0, size) == size;
    }

    private static Vector3 ConvertSoundPosition(
        Vector3 position,
        int tileX,
        int tileY,
        int chunkX,
        int chunkY)
    {
        // Alpha MCSE stores a chunk-local C3Vector. The existing terrain
        // convention is corner-minus-local with the X/Y axis swap.
        Vector3 chunkWorldPosition = TerrainCoordinateTransform.ChunkCorner(
            tileX,
            tileY,
            chunkX,
            chunkY,
            WoWConstants.MapOrigin,
            WoWConstants.ChunkSize,
            WoWConstants.ChunksPerTileEdge);
        return TerrainCoordinateTransform.FromChunkLocal(position, chunkWorldPosition);
    }

    /// <summary>
    /// Parse MDDF raw bytes into placement entries. Entry size = 36 bytes.
    /// Layout: nameIndex(4) uniqueId(4) posX(4) posZ(4) posY(4) rotX(4) rotZ(4) rotY(4) scale(2) flags(2)
    /// </summary>
    private bool _mddfDiagPrinted = false;
    private void CollectMddfPlacements(byte[] mddfData)
    {
        var temp = new List<MddfPlacement>();
        ParseMddfEntries(mddfData, temp);
        lock (_placementLock)
        {
            foreach (var p in temp)
                MddfPlacements.Add(p);
        }
    }

    /// <summary>
    /// Parse MDDF entries into a list WITHOUT dedup. Always returns all placements in the data.
    /// </summary>
    private void ParseMddfEntries(byte[] mddfData, List<MddfPlacement> target)
    {
        const int entrySize = 36;
        for (int off = 0; off + entrySize <= mddfData.Length; off += entrySize)
        {
            int nameIdx = BitConverter.ToInt32(mddfData, off);
            int uniqueId = BitConverter.ToInt32(mddfData, off + 4);

            // Raw floats at file offsets
            float rawX = BitConverter.ToSingle(mddfData, off + 8);
            float rawZ = BitConverter.ToSingle(mddfData, off + 12); // height
            float rawY = BitConverter.ToSingle(mddfData, off + 16);
            // C3Vector rotation stored as (X, Z, Y) in file — same layout as position
            float rotX = BitConverter.ToSingle(mddfData, off + 20);
            float rotZ = BitConverter.ToSingle(mddfData, off + 24);
            float rotY = BitConverter.ToSingle(mddfData, off + 28);
            ushort scale = BitConverter.ToUInt16(mddfData, off + 32);

            // Diagnostic: dump first 3 raw entries
            if (!_mddfDiagPrinted && MddfPlacements.Count < 3)
            {
                string name = nameIdx < MdxModelNames.Count ? Path.GetFileName(MdxModelNames[nameIdx]) : "?";
                ViewerLog.Trace($"[MDDF RAW] [{MddfPlacements.Count}] pos=({rawX:F2}, {rawZ:F2}, {rawY:F2}) rot=({rotX:F2}, {rotY:F2}, {rotZ:F2}) scale={scale}  model={name}");
                if (MddfPlacements.Count == 2) _mddfDiagPrinted = true;
            }

            // Convert to renderer coords: terrainX=wowY, terrainY=wowX (swap + subtract)
            // Rotation is stored as-is (X, Y, Z degrees) — no axis swap needed
            target.Add(new MddfPlacement
            {
                NameIndex = nameIdx,
                UniqueId = uniqueId,
                Position = new Vector3(
                    WoWConstants.MapOrigin - rawY,
                    WoWConstants.MapOrigin - rawX,
                    rawZ),
                Rotation = new Vector3(rotX, rotY, rotZ),
                Scale = scale / 1024f
            });
        }
    }

    /// <summary>
    /// Parse MODF raw bytes into placement entries. Entry size = 64 bytes.
    /// Layout: nameIndex(4) uniqueId(4) pos(12) rot(12) bbMin(12) bbMax(12) flags(2) doodadSet(2) nameSet(2) pad(2)
    /// </summary>
    private void CollectModfPlacements(byte[] modfData)
    {
        var temp = new List<ModfPlacement>();
        ParseModfEntries(modfData, temp);
        lock (_placementLock)
        {
            foreach (var p in temp)
                ModfPlacements.Add(p);
        }
    }

    /// <summary>
    /// Parse MODF entries into a list WITHOUT dedup. Always returns all placements in the data.
    /// </summary>
    private void ParseModfEntries(byte[] modfData, List<ModfPlacement> target)
    {
        const int entrySize = 64;
        for (int off = 0; off + entrySize <= modfData.Length; off += entrySize)
        {
            int nameIdx = BitConverter.ToInt32(modfData, off);
            int uniqueId = BitConverter.ToInt32(modfData, off + 4);

            // Raw floats from file
            float rawX = BitConverter.ToSingle(modfData, off + 8);
            float rawZ = BitConverter.ToSingle(modfData, off + 12); // height
            float rawY = BitConverter.ToSingle(modfData, off + 16);
            float rotX = BitConverter.ToSingle(modfData, off + 20);
            float rotZ = BitConverter.ToSingle(modfData, off + 24);
            float rotY = BitConverter.ToSingle(modfData, off + 28);

            // Diagnostic: dump first 3 MODF raw entries
            if (ModfPlacements.Count < 3)
            {
                string mname = nameIdx < WmoModelNames.Count ? Path.GetFileName(WmoModelNames[nameIdx]) : "?";
                ViewerLog.Trace($"[MODF RAW] [{ModfPlacements.Count}] pos=({rawX:F2}, {rawZ:F2}, {rawY:F2}) rot=({rotX:F2}, {rotZ:F2}, {rotY:F2})  model={mname}");
            }
            float bbMinX = BitConverter.ToSingle(modfData, off + 32);
            float bbMinZ = BitConverter.ToSingle(modfData, off + 36);
            float bbMinY = BitConverter.ToSingle(modfData, off + 40);
            float bbMaxX = BitConverter.ToSingle(modfData, off + 44);
            float bbMaxZ = BitConverter.ToSingle(modfData, off + 48);
            float bbMaxY = BitConverter.ToSingle(modfData, off + 52);
            ushort flags = BitConverter.ToUInt16(modfData, off + 56);

            Vector3 position;
            Vector3 boundsMin, boundsMax;

            if (IsWmoBased)
            {
                // WMO-only maps: vertices are in WoW world coords (X, Y, Z with Z=up in file).
                // MODF file layout: pos=(X, Z, Y), bb=(X, Z, Y) — middle component is height.
                // WMO vertex file layout: (X, Y, Z) — Z is height.
                // So position = (rawX, rawY, rawZ=height) and BB = (bbX, bbY, bbZ=height).
                position = new Vector3(rawX, rawY, rawZ);
                boundsMin = new Vector3(
                    MathF.Min(bbMinX, bbMaxX), MathF.Min(bbMinY, bbMaxY), MathF.Min(bbMinZ, bbMaxZ));
                boundsMax = new Vector3(
                    MathF.Max(bbMinX, bbMaxX), MathF.Max(bbMinY, bbMaxY), MathF.Max(bbMinZ, bbMaxZ));
                if (ModfPlacements.Count < 3)
                    ViewerLog.Trace($"[MODF WMO-ONLY] pos=({position.X:F1},{position.Y:F1},{position.Z:F1}) bb=({boundsMin.X:F1},{boundsMin.Y:F1},{boundsMin.Z:F1})→({boundsMax.X:F1},{boundsMax.Y:F1},{boundsMax.Z:F1})  raw bb file: X({bbMinX:F1}..{bbMaxX:F1}) Z({bbMinZ:F1}..{bbMaxZ:F1}) Y({bbMinY:F1}..{bbMaxY:F1})");
            }
            else
            {
                // Normal terrain maps: convert to renderer coords
                // rendererX=MapOrigin-wowY, rendererY=MapOrigin-wowX, rendererZ=wowZ
                position = new Vector3(
                    WoWConstants.MapOrigin - rawY,
                    WoWConstants.MapOrigin - rawX,
                    rawZ);
                // Note: MapOrigin-min > MapOrigin-max, so swap min/max after conversion
                float rBBMinX = WoWConstants.MapOrigin - bbMaxY;
                float rBBMaxX = WoWConstants.MapOrigin - bbMinY;
                float rBBMinY = WoWConstants.MapOrigin - bbMaxX;
                float rBBMaxY = WoWConstants.MapOrigin - bbMinX;
                boundsMin = new Vector3(rBBMinX, rBBMinY, bbMinZ);
                boundsMax = new Vector3(rBBMaxX, rBBMaxY, bbMaxZ);
            }

            target.Add(new ModfPlacement
            {
                NameIndex = nameIdx,
                UniqueId = uniqueId,
                Position = position,
                Rotation = new Vector3(rotX, rotY, rotZ),
                BoundsMin = boundsMin,
                BoundsMax = boundsMax,
                Flags = flags
            });
        }
    }

    private TerrainChunkData? ExtractChunkData(McnkAlpha mcnk, int tileX, int tileY, int tileIdx)
    {
        int chunkX = mcnk.IndexX;
        int chunkY = mcnk.IndexY;

        // Extract heights (145 floats = 580 bytes, Alpha non-interleaved format)
        var heights = ExtractHeights(mcnk.McvtData);
        if (heights == null) return null;

        // Ghidra-verified (CMapChunk::CreateVertices, 0x5.3.3368):
        // Alpha MCVT heights are ABSOLUTE world-space Z values — no base height addition.
        // The client's CreateVertices uses v->z = *he directly, then subtracts the chunk's
        // position vector (field_0x64/0x68/0x6C) to make vertices relative to the chunk center.
        // That position Z comes from MCNK offset 0x88 (chunk start), not offset 0x70.
        // The old code was reading offset 0x68 from header data (=0x70 from chunk start)
        // and adding it to absolute heights, which caused each chunk to float at a
        // disconnected elevation from its neighbors.
        // mcnk.Header.Unused1/2/3 fields contain the chunk's world position (not a height delta)
        // so they must NOT be added to the MCVT heights.

        // Extract normals (145 × 3 signed bytes, Alpha non-interleaved format)
        var normals = ExtractNormals(mcnk.McnrData);

        // Extract layers from MCLY (16 bytes per layer)
        var layers = ExtractLayers(mcnk.MclyData, mcnk.NLayers);

        // Extract alpha maps from MCAL
        var alphaMaps = ExtractAlphaMaps(mcnk.McalData, mcnk.MclyData, mcnk.NLayers);

        // Extract MCSH shadow map (64×64 bits → 64×64 bytes)
        byte[]? shadowMap = ExtractShadowMap(mcnk.McshData, mcnk.McshSize);

        // Compute world position for this chunk in renderer coordinates.
        // WDT MAIN index = tileX*64+tileY (column-major).
        // Renderer coords match MODF: rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX
        float chunkSmall = WoWConstants.ChunkSize / 16f;
        float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall;
        float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall;

        LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));

        // Extract MCLQ inline liquid data (type from MCNK header flags bits 2-5)
        var liquid = ExtractLiquid(mcnk.MclqData, mcnk.Header.Flags, tileX, tileY, chunkX, chunkY,
            new Vector3(worldX, worldY, 0f));

        return new TerrainChunkData
        {
            TileX = tileX,
            TileY = tileY,
            ChunkX = chunkX,
            ChunkY = chunkY,
            Heights = heights,
            Normals = normals,
            HoleMask = mcnk.Holes,
            Layers = layers,
            AlphaMaps = alphaMaps,
            ShadowMap = shadowMap,
            Liquid = liquid,
            WorldPosition = new Vector3(worldX, worldY, 0f),
            // Alpha 0.5.3 stores the complete packed AreaNumber in Unknown3:
            // high 16 bits = zone and low 16 bits = subzone.
            AreaId = mcnk.Header.Unknown3,
            McnkFlags = mcnk.Header.Flags
        };
    }

    /// <summary>
    /// Extract 145 height floats from Alpha MCVT data, reordering from non-interleaved to interleaved.
    /// Alpha format: 81 outer vertices first, then 64 inner vertices.
    /// Interleaved format: row of 9 outer, row of 8 inner, alternating for 17 rows.
    /// </summary>
    private static float[]? ExtractHeights(byte[] mcvtData)
    {
        if (mcvtData == null || mcvtData.Length < 580) return null;

        var heights = new float[145];
        int destIdx = 0;

        // Alpha layout: [81 outer floats][64 inner floats]
        // Interleaved layout: 9 outer, 8 inner, 9 outer, 8 inner, ... 9 outer (17 rows total)
        for (int row = 0; row < 17; row++)
        {
            if (row % 2 == 0)
            {
                // Outer row (9 vertices)
                int outerRow = row / 2;
                for (int col = 0; col < 9; col++)
                {
                    int srcIdx = (outerRow * 9 + col) * 4; // Alpha: all 81 outer first
                    heights[destIdx++] = BitConverter.ToSingle(mcvtData, srcIdx);
                }
            }
            else
            {
                // Inner row (8 vertices)
                int innerRow = row / 2;
                for (int col = 0; col < 8; col++)
                {
                    int srcIdx = (81 + innerRow * 8 + col) * 4; // Alpha: 64 inner after 81 outer
                    heights[destIdx++] = BitConverter.ToSingle(mcvtData, srcIdx);
                }
            }
        }

        return heights;
    }

    /// <summary>
    /// Extract 145 normals from Alpha MCNR data, reordering from non-interleaved to interleaved.
    /// Each normal is 3 signed bytes (X, Z, Y in WoW coords), normalized to [-1,1].
    /// Alpha format: 81 outer normals first (243 bytes), then 64 inner normals (192 bytes).
    /// </summary>
    private static Vector3[] ExtractNormals(byte[] mcnrData)
    {
        var normals = new Vector3[145];

        if (mcnrData == null || mcnrData.Length < 435) // 145 * 3 = 435 minimum
        {
            // Default to up-facing normals
            for (int i = 0; i < 145; i++)
                normals[i] = Vector3.UnitZ;
            return normals;
        }

        int destIdx = 0;

        for (int row = 0; row < 17; row++)
        {
            if (row % 2 == 0)
            {
                // Outer row (9 normals)
                int outerRow = row / 2;
                for (int col = 0; col < 9; col++)
                {
                    int srcIdx = (outerRow * 9 + col) * 3;
                    normals[destIdx++] = DecodeNormal(mcnrData, srcIdx);
                }
            }
            else
            {
                // Inner row (8 normals)
                int innerRow = row / 2;
                for (int col = 0; col < 8; col++)
                {
                    int srcIdx = (81 * 3) + (innerRow * 8 + col) * 3;
                    normals[destIdx++] = DecodeNormal(mcnrData, srcIdx);
                }
            }
        }

        return normals;
    }

    private static Vector3 DecodeNormal(byte[] data, int offset)
    {
        if (offset + 2 >= data.Length) return Vector3.UnitZ;

        // MCNR stores normals as signed bytes: X, Z, Y (WoW convention)
        float nx = (sbyte)data[offset] / 127f;
        float nz = (sbyte)data[offset + 1] / 127f;
        float ny = (sbyte)data[offset + 2] / 127f;

        return TerrainNormalGeometry.TransformAdtNormalToRenderer(new Vector3(nx, ny, nz));
    }

    private static TerrainLayer[] ExtractLayers(byte[] mclyData, int nLayers)
    {
        if (mclyData == null || mclyData.Length < 16 || nLayers <= 0)
            return Array.Empty<TerrainLayer>();

        int count = Math.Min(nLayers, 4);
        count = Math.Min(count, mclyData.Length / 16);

        var layers = new TerrainLayer[count];
        for (int i = 0; i < count; i++)
        {
            int off = i * 16;
            layers[i] = new TerrainLayer
            {
                TextureIndex = BitConverter.ToInt32(mclyData, off),
                Flags = BitConverter.ToUInt32(mclyData, off + 4),
                AlphaOffset = BitConverter.ToUInt32(mclyData, off + 8),
                EffectId = BitConverter.ToUInt32(mclyData, off + 12)
            };
        }

        return layers;
    }

    /// <summary>
    /// Extract alpha maps from MCAL data. Layer 0 is always fully opaque (no alpha map).
    /// Each alpha map is 64×64 bytes (4096 bytes) for 8-bit, or 32×64 (2048 bytes) for 4-bit.
    /// </summary>
    private static Dictionary<int, byte[]> ExtractAlphaMaps(byte[] mcalData, byte[] mclyData, int nLayers)
    {
        var maps = new Dictionary<int, byte[]>();
        if (mcalData == null || mcalData.Length == 0 || nLayers <= 1)
            return maps;

        int offset = 0;
        for (int layer = 1; layer < nLayers && layer < 4; layer++)
        {
            if (layer * 16 > mclyData.Length) break;

            uint flags = BitConverter.ToUInt32(mclyData, layer * 16 + 4);
            bool isCompressed = (flags & 0x200) != 0;

            // Alpha 0.5.3 typically uses uncompressed 4-bit alpha (2048 bytes = 64×64 / 2)
            int alphaSize = isCompressed ? 4096 : 2048;
            if (offset + alphaSize > mcalData.Length)
            {
                // Try remaining data
                alphaSize = mcalData.Length - offset;
                if (alphaSize <= 0) break;
            }

            byte[] alpha;
            if (alphaSize == 2048)
            {
                // 4-bit alpha: expand to 8-bit (64×64)
                alpha = new byte[4096];
                for (int j = 0; j < Math.Min(2048, alphaSize); j++)
                {
                    byte packed = mcalData[offset + j];
                    alpha[j * 2] = (byte)((packed & 0x0F) * 17);     // low nibble → 0-255
                    alpha[j * 2 + 1] = (byte)((packed >> 4) * 17);   // high nibble → 0-255
                }

                ApplyLegacyEdgeFix(alpha);
            }
            else
            {
                // 8-bit alpha: copy directly
                alpha = new byte[alphaSize];
                Array.Copy(mcalData, offset, alpha, 0, alphaSize);
            }

            maps[layer] = alpha;
            offset += alphaSize;
        }

        return maps;
    }

    private static void ApplyLegacyEdgeFix(byte[] alpha)
    {
        if (alpha.Length < 64 * 64)
            return;

        for (int y = 0; y < 64; y++)
            alpha[y * 64 + 63] = alpha[y * 64 + 62];

        Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);
    }

    /// <summary>
    /// Extract MCLQ liquid data from raw bytes.
    ///
    /// Alpha 0.5.3: often inline payload referenced by ofsLiquid (no chunk header).
    /// Alpha 0.6.0: client code treats MCLQ as a normal chunk and uses payload at +8 (FourCC+size header).
    /// This method strips an MCLQ chunk header if present so decoding starts at the payload.
    /// The 0.5.3 extractor lineage stores 8-byte vertex records (packed
    /// flow/depth word followed by height) and an 8×8 tile flag grid. Later
    /// clients may append flow-vector state; it is not required to decode the
    /// surface or create the environmental sound candidate.
    /// Liquid type determined from MCNK header flags bits 2-5.
    /// Up to 4 liquid instances per chunk (one per type).
    /// Returns the first valid liquid instance found, or null.
    /// </summary>
    private static LiquidChunkData? ExtractLiquid(byte[] mclqData, int mcnkFlags, int tileX, int tileY,
        int chunkX, int chunkY, Vector3 worldPos)
    {
        // Minimum data is the height range. The legacy extractor format uses
        // 81 records after it: a packed flow/depth word followed by the
        // absolute vertex height, then 64 tile flags. Treating the packed word
        // as a float was the source of the stretched/incorrect Alpha liquid
        // surfaces.
        if (mclqData == null || mclqData.Length < 8)
            return null;

        mclqData = StripMclqChunkHeaderIfPresent(mclqData);
        if (mclqData.Length < 8)
            return null;

        // Determine liquid type from MCNK flags.
        // Alpha 0.5.3: bit 2 (0x04) = has liquid, bit 3 (0x08) = ocean override
        // Bits 4-5 encode basic liquid type: 0=water, 1=ocean, 2=magma, 3=slime
        int liquidBits = (mcnkFlags >> 4) & 3; // extract bits 4-5
        LiquidType liquidType;
        if ((mcnkFlags & 0x08) != 0) liquidType = LiquidType.Ocean; // ocean flag override
        else liquidType = liquidBits switch
        {
            1 => LiquidType.Ocean,
            2 => LiquidType.Magma,
            3 => LiquidType.Slime,
            _ => LiquidType.Water
        };
        ViewerLog.Trace($"[MCLQ] tile({tileX},{tileY}) chunk({chunkX},{chunkY}): mcnkFlags=0x{mcnkFlags:X8} liquidBits={liquidBits} type={liquidType}");

        // Read min/max height (8 bytes)
        float minHeight = BitConverter.ToSingle(mclqData, 0);
        float maxHeight = BitConverter.ToSingle(mclqData, 4);

        // Sanity check
        if (float.IsNaN(minHeight) || float.IsNaN(maxHeight))
            return null;

        // Use the average of min/max as the flat liquid surface height
        float liquidHeight = (minHeight + maxHeight) * 0.5f;

        // Alpha MCLQ heights are absolute world-space Z — no base height offset needed.

        // Diagnostic
        if (chunkX == 0 && chunkY == 0)
            ViewerLog.Trace($"[MCLQ] tile({tileX},{tileY}) chunk(0,0): minH={minHeight:F2} maxH={maxHeight:F2} liquidH={liquidHeight:F2} dataLen={mclqData.Length} type={liquidType}");

        // If the height range is absurd after offset, skip (bad data)
        if (MathF.Abs(liquidHeight) > 50000f)
            return null;

        var heights = new float[81];
        var vertexData = new uint[81];
        bool hasVertexHeights = mclqData.Length >= 8 + (81 * 8);
        for (int vertex = 0; vertex < heights.Length; vertex++)
        {
            int offset = 8 + vertex * 8;
            if (!hasVertexHeights || offset + 8 > mclqData.Length)
            {
                heights[vertex] = liquidHeight;
                continue;
            }

            vertexData[vertex] = BitConverter.ToUInt32(mclqData, offset);
            float vertexHeight = BitConverter.ToSingle(mclqData, offset + 4);
            heights[vertex] = float.IsFinite(vertexHeight) && MathF.Abs(vertexHeight) <= 50000f
                ? vertexHeight
                : liquidHeight;
        }

        byte[]? tileFlags = null;
        const int tileFlagsOffset = 8 + (81 * 8);
        if (mclqData.Length >= tileFlagsOffset + 64)
        {
            tileFlags = new byte[64];
            Buffer.BlockCopy(mclqData, tileFlagsOffset, tileFlags, 0, tileFlags.Length);
        }

        return new LiquidChunkData
        {
            MinHeight = minHeight,
            MaxHeight = maxHeight,
            Heights = heights,
            VertexData = vertexData,
            TileGrid = new float[16],
            TileFlags = tileFlags,
            Type = liquidType,
            WorldPosition = worldPos,
            TileX = tileX,
            TileY = tileY,
            ChunkX = chunkX,
            ChunkY = chunkY
        };
    }

    private static byte[] StripMclqChunkHeaderIfPresent(byte[] mclqData)
    {
        if (mclqData.Length < 8)
            return mclqData;

        bool isMclq = mclqData[0] == (byte)'M' && mclqData[1] == (byte)'C' && mclqData[2] == (byte)'L' && mclqData[3] == (byte)'Q';
        bool isReversed = mclqData[0] == (byte)'Q' && mclqData[1] == (byte)'L' && mclqData[2] == (byte)'C' && mclqData[3] == (byte)'M';
        if (!isMclq && !isReversed)
            return mclqData;

        uint size = BitConverter.ToUInt32(mclqData, 4);
        if (size == 0)
            return Array.Empty<byte>();

        int available = mclqData.Length - 8;
        if (size > (uint)available)
            return mclqData;

        var payload = new byte[size];
        Buffer.BlockCopy(mclqData, 8, payload, 0, (int)size);
        return payload;
    }

    /// <summary>
    /// Extract MCSH shadow map: 64×64 bits (512 bytes = 64 rows × 8 bytes/row).
    /// Each bit represents one cell: 1=shadowed, 0=lit.
    /// Expands to 64×64 bytes (0=lit, 255=shadowed) for GPU upload as R8 texture.
    /// </summary>
    private static byte[]? ExtractShadowMap(byte[] mcshData, int mcshSize)
    {
        if (mcshData == null || mcshData.Length == 0 || mcshSize <= 0)
            return null;

        // MCSH is 64 rows × 8 bytes/row = 512 bytes (64×64 bits)
        int rows = Math.Min(64, mcshSize / 8);
        if (rows == 0) return null;

        var shadow = new byte[64 * 64];
        for (int y = 0; y < rows; y++)
        {
            int srcRow = y * 8;
            for (int byteIdx = 0; byteIdx < 8 && srcRow + byteIdx < mcshData.Length; byteIdx++)
            {
                byte bits = mcshData[srcRow + byteIdx];
                for (int bit = 0; bit < 8; bit++)
                {
                    int x = byteIdx * 8 + bit;
                    if (x < 64)
                        shadow[y * 64 + x] = (byte)(((bits >> bit) & 1) * 255);
                }
            }
        }

        return shadow;
    }
}
