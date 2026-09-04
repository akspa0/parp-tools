using System.Collections.Concurrent;
using System.Numerics;
using System.Text;
using DBCD;
using DBCD.Providers;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WowViewer.Core.Diagnostics;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Lk;
using WowViewer.Core.IO.Liquids;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;

namespace WoWViewer.Terrain;

/// <summary>
/// Terrain adapter for standard (LK/Cata+) WDT + split ADT files.
/// Reads from IDataSource (MPQ archives or loose files).
/// Produces <see cref="TerrainChunkData"/> compatible with the existing rendering pipeline.
/// </summary>
public class StandardTerrainAdapter : ITerrainAdapter
{
    private const uint WdtUsesGlobalMapObjFlag = 0x0001u;

    private readonly Mcnk.ParseOptions _mcnkParseOptions;
    private readonly IDataSource _dataSource;
    private readonly string _mapName;
    private readonly string _mapDir; // e.g. "World\\Maps\\Azeroth"
    private readonly List<int> _existingTiles;
    private readonly HashSet<int> _existingTileSet;
    private readonly uint _mphdFlags;
    private readonly bool _useBigAlpha;
    private readonly string? _buildVersion;
    private readonly AdtProfile _adtProfile;
    private readonly Dictionary<ushort, LiquidType> _mh2oLiquidTypesById = new();
    private readonly HashSet<ushort> _reportedUnknownMh2oLiquidTypeIds = new();
    private bool _mh2oLiquidTypeLookupAttempted;

    /// <summary>
    /// Resolves MH2O's <c>liquid_object_or_lvf</c> to a real vertex format (spec 205). Never null;
    /// falls back to <see cref="LiquidVertexFormatChain.Empty"/>, which reproduces the pre-fix
    /// behaviour exactly rather than guessing.
    /// </summary>
    private LiquidVertexFormatChain _liquidVertexFormatChain = LiquidVertexFormatChain.Empty;
    private readonly HashSet<ushort> _reportedUnresolvedVertexFormats = new();

    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();
    public IReadOnlyList<string> MdxModelNames => _mdxNames;
    public IReadOnlyList<string> WmoModelNames => _wmoNames;
    public List<MddfPlacement> MddfPlacements { get; } = new();
    public List<ModfPlacement> ModfPlacements { get; } = new();
    public bool IsWmoBased { get; }
    public List<Vector3> LastLoadedChunkPositions { get; } = new();
    public IReadOnlyList<int> ExistingTiles => _existingTiles;

    private readonly List<PhaseLayerSettings> _phaseLayers = [];

    /// <summary>
    /// The ordered phase overlay stack. Applied in list order, so a later layer wins on any channel
    /// it shares with an earlier one.
    /// </summary>
    public IList<PhaseLayerSettings> PhaseLayers => _phaseLayers;

    /// <summary>
    /// Single-overlay shim over <see cref="PhaseLayers"/>, kept so existing callers and saved
    /// settings keep working. Reads the first enabled layer; assigning replaces the whole stack.
    /// </summary>
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

    /// <summary>Distinct phase warnings already reported, so each is logged once.</summary>
    private readonly HashSet<string> _reportedPhaseIssues = new(StringComparer.OrdinalIgnoreCase);

    private IEnumerable<PhaseLayerSettings> ActivePhaseLayers =>
        _phaseLayers.Where(static layer => layer.Enabled && !string.IsNullOrWhiteSpace(layer.MapName));

    private readonly List<string> _mdxNames = new();
    private readonly List<string> _wmoNames = new();
    private readonly object _placementLock = new();
    private readonly Dictionary<string, int> _mdxNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, int> _wmoNameIndex = new(StringComparer.OrdinalIgnoreCase);

    public StandardTerrainAdapter(
        byte[] wdtBytes,
        string mapName,
        IDataSource dataSource,
        string? buildVersion = null,
        IDBCProvider? dbcProvider = null,
        string? dbdDir = null)
    {
        _dataSource = dataSource;
        _mapName = mapName;
        _mapDir = $"World\\Maps\\{mapName}";
        _buildVersion = buildVersion;

        // Parse MPHD
        _mphdFlags = ReadMphdFlags(wdtBytes);

        // Parse MAIN chunk to enumerate tiles
        _existingTiles = ResolveExistingTiles(wdtBytes);
        _existingTileSet = new HashSet<int>(_existingTiles);

        _adtProfile = ResolveTerrainProfile(buildVersion, _existingTiles);
        _mcnkParseOptions = new Mcnk.ParseOptions
        {
            UseHeaderAlphaSize = _adtProfile.UseMcnkHeaderAlphaSize,
            UseHeaderShadowSize = _adtProfile.UseMcnkHeaderShadowSize
        };

        _useBigAlpha = (_mphdFlags & _adtProfile.BigAlphaFlagsMask) != 0;

        bool hasAnyAdtOnDisk = _existingTiles.Any(idx => {
            int tx = idx / 64, ty = idx % 64;
            string fn = $"{_mapDir}\\{_mapName}_{ty}_{tx}.adt";
            return _dataSource.FileExists(fn);
        });

        // The MPHD global-map-object flag is what makes a map WMO-only — the client's WDT reader
        // gates MWMO/MODF on it, and such a WDT may still flag MAIN tiles, so tile count alone
        // misses those maps and leaves their WMO parsed but never loaded. Also check if ADT files exist.
        IsWmoBased = (_mphdFlags & WdtUsesGlobalMapObjFlag) != 0 || _existingTiles.Count == 0 || !hasAnyAdtOnDisk;

        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"Standard WDT: {_existingTiles.Count} tiles, MPHD=0x{_mphdFlags:X}, bigAlpha={_useBigAlpha}");
        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"Standard ADT profile: {_adtProfile.ProfileId} (build={_buildVersion ?? "unknown"})");

        if (dbcProvider != null && !string.IsNullOrWhiteSpace(dbdDir) && !string.IsNullOrWhiteSpace(buildVersion))
        {
            LoadMh2oLiquidTypeLookup(dbcProvider, dbdDir, buildVersion);
            LoadLiquidVertexFormatChain(dbcProvider, dbdDir, buildVersion);
        }

        bool shouldParseWdtGlobalWmoPlacements = IsWmoBased
            || (_mphdFlags & WdtUsesGlobalMapObjFlag) != 0
            || (FindChunk(wdtBytes, "MWMO") >= 0 && FindChunk(wdtBytes, "MODF") >= 0);

        // WDTs can carry global MWMO/MODF placements both for pure WMO maps and for
        // terrain maps that also use a global WMO shell.
        if (shouldParseWdtGlobalWmoPlacements)
        {
            ParseWdtWmoPlacement(wdtBytes, useRawWorldCoordinates: IsWmoBased);
            ViewerLog.Important(ViewerLog.Category.Terrain,
                IsWmoBased
                    ? $"  WMO-only map: {_wmoNames.Count} WMO names, {ModfPlacements.Count} MODF placements"
                    : $"  WDT global WMO placements: {_wmoNames.Count} WMO names, {ModfPlacements.Count} MODF placements");
        }

        // Diagnostic: dump first 5 tile indices and their decoded coordinates + filenames
        for (int di = 0; di < Math.Min(5, _existingTiles.Count); di++)
        {
            int idx = _existingTiles[di];
            int tx = idx / 64, ty = idx % 64; // tx=row(y), ty=col(x)
            string fn = $"{_mapDir}\\{_mapName}_{ty}_{tx}.adt"; // MapName_x_y = MapName_{col}_{row}
            bool exists = _dataSource.FileExists(fn);
            if (ViewerLog.Verbose)
                ViewerLog.Trace($"[Terrain] Tile[{di}]: rawIdx={idx}, tx={tx}(row), ty={ty}(col), file={fn}, exists={exists}");
        }
    }

    private AdtProfile ResolveTerrainProfile(string? buildVersion, IReadOnlyList<int> existingTiles)
    {
        var resolvedProfile = FormatProfileRegistry.ResolveAdtProfile(buildVersion);
        if (existingTiles.Count == 0 ||
            string.Equals(
                resolvedProfile.ProfileId,
                FormatProfileRegistry.AdtProfile50xUnknown.ProfileId,
                StringComparison.OrdinalIgnoreCase))
            return resolvedProfile;

        if (!TryDetectSplitAdtCompanion(existingTiles, out bool hasTexCompanion, out bool hasObjCompanion, out string? detectedTileBasePath))
            return resolvedProfile;

        var splitProfile = FormatProfileRegistry.AdtProfile40xUnknown;
        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"Split ADT companions detected for map '{_mapName}' at '{detectedTileBasePath}' (texture={hasTexCompanion}, object={hasObjCompanion}); promoting terrain profile from {resolvedProfile.ProfileId} to {splitProfile.ProfileId} while keeping build={buildVersion ?? "unknown"} for non-terrain systems.");
        return splitProfile;
    }

    private bool TryDetectSplitAdtCompanion(IReadOnlyList<int> existingTiles, out bool hasTextureCompanion, out bool hasObjectCompanion, out string? detectedTileBasePath)
    {
        hasTextureCompanion = false;
        hasObjectCompanion = false;
        detectedTileBasePath = null;

        foreach (int idx in existingTiles)
        {
            int tileRow = idx / 64;
            int tileColumn = idx % 64;
            string basePath = $"{_mapDir}\\{_mapName}_{tileColumn}_{tileRow}";

            bool textureExists = _dataSource.FileExists($"{basePath}_tex0.adt")
                || _dataSource.FileExists($"{basePath}_tex1.adt");
            bool objectExists = _dataSource.FileExists($"{basePath}_obj0.adt")
                || _dataSource.FileExists($"{basePath}_obj1.adt");
            if (!textureExists && !objectExists)
                continue;

            hasTextureCompanion = textureExists;
            hasObjectCompanion = objectExists;
            detectedTileBasePath = basePath;
            return true;
        }

        return false;
    }

    public bool TileExists(int tileX, int tileY)
    {
        int idx = tileX * 64 + tileY;
        if (_existingTileSet.Contains(idx))
            return true;

        foreach (PhaseLayerSettings layer in ActivePhaseLayers)
        {
            if (OverlayTileExists(layer.MapName, tileX - layer.TileOffsetX, tileY - layer.TileOffsetY))
                return true;
        }

        return false;
    }

    /// <summary>
    /// Check whether the overlay map has an ADT at the given tile coordinate.
    /// </summary>
    private bool OverlayTileExists(string overlayMapName, int tileX, int tileY)
    {
        if (string.IsNullOrEmpty(overlayMapName))
            return false;

        string overlayBase = $"World\\Maps\\{overlayMapName}\\{overlayMapName}_{tileY}_{tileX}";
        return HasTilePayload(overlayBase);
    }

    /// <summary>Occupied-tile footprints per map, cached — each scan is 4096 existence probes.</summary>
    private readonly Dictionary<string, IReadOnlyList<(int TileX, int TileY)>> _occupiedTileCache =
        new(StringComparer.OrdinalIgnoreCase);

    /// <inheritdoc />
    public bool TryResolveMap(string mapName)
    {
        if (string.IsNullOrWhiteSpace(mapName))
            return false;

        // The base map is trivially resolvable — this adapter is its WDT/ADT source.
        if (string.Equals(mapName, _mapName, StringComparison.OrdinalIgnoreCase))
            return true;

        return OverlayTileExists(mapName, 0, 0)
            || GetOccupiedTiles(mapName).Count > 0;
    }

    /// <inheritdoc />
    public bool IsMapWmoBased(string mapName)
    {
        // Standard overlay WDTs are read through the data source without a parsed WDT header
        // cache; assume terrain. WMO-only standard maps are rare and handled by the global-WMO
        // path, not phase composition.
        return false;
    }

    /// <inheritdoc />
    public IReadOnlyList<(int TileX, int TileY)> GetOccupiedTiles(string mapName)
    {
        if (string.IsNullOrWhiteSpace(mapName))
            return Array.Empty<(int, int)>();

        if (_occupiedTileCache.TryGetValue(mapName, out var cached))
            return cached;

        var tiles = new List<(int, int)>();
        if (string.Equals(mapName, _mapName, StringComparison.OrdinalIgnoreCase))
        {
            foreach (int idx in _existingTileSet)
                tiles.Add((idx / 64, idx % 64));
        }
        else
        {
            for (int tileX = 0; tileX < 64; tileX++)
            {
                for (int tileY = 0; tileY < 64; tileY++)
                {
                    if (OverlayTileExists(mapName, tileX, tileY))
                        tiles.Add((tileX, tileY));
                }
            }
        }

        IReadOnlyList<(int, int)> result = tiles;
        _occupiedTileCache[mapName] = result;
        return result;
    }

    public TileLoadResult LoadTileWithPlacements(int tileX, int tileY)
    {
        var result = new TileLoadResult();
        if (!TileExists(tileX, tileY))
            return result;

        ParsedTileSource parent = LoadMapTile(_mapName, tileX, tileY);

        foreach (PhaseLayerSettings layer in ActivePhaseLayers)
        {
            if (layer.Channels == PhaseDataChannel.None)
                continue;

            // A layer may be authored at different tile coordinates than the map it overlays --
            // instance and dungeon maps are often copies of an earlier revision of a zone stored
            // elsewhere in the grid -- so read the shifted source tile.
            int sourceTileX = tileX - layer.TileOffsetX;
            int sourceTileY = tileY - layer.TileOffsetY;
            if (!OverlayTileExists(layer.MapName, sourceTileX, sourceTileY))
                continue;

            ParsedTileSource phase = LoadMapTile(layer.MapName, sourceTileX, sourceTileY);
            if (layer.HasTileOffset)
                TranslatePhasePlacements(phase, layer);

            MergePhaseTile(parent, phase, layer, tileX, tileY);
        }

        PublishTileTextures(tileX, tileY, parent.Textures);
        PostProcessTileAlpha(parent.Result.Chunks);
        return parent.Result;
    }

    private ParsedTileSource LoadMapTile(string mapName, int tileX, int tileY)
    {
        var result = new TileLoadResult();
        string mapDir = $"World\\Maps\\{mapName}";
        string basePath = $"{mapDir}\\{mapName}_{tileY}_{tileX}";
        string rootPath = $"{basePath}.adt";
        CompanionPaths companions = ResolveCompanionPaths(basePath);

        string? texPath = companions.TexturePath;
        string? objPath = companions.ObjectPath;

        byte[]? texBytes = texPath != null && _dataSource.FileExists(texPath) ? _dataSource.ReadFile(texPath) : null;
        byte[]? objBytes = objPath != null && _dataSource.FileExists(objPath) ? _dataSource.ReadFile(objPath) : null;
        byte[]? adtBytes = _dataSource.ReadFile(rootPath);

        // MPQ-era clients (Cata+ / MoP) ship patched ADTs as PTCH/BSDIFF artifacts; the native
        // resource layer hands MapArea already-reconstructed bytes (MapAdtFileData.cpp). The
        // viewer must apply the same reconstruction before parsing, otherwise the artifact bytes
        // reach ParseAdt and the tile silently produces zero chunks.
        adtBytes = ResolvePatchArtifactBytes(adtBytes, rootPath);
        if (texPath != null)
            texBytes = ResolvePatchArtifactBytes(texBytes, texPath);
        if (objPath != null)
            objBytes = ResolvePatchArtifactBytes(objBytes, objPath);

        if (adtBytes == null || adtBytes.Length == 0)
        {
            bool rootIsEmptyPlaceholder = adtBytes != null && adtBytes.Length == 0;
            if (objBytes != null && objBytes.Length >= 16)
            {
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    rootIsEmptyPlaceholder
                        ? $"[StandardADT] Root ADT is a zero-byte placeholder for tile ({tileX},{tileY}); loading placements from {objPath} only."
                        : $"[StandardADT] Root ADT missing for tile ({tileX},{tileY}); loading placements from {objPath} only.");
                if (TryGetMhdr(objBytes, out int objMhdrStart, out var objMhdr) && objMhdr != null)
                    CollectPlacementsViaMhdr(objBytes, objMhdrStart, objMhdr, tileX, tileY, result);
                else
                    CollectPlacementsFlat(objBytes, tileX, tileY, result);
            }
            else
            {
                ViewerLog.Trace(rootIsEmptyPlaceholder
                    ? $"[StandardADT] ADT is a zero-byte placeholder: {rootPath}"
                    : $"[StandardADT] ADT not found or empty: {rootPath}");
            }

            return new ParsedTileSource(result, []);
        }

        ViewerLog.Trace($"[StandardADT] Loaded {rootPath}: {adtBytes.Length} bytes, first4='{Encoding.ASCII.GetString(adtBytes, 0, Math.Min(4, adtBytes.Length))}', companionBand={FormatCompanionBand(companions.SelectedBand)}");
        try
        {
            List<string> textures = ParseAdt(
                adtBytes,
                texBytes,
                objBytes,
                tileX,
                tileY,
                result,
                companions.SelectedBand);
            return new ParsedTileSource(result, textures);
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"Failed to parse ADT ({tileX},{tileY}) from '{mapName}': {ex.Message}");
            return new ParsedTileSource(result, []);
        }
    }

    /// <summary>
    /// Compose one phase layer onto the tile, channel by channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This used to assign <c>parent.Chunks[i] = phaseChunk</c> and preserve only <c>Liquid</c>, so
    /// every base field the phase chunk did not carry was destroyed. Once split <c>_obj0</c>/
    /// <c>_tex0</c> companions landed, a phase tile became a <em>partial</em> chunk, and a partial
    /// chunk replacing a complete one is exactly the reported symptom: missing trees, and blank
    /// plates where the phase had no texturing of its own.
    /// </para>
    /// <para>
    /// Placements had the mirror-image bug: they were appended unconditionally, so a phase that
    /// restaged objects produced the old set <em>and</em> the new one. They are now presence-gated
    /// replace -- a phase that ships objects owns them for this tile; a phase that ships none is
    /// saying nothing about objects and the base map's survive.
    /// </para>
    /// </remarks>
    private void MergePhaseTile(ParsedTileSource parent, ParsedTileSource phase, PhaseLayerSettings layer, int tileX, int tileY)
    {
        var mergedTextures = new List<string>(parent.Textures);
        var textureIndices = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < mergedTextures.Count; i++)
            textureIndices.TryAdd(mergedTextures[i], i);

        int replacedChunks = 0;
        int addedChunks = 0;
        int skippedBlank = 0;
        PhaseDataChannel contributed = PhaseDataChannel.None;

        foreach (TerrainChunkData phaseChunk in phase.Result.Chunks)
        {
            TerrainChunkData? parentChunk = parent.Result.Chunks.FirstOrDefault(candidate =>
                candidate.ChunkX == phaseChunk.ChunkX && candidate.ChunkY == phaseChunk.ChunkY);

            PhaseDataChannel present = PhaseChunkMerger.DescribePresence(phaseChunk);
            PhaseDataChannel take = PhaseCompositionPolicy.ResolveChannelsToTake(
                layer.Channels, present, layer.OnlyTakeWhatThePhaseCarries);

            // Only remap texture indices for a chunk whose texturing is actually being taken;
            // rewriting them otherwise would renumber layers the composed chunk never uses.
            if ((take & PhaseDataChannel.TextureLayers) != 0)
                RemapPhaseTextureIndices(phaseChunk, phase.Textures, mergedTextures, textureIndices);

            if (parentChunk == null)
            {
                // The base map has no chunk here, so there is nothing to protect and nothing to
                // patch: the phase supplies the whole chunk or the tile stays empty here.
                if (present == PhaseDataChannel.None)
                {
                    skippedBlank++;
                    continue;
                }

                RemapPhaseTextureIndices(phaseChunk, phase.Textures, mergedTextures, textureIndices);
                parent.Result.Chunks.Add(phaseChunk);
                addedChunks++;
                contributed |= present;
                continue;
            }

            if (take == PhaseDataChannel.None)
            {
                skippedBlank++;
                continue;
            }

            int parentIndex = parent.Result.Chunks.IndexOf(parentChunk);
            parent.Result.Chunks[parentIndex] = PhaseChunkMerger.Merge(parentChunk, phaseChunk, take);
            replacedChunks++;
            contributed |= take;
        }

        MergePhasePlacements(parent, phase, layer, ref contributed);

        parent.Textures = mergedTextures;

        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"[StandardADT] Phase patch ({tileX},{tileY}) from '{layer.MapName}': "
            + $"phaseChunks={phase.Result.Chunks.Count} patched={replacedChunks} added={addedChunks} "
            + $"skippedEmpty={skippedBlank} channels={PhaseCompositionPolicy.Describe(contributed)} "
            + $"requested={PhaseCompositionPolicy.Describe(layer.Channels)} "
            + $"presenceGated={layer.OnlyTakeWhatThePhaseCarries}"
            + (layer.HasTileOffset ? $" tileOffset=({layer.TileOffsetX},{layer.TileOffsetY})" : string.Empty));
    }

    /// <summary>
    /// Shift a shifted layer's placements into the base map's coordinates.
    /// </summary>
    /// <remarks>
    /// Chunk identity is taken from the base chunk during the merge, so terrain relocates for free.
    /// Placements do not: MDDF/MODF carry world coordinates, so without this an offset layer's
    /// terrain moves while its objects stay behind at the donor map's coordinates.
    /// </remarks>
    private static void TranslatePhasePlacements(ParsedTileSource phase, PhaseLayerSettings layer)
    {
        // One tile of offset = one ADT in the 64x64 grid = 533.33 yds. Despite its name,
        // WoWConstants.ChunkSize IS that span; WoWConstants.TileSize is 16 ADTs and would
        // overshoot placements 16x. The 2026-09-03 "TileSize fix" was a magnitude error and
        // is reverted here; only the sign/label fix stands.
        (float dx, float dy) = PhaseCompositionPolicy.TileOffsetToWorldTranslation(
            layer.TileOffsetX, layer.TileOffsetY, WoWConstants.ChunkSize);

        for (int i = 0; i < phase.Result.MddfPlacements.Count; i++)
        {
            MddfPlacement placement = phase.Result.MddfPlacements[i];
            placement.Position = new Vector3(placement.Position.X + dx, placement.Position.Y + dy, placement.Position.Z);
            phase.Result.MddfPlacements[i] = placement;
        }

        for (int i = 0; i < phase.Result.ModfPlacements.Count; i++)
        {
            ModfPlacement placement = phase.Result.ModfPlacements[i];
            placement.Position = new Vector3(placement.Position.X + dx, placement.Position.Y + dy, placement.Position.Z);
            phase.Result.ModfPlacements[i] = placement;
        }
    }

    /// <summary>
    /// Presence-gated replace for placements. Appending both sets -- the previous behaviour -- is
    /// the one option that is never right, because it duplicates everything the phase restaged.
    /// </summary>
    private void MergePhasePlacements(
        ParsedTileSource parent,
        ParsedTileSource phase,
        PhaseLayerSettings layer,
        ref PhaseDataChannel contributed)
    {
        if (PhaseCompositionPolicy.PhaseOwnsPlacements(layer.Channels, PhaseDataChannel.Doodads, phase.Result.MddfPlacements.Count))
        {
            parent.Result.MddfPlacements.Clear();
            parent.Result.MddfPlacements.AddRange(phase.Result.MddfPlacements);
            contributed |= PhaseDataChannel.Doodads;
        }
        else if ((layer.Channels & PhaseDataChannel.Doodads) != 0 && !layer.OnlyTakeWhatThePhaseCarries)
        {
            // The phase is authoritative and deliberately empty: it is clearing the doodads.
            parent.Result.MddfPlacements.Clear();
            contributed |= PhaseDataChannel.Doodads;
        }

        if (PhaseCompositionPolicy.PhaseOwnsPlacements(layer.Channels, PhaseDataChannel.WorldObjects, phase.Result.ModfPlacements.Count))
        {
            parent.Result.ModfPlacements.Clear();
            parent.Result.ModfPlacements.AddRange(phase.Result.ModfPlacements);
            contributed |= PhaseDataChannel.WorldObjects;
        }
        else if ((layer.Channels & PhaseDataChannel.WorldObjects) != 0 && !layer.OnlyTakeWhatThePhaseCarries)
        {
            parent.Result.ModfPlacements.Clear();
            contributed |= PhaseDataChannel.WorldObjects;
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

    private void PublishTileTextures(int tileX, int tileY, IReadOnlyList<string> textures)
    {
        TileTextures[(tileX, tileY)] = textures.ToList();
    }

    private void PostProcessTileAlpha(List<TerrainChunkData> chunks)
    {
        if (_adtProfile.AlphaDecodeMode == TerrainAlphaDecodeMode.Cataclysm400)
            PostProcessCataclysm400AlphaMaps(chunks, _useBigAlpha);
    }

    private bool HasTilePayload(string basePath)
        => _dataSource.FileExists($"{basePath}.adt")
            || _dataSource.FileExists($"{basePath}_obj0.adt")
            || _dataSource.FileExists($"{basePath}_tex0.adt")
            || _dataSource.FileExists($"{basePath}_obj1.adt")
            || _dataSource.FileExists($"{basePath}_tex1.adt");

    private sealed record CompanionPaths(
        string? TexturePath,
        string? ObjectPath,
        AdtLodBand? SelectedBand);

    private CompanionPaths ResolveCompanionPaths(string basePath)
    {
        string rootPath = $"{basePath}.adt";
        string tex0Path = $"{basePath}_tex0.adt";
        string tex1Path = $"{basePath}_tex1.adt";
        string obj0Path = $"{basePath}_obj0.adt";
        string obj1Path = $"{basePath}_obj1.adt";

        bool rootExists = _dataSource.FileExists(rootPath);
        bool tex0Exists = _dataSource.FileExists(tex0Path);
        bool tex1Exists = _dataSource.FileExists(tex1Path);
        bool obj0Exists = _dataSource.FileExists(obj0Path);
        bool obj1Exists = _dataSource.FileExists(obj1Path);

        var family = new AdtTileFamily(
            sourcePath: basePath,
            basePath: basePath,
            rootPath: rootPath,
            tex0Path: tex0Path,
            obj0Path: obj0Path,
            tex1Path: tex1Path,
            obj1Path: obj1Path,
            lodPath: $"{basePath}_lod.adt",
            hasRoot: rootExists,
            hasTex0: tex0Exists,
            hasObj0: obj0Exists,
            hasTex1: tex1Exists,
            hasObj1: obj1Exists,
            hasLod: _dataSource.FileExists($"{basePath}_lod.adt"));

        AdtLodBand? selectedBand = family.SelectCompanionBand();
        AdtLodBand sourceBand = selectedBand ?? AdtLodBand.Band0;
        return new CompanionPaths(
            GetExistingCompanionPath(family.GetTextureSourcePath(sourceBand), rootPath),
            GetExistingCompanionPath(family.GetPlacementSourcePath(sourceBand), rootPath),
            selectedBand);
    }

    private string? GetExistingCompanionPath(string? candidatePath, string rootPath)
    {
        if (string.IsNullOrWhiteSpace(candidatePath)
            || string.Equals(candidatePath, rootPath, StringComparison.OrdinalIgnoreCase)
            || !_dataSource.FileExists(candidatePath))
        {
            return null;
        }

        return candidatePath;
    }

    private static string FormatCompanionBand(AdtLodBand? band)
        => band.HasValue ? band.Value.ToString() : "root-only";

    /// <summary>
    /// If the raw bytes are a PTCH/BSDIFF patch artifact, reconstruct the final file bytes by
    /// matching the artifact's embedded base MD5 against raw copies of the same virtual file
    /// across all backing sources. Returns null (treated as missing) when reconstruction fails,
    /// so artifact bytes never reach the ADT parser.
    /// </summary>
    private byte[]? ResolvePatchArtifactBytes(byte[]? data, string virtualPath)
    {
        if (data is not { Length: > 8 } || !AdtPatchArtifactDecoder.IsPatchArtifact(data))
            return data;

        bool reconstructed = AdtPatchArtifactDecoder.TryReconstruct(
            data,
            _dataSource.ReadFileCopies(virtualPath),
            out byte[]? result,
            out string failureReason);

        if (reconstructed && result != null)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[StandardADT] Reconstructed patched ADT '{virtualPath}' ({data.Length} → {result.Length} bytes) via embedded BSDIFF patch.");
            return result;
        }

        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"[StandardADT] '{virtualPath}' is a PTCH patch artifact ({data.Length} bytes) but could not be reconstructed: {failureReason}. Treating as missing.");
        return null;
    }

    private sealed class ParsedTileSource
    {
        public ParsedTileSource(TileLoadResult result, IReadOnlyList<string> textures)
        {
            Result = result;
            Textures = textures.ToList();
        }

        public TileLoadResult Result { get; }
        public IReadOnlyList<string> Textures { get; set; }
    }

    public bool TryGetPlacementSourceData(int tileX, int tileY, out string sourcePath, out byte[] sourceBytes)
    {
        sourcePath = string.Empty;
        sourceBytes = Array.Empty<byte>();

        if (!TileExists(tileX, tileY))
            return false;

        sourcePath = GetPlacementSourceVirtualPath(tileX, tileY);
        byte[]? bytes = _dataSource.ReadFile(sourcePath);
        if (bytes == null || bytes.Length == 0)
        {
            sourcePath = string.Empty;
            return false;
        }

        sourceBytes = bytes;
        return true;
    }

    public bool TryGetPlacementWritablePath(int tileX, int tileY, out string? fullPath)
    {
        fullPath = null;
        if (!TileExists(tileX, tileY))
            return false;

        return _dataSource.TryResolveWritablePath(GetPlacementSourceVirtualPath(tileX, tileY), out fullPath);
    }

    private string GetPlacementSourceVirtualPath(int tileX, int tileY)
    {
        string basePath = $"{_mapDir}\\{_mapName}_{tileY}_{tileX}";
        string rootPath = $"{basePath}.adt";
        CompanionPaths companions = ResolveCompanionPaths(basePath);
        return companions.ObjectPath ?? rootPath;
    }

    private List<int> ResolveExistingTiles(byte[] wdtBytes)
    {
        var wdtTiles = ReadMainChunk(wdtBytes);
        var indexedTiles = EnumerateIndexedMapTiles();
        if (indexedTiles.Count == 0)
            return wdtTiles;

        var mergedTiles = new HashSet<int>(indexedTiles);
        int keptWdtTiles = 0;
        int droppedWdtTiles = 0;

        foreach (int tileIndex in wdtTiles)
        {
            if (mergedTiles.Contains(tileIndex) || TileHasBackingFiles(tileIndex))
            {
                mergedTiles.Add(tileIndex);
                keptWdtTiles++;
            }
            else
            {
                droppedWdtTiles++;
            }
        }

        int addedIndexedTiles = mergedTiles.Count - keptWdtTiles;
        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"Standard WDT tile coverage for '{_mapName}': WDT={wdtTiles.Count}, indexed={indexedTiles.Count}, merged={mergedTiles.Count}, droppedMissing={droppedWdtTiles}, addedFromFiles={Math.Max(0, addedIndexedTiles)}");

        return mergedTiles.OrderBy(static tileIndex => tileIndex).ToList();
    }

    private HashSet<int> EnumerateIndexedMapTiles()
    {
        var tiles = new HashSet<int>();
        string mapPrefix = $"{_mapDir}\\";
        string filePrefix = $"{_mapName}_";

        foreach (string filePath in _dataSource.GetFileList(".adt"))
        {
            if (!filePath.StartsWith(mapPrefix, StringComparison.OrdinalIgnoreCase))
                continue;

            string fileName = Path.GetFileNameWithoutExtension(filePath);
            if (!fileName.StartsWith(filePrefix, StringComparison.OrdinalIgnoreCase))
                continue;

            string suffix = fileName[filePrefix.Length..];
            string[] parts = suffix.Split('_', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2 || parts.Length > 3)
                continue;

            if (!int.TryParse(parts[0], out int tileColumn) || !int.TryParse(parts[1], out int tileRow))
                continue;

            if (tileColumn is < 0 or > 63 || tileRow is < 0 or > 63)
                continue;

            if (parts.Length == 3 &&
                !string.Equals(parts[2], "obj0", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(parts[2], "tex0", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(parts[2], "obj1", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(parts[2], "tex1", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            tiles.Add((tileRow * 64) + tileColumn);
        }

        return tiles;
    }

    private bool TileHasBackingFiles(int tileIndex)
    {
        int tileRow = tileIndex / 64;
        int tileColumn = tileIndex % 64;
        string basePath = $"{_mapDir}\\{_mapName}_{tileColumn}_{tileRow}";

        return _dataSource.FileExists($"{basePath}.adt") ||
               _dataSource.FileExists($"{basePath}_obj0.adt") ||
               _dataSource.FileExists($"{basePath}_tex0.adt") ||
               _dataSource.FileExists($"{basePath}_obj1.adt") ||
               _dataSource.FileExists($"{basePath}_tex1.adt");
    }

    private List<string> ParseAdt(
        byte[] adtBytes,
        byte[]? texBytes,
        byte[]? objBytes,
        int tileX,
        int tileY,
        TileLoadResult result,
        AdtLodBand? selectedBand)
    {
        // Parse top-level MTEX chunk for texture names
        var textures = new List<string>();

        Dictionary<(int x, int y), Mcnk>? texMcnkByIndex = null;
        GillijimProject.WowFiles.Mhdr? texMhdr = null;
        int texMhdrStart = 0;
        Dictionary<(int x, int y), Mcnk>? objMcnkByIndex = null;
        if (texBytes != null && texBytes.Length >= 16)
        {
            if (TryBuildMcnkIndexMap(texBytes, _mcnkParseOptions, out texMcnkByIndex, out texMhdrStart, out texMhdr,
                    $"tile({tileX},{tileY}) texture companion"))
            {
                if (ViewerLog.Verbose)
                    ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) split: parsed {texMcnkByIndex.Count} texture MCNKs from selected {FormatCompanionBand(selectedBand)} texture companion");
            }
            else
            {
                texMcnkByIndex = null;
            }
        }

        if (objBytes != null && objBytes.Length >= 16)
        {
            if (TryBuildMcnkIndexMap(
                    objBytes,
                    _mcnkParseOptions,
                    out objMcnkByIndex,
                    out _,
                    out _,
                    $"tile({tileX},{tileY}) object companion")
                && ViewerLog.Verbose)
            {
                ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) split: parsed {objMcnkByIndex.Count} object MCNKs from selected {FormatCompanionBand(selectedBand)} object companion");
            }
        }

        // Parse MTEX texture names (preferring _tex0.adt when present)
        if (texBytes != null && texBytes.Length >= 16)
        {
            int texMtexOff = FindChunk(texBytes, "MTEX");
            if (texMtexOff >= 0 && texMtexOff + 8 <= texBytes.Length)
            {
                int mtexSize = BitConverter.ToInt32(texBytes, texMtexOff + 4);
                if (mtexSize > 0 && texMtexOff + 8 + mtexSize <= texBytes.Length)
                    textures.AddRange(ParseNullStrings(texBytes, texMtexOff + 8, mtexSize));
            }
            else if (texMhdr != null && texMhdrStart > 0)
            {
                textures.AddRange(ParseMtexViaMhdr(texBytes, texMhdrStart, texMhdr));
            }

        }

        // Find MHDR in root ADT — all other chunks located via MHDR offsets (or flat scan fallback)
        int mhdrOffset = FindChunk(adtBytes, "MHDR");
        GillijimProject.WowFiles.Mhdr? mhdr = null;
        int mhdrStart = 0;
        if (mhdrOffset >= 0 && mhdrOffset + 8 <= adtBytes.Length)
        {
            mhdr = new GillijimProject.WowFiles.Mhdr(adtBytes, mhdrOffset);
            mhdrStart = mhdrOffset + 8;
        }

        if (textures.Count == 0 && mhdr != null && _adtProfile.UseMhdrOffsetsOnly)
        {
            textures.AddRange(ParseMtexViaMhdr(adtBytes, mhdrStart, mhdr));
        }
        else if (textures.Count == 0)
        {
            int mtexOff = FindChunk(adtBytes, "MTEX");
            if (mtexOff >= 0 && mtexOff + 8 <= adtBytes.Length)
            {
                int mtexSize = BitConverter.ToInt32(adtBytes, mtexOff + 4);
                if (mtexSize > 0 && mtexOff + 8 + mtexSize <= adtBytes.Length)
                    textures.AddRange(ParseNullStrings(adtBytes, mtexOff + 8, mtexSize));
            }

        }

        // Locate MCIN / MCNK chunks
        List<int> mcnkOffsets;
        int mcinOff = mhdr?.GetOffset(GillijimProject.WowFiles.Mhdr.McinOffset) ?? 0;
        if (mcinOff == 0)
        {
            mcnkOffsets = ReadMcnkOffsetsByChunkScan(adtBytes, $"tile({tileX},{tileY}) root ADT");
            if (mcnkOffsets.Count == 0)
            {
                Build335Diagnostics.Increment("MissingRequiredChunkCount");
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"MCIN offset zero in ADT ({tileX},{tileY}) and no top-level MCNK fallback was found");
                return textures;
            }

            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"MCIN offset zero in ADT ({tileX},{tileY}); using top-level MCNK scan fallback with {mcnkOffsets.Count} chunks");
        }
        else
        {
            int mcinAbsPos = mhdrStart + mcinOff;
            // Diagnostic: log MCIN position and first bytes
            if (mcinAbsPos + 8 <= adtBytes.Length)
            {
                string mcinSig = Encoding.ASCII.GetString(adtBytes, mcinAbsPos, 4);
                if (ViewerLog.Verbose)
                    ViewerLog.Trace($"[Terrain] MCIN at file pos {mcinAbsPos}: sig='{mcinSig}' (mhdrOff={mhdrOffset}, mhdrStart={mhdrStart}, mcinOff={mcinOff})");

                int mcinSize = BitConverter.ToInt32(adtBytes, mcinAbsPos + 4);
                if (mcinSig != "NICM")
                {
                    Build335Diagnostics.Increment("InvalidChunkSignatureCount");
                    ViewerLog.Info(ViewerLog.Category.Terrain,
                        $"MCIN signature mismatch in ADT ({tileX},{tileY}): '{mcinSig}'");
                    return textures;
                }

                if (mcinSize <= 0 || mcinAbsPos + 8 + mcinSize > adtBytes.Length)
                {
                    Build335Diagnostics.Increment("InvalidChunkSizeCount");
                    ViewerLog.Info(ViewerLog.Category.Terrain,
                        $"MCIN size invalid in ADT ({tileX},{tileY}): size={mcinSize}, file={adtBytes.Length}");
                    return textures;
                }

                if ((mcinSize % _adtProfile.McinEntrySize) != 0)
                {
                    Build335Diagnostics.Increment("UnknownFieldUsageCount");
                    ViewerLog.Important(ViewerLog.Category.Terrain,
                        $"MCIN size misaligned in ADT ({tileX},{tileY}): size={mcinSize}, entrySize={_adtProfile.McinEntrySize}");
                }
            }

            var mcin = new GillijimProject.WowFiles.Mcin(adtBytes, mcinAbsPos);
            mcnkOffsets = mcin.GetMcnkOffsets();
        }

        float chunkSmall = WoWConstants.ChunkSize / 16f;
        var chunks = new List<TerrainChunkData>(256);

        if (mcnkOffsets.Count != 256)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"ADT ({tileX},{tileY}) MCIN entries={mcnkOffsets.Count} (expected 256); parsing available entries only");
        }

        int invalidOffsetCount = 0;
        int invalidSignatureCount = 0;
        int invalidSizeCount = 0;

        // Diagnostic: log first up-to-3 MCIN offsets
        string mcinOffsetPreview = mcnkOffsets.Count switch
        {
            >= 3 => $"{mcnkOffsets[0]}, {mcnkOffsets[1]}, {mcnkOffsets[2]}",
            2 => $"{mcnkOffsets[0]}, {mcnkOffsets[1]}",
            1 => $"{mcnkOffsets[0]}",
            _ => "<none>"
        };
        if (ViewerLog.Verbose)
            ViewerLog.Trace($"[Terrain] MCIN offsets: {mcinOffsetPreview} (fileLen={adtBytes.Length})");

        int mcnkEntryCount = Math.Min(256, mcnkOffsets.Count);
        for (int ci = 0; ci < mcnkEntryCount; ci++)
        {
            int off = mcnkOffsets[ci];
            if (off <= 0) continue;

            try
            {
                // Strict MCNK guardrails: skip malformed chunks immediately (no truncation fallback).
                if (off < 0 || off + 8 > adtBytes.Length)
                {
                    invalidOffsetCount++;
                    continue;
                }

                string sig = Encoding.ASCII.GetString(adtBytes, off, 4);
                if (sig != "KNCM") // MCNK reversed
                {
                    invalidSignatureCount++;
                    continue;
                }

                int mcnkSize = BitConverter.ToInt32(adtBytes, off + 4);
                if (mcnkSize <= 0 || off + 8 + mcnkSize > adtBytes.Length)
                {
                    invalidSizeCount++;
                    continue;
                }

                // Extract MCNK data (skip the 8-byte chunk header)
                int dataLen = mcnkSize;
                if (dataLen < 128) continue;
                var mcnkData = new byte[dataLen];
                Array.Copy(adtBytes, off + 8, mcnkData, 0, dataLen);

                var mcnk = new Mcnk(mcnkData, _mcnkParseOptions);

                int mcinChunkX = ci % 16;
                int mcinChunkY = ci / 16;

                int chunkX = (mcnk.Header.IndexX is >= 0 and < 16) ? (int)mcnk.Header.IndexX : mcinChunkX;
                int chunkY = (mcnk.Header.IndexY is >= 0 and < 16) ? (int)mcnk.Header.IndexY : mcinChunkY;

                if (mcinOff == 0 || (mcnk.Header.IndexX == 0 && mcnk.Header.IndexY == 0 && ci != 0))
                {
                    chunkX = mcinChunkX;
                    chunkY = mcinChunkY;
                }
                AdtMcseData mcse = AdtMcseReader.Read(
                    mcnk.McseData ?? Array.Empty<byte>(),
                    mcnk.Header.NSndEmitters > int.MaxValue ? 0 : (int)mcnk.Header.NSndEmitters);
                foreach (AdtMcseEmitter emitter in mcse.Emitters)
                {
                    result.SoundEmitters.Add(new TerrainSoundEmitter(
                        tileX,
                        tileY,
                        chunkX,
                        chunkY,
                        emitter.SoundPointId,
                        emitter.SoundNameId,
                        emitter.Position,
                        ConvertSoundPosition(emitter.Position, tileX, tileY, chunkX, chunkY),
                        emitter.MinDistance,
                        emitter.MaxDistance,
                        emitter.CutoffDistance,
                        emitter.StartTime,
                        emitter.EndTime,
                        emitter.Mode,
                        emitter.RawEntry,
                        CoordinateProfile: "MCSE chunk-local -> owning chunk -> renderer world"));
                }

                // Diagnostic: log first chunk of each tile
                if (ci == 0)
                {
                    var pos = mcnk.Header.Position;
                    string posStr = pos != null && pos.Length >= 3
                        ? $"Z={pos[0]:F1}, X={pos[1]:F1}, Y={pos[2]:F1}"
                        : "null";
                    if (ViewerLog.Verbose)
                        ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) chunk0: idx=({chunkX},{chunkY}), pos=[{posStr}], baseZ={((pos != null && pos.Length >= 1) ? pos[0] : 0f):F1}");
                }

                // Heights (already interleaved in LK format)
                // MCVT values are deltas from the chunk's base Z (Position[0]).
                float[]? heights = mcnk.Heightmap;
                if (heights == null || heights.Length < 145) continue;

                // Add base height from MCNK Position field (Z, X, Y order)
                float baseZ = (mcnk.Header.Position != null && mcnk.Header.Position.Length >= 1)
                    ? mcnk.Header.Position[0] : 0f;
                if (!float.IsNaN(baseZ) && MathF.Abs(baseZ) < 50000f && baseZ != 0f)
                {
                    for (int hi = 0; hi < heights.Length; hi++)
                        heights[hi] += baseZ;
                }

                // Normals (interleaved in LK)
                var normals = ExtractNormals(mcnk.McnrData);

                var layerSource = mcnk;
                if (texMcnkByIndex != null)
                {
                    if (!texMcnkByIndex.TryGetValue((mcinChunkX, mcinChunkY), out layerSource) &&
                        !texMcnkByIndex.TryGetValue((chunkX, chunkY), out layerSource))
                    {
                        layerSource = mcnk;
                    }
                }

                Mcnk? objectChunk = null;
                if (objMcnkByIndex != null)
                {
                    objMcnkByIndex.TryGetValue((mcinChunkX, mcinChunkY), out objectChunk);
                    if (objectChunk == null)
                        objMcnkByIndex.TryGetValue((chunkX, chunkY), out objectChunk);
                }

                // Layers
                var layers = ExtractLayers(layerSource.TextureLayers);

                // Alpha maps
                uint alphaSourceFlagsRaw = (uint)layerSource.Header.Flags != 0
                    ? (uint)layerSource.Header.Flags
                    : (uint)mcnk.Header.Flags;
                bool doNotFixAlphaMap = (alphaSourceFlagsRaw & 0x8000u) != 0;
                var alphaMaps = ExtractAlphaMaps(layerSource, _adtProfile.AlphaDecodeMode, _useBigAlpha, doNotFixAlphaMap);

                // Shadow map
                byte[]? shadowMap = ExtractShadowMap(layerSource.McshData ?? mcnk.McshData);

                // MCCV vertex colors (WotLK+)
                // MCCV is independent from MCLY/MCAL. In split 3.x/4.x ADTs
                // it may be carried by the texture MCNK even when the root
                // MCNK has no layer or alpha payload, so preserve either
                // valid source instead of making vertex color availability
                // depend on texture-layer presence.
                byte[]? mccvColors = SelectMccvData(mcnk, layerSource);

                // Hole mask
                int holeMask = (int)mcnk.Header.Holes;

                // Cataclysm+/MoP object companions carry per-slot reference
                // streams in headerless MCNK wrappers. Preserve those streams
                // on the matching terrain chunk without changing tile admission.
                int[] mcrdReferences = ReadReferenceIndices(objectChunk?.McrdData);
                int[] mcrwReferences = ReadReferenceIndices(objectChunk?.McrwData);

                // World position: tileX=row (north-south→rendererX), tileY=col (east-west→rendererY)
                // Same convention as Alpha adapter (tx=row, ty=col).
                float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall;
                float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall;

                // MCLQ inline liquid (per-chunk, legacy format used alongside MH2O)
                LiquidChunkData? liquid = null;
                uint mcnkFlagsRaw = (uint)mcnk.Header.Flags;
                bool hasLiquidFlags = (mcnkFlagsRaw & 0x3C) != 0;
                if (mcnk.MclqData != null && mcnk.MclqData.Length >= 8)
                {
                    liquid = ExtractMclq(mcnk.MclqData, mcnkFlagsRaw,
                        tileX, tileY, chunkX, chunkY,
                        new Vector3(worldX, worldY, 0f), baseZ,
                        _adtProfile.MclqLayerStride, _adtProfile.MclqTileFlagsOffset);
                }
                // Diagnostic: log first few chunks with liquid flags
                if (hasLiquidFlags && ci < 4)
                {
                    var sb = new System.Text.StringBuilder();
                    sb.Append($"[MCLQ-DIAG] tile({tileX},{tileY}) chunk({chunkX},{chunkY}) flags=0x{mcnkFlagsRaw:X} ofsMclq=0x{mcnk.Header.OfsMclq:X}");
                    sb.Append($" baseZ={baseZ:F2} mclqData={mcnk.MclqData?.Length ?? -1}");
                    if (mcnk.MclqData != null && mcnk.MclqData.Length >= 16)
                    {
                        float v0 = BitConverter.ToSingle(mcnk.MclqData, 0);
                        float v1 = BitConverter.ToSingle(mcnk.MclqData, 4);
                        float v2 = BitConverter.ToSingle(mcnk.MclqData, 8);
                        float v3 = BitConverter.ToSingle(mcnk.MclqData, 12);
                        sb.Append($" raw[0..3]: {v0:F2} {v1:F2} {v2:F2} {v3:F2}");
                    }
                    sb.Append($" liquid={liquid != null}");
                    if (liquid != null)
                        sb.Append($" minH={liquid.MinHeight:F2} maxH={liquid.MaxHeight:F2} h[0]={liquid.Heights[0]:F2} h[40]={liquid.Heights[40]:F2}");
                    if (ViewerLog.Verbose)
                        ViewerLog.Trace($"[Terrain] {sb}");
                }

                TerrainChunkData chunkData = new()
                {
                    McinIndex = ci,
                    TileX = tileX,
                    TileY = tileY,
                    ChunkX = chunkX,
                    ChunkY = chunkY,
                    Heights = heights,
                    Normals = normals,
                    HoleMask = holeMask,
                    Layers = layers,
                    AlphaMaps = alphaMaps,
                    ShadowMap = shadowMap,
                    MccvColors = mccvColors,
                    Liquid = liquid,
                    WorldPosition = new Vector3(worldX, worldY, 0f),
                    AreaId = (int)mcnk.Header.AreaId,
                    McnkFlags = (int)mcnkFlagsRaw,
                    AlphaSourceFlags = (int)alphaSourceFlagsRaw,
                    McrdReferences = mcrdReferences,
                    McrwReferences = mcrwReferences
                };
                chunks.Add(chunkData);
                LegacyLiquidSoundEmitterFactory.Append(result.SoundEmitters, chunkData);

                LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"Chunk {ci} parse error in ({tileX},{tileY}): {ex.Message}");
            }
        }

        if (invalidOffsetCount > 0 || invalidSignatureCount > 0 || invalidSizeCount > 0)
        {
            if (invalidSignatureCount > 0)
                Build335Diagnostics.Increment("InvalidChunkSignatureCount", invalidSignatureCount);
            if (invalidSizeCount > 0)
                Build335Diagnostics.Increment("InvalidChunkSizeCount", invalidSizeCount);
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"ADT ({tileX},{tileY}) skipped malformed MCNKs: invalidOff={invalidOffsetCount}, invalidSig={invalidSignatureCount}, invalidSize={invalidSizeCount}");
        }

        int mclqCount = chunks.Count(c => c.Liquid != null);
        if (chunks.Count > 0)
        {
            if (ViewerLog.Verbose)
                ViewerLog.Trace($"[Terrain] ADT ({tileX},{tileY}): parsed {chunks.Count} chunks with heightmaps, {mclqCount} with MCLQ liquid");
        }
        else
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"ADT ({tileX},{tileY}): WARNING - 0 chunks parsed (no heightmaps found)");

        result.Chunks.AddRange(chunks);

        // Ghidra-verified (FUN_007d6ef0): MH2O located via MHDR offset +0x28.
        // For WotLK+, MH2O is the primary liquid system and should always be parsed.
        // MH2O supplements MCLQ; chunks that already have MCLQ liquid are not overwritten.
        if (_adtProfile.EnableMh2oFallbackWhenNoMclq && mhdr != null)
            ParseMh2o(adtBytes, mhdrStart, mhdr, tileX, tileY, chunkSmall, result);

        // Placement parsing: check objBytes first, then adtBytes (supporting both MHDR offsets and flat chunk scan)
        if (objBytes != null && objBytes.Length >= 16)
        {
            if (TryGetMhdr(objBytes, out int objMhdrStart, out var objMhdr) && objMhdr != null)
                CollectPlacementsViaMhdr(objBytes, objMhdrStart, objMhdr, tileX, tileY, result);
            else
                CollectPlacementsFlat(objBytes, tileX, tileY, result);
        }
        else if (mhdr != null)
        {
            CollectPlacementsViaMhdr(adtBytes, mhdrStart, mhdr, tileX, tileY, result);
        }
        else
        {
            CollectPlacementsFlat(adtBytes, tileX, tileY, result);
        }

        return textures;
    }

    private static Vector3 ConvertSoundPosition(
        Vector3 position,
        int tileX,
        int tileY,
        int chunkX,
        int chunkY)
    {
        Vector3 chunkWorldPosition = TerrainCoordinateTransform.ChunkCorner(
            tileX,
            tileY,
            chunkX,
            chunkY,
            WoWConstants.MapOrigin,
            WoWConstants.ChunkSize,
            WoWConstants.ChunksPerTileEdge);
        // Standard MCSE uses the same resident chunk-local hand-off as the
        // Alpha reader; keep the transform in the shared core owner.
        return TerrainCoordinateTransform.FromChunkLocal(position, chunkWorldPosition);
    }

    private static bool TryGetMhdr(byte[] adtBytes, out int mhdrStart, out GillijimProject.WowFiles.Mhdr? mhdr)
    {
        mhdrStart = 0;
        mhdr = null;

        int mhdrOffset = FindChunk(adtBytes, "MHDR");
        if (mhdrOffset < 0 || mhdrOffset + 8 > adtBytes.Length)
            return false;

        mhdr = new GillijimProject.WowFiles.Mhdr(adtBytes, mhdrOffset);
        mhdrStart = mhdrOffset + 8;
        return true;
    }

    private static bool TryBuildMcnkIndexMap(
        byte[] adtBytes,
        Mcnk.ParseOptions parseOptions,
        out Dictionary<(int x, int y), Mcnk> mcnkByIndex,
        out int mhdrStart,
        out GillijimProject.WowFiles.Mhdr? mhdr,
        string? diagnosticLabel = null)
    {
        mcnkByIndex = new Dictionary<(int x, int y), Mcnk>();
        mhdrStart = 0;
        mhdr = null;

        List<McnkOffset> offsets;
        if (TryGetMhdr(adtBytes, out mhdrStart, out var foundMhdr) && foundMhdr != null)
        {
            mhdr = foundMhdr;
            int mcinOff = foundMhdr.GetOffset(GillijimProject.WowFiles.Mhdr.McinOffset);
            if (mcinOff == 0)
            {
                offsets = ReadMcnkOffsetsByChunkScan(adtBytes, diagnosticLabel)
                    .Select(static (offset, ordinal) => new McnkOffset(ordinal, offset))
                    .ToList();
            }
            else
            {
                int mcinAbsPos = mhdrStart + mcinOff;
                if (mcinAbsPos + 8 <= adtBytes.Length &&
                    Encoding.ASCII.GetString(adtBytes, mcinAbsPos, 4) == "NICM")
                {
                    var mcin = new GillijimProject.WowFiles.Mcin(adtBytes, mcinAbsPos);
                    offsets = mcin.GetMcnkOffsets()
                        .Select(static (offset, slot) => new McnkOffset(slot, offset))
                        .ToList();
                }
                else
                {
                    offsets = ReadMcnkOffsetsByChunkScan(adtBytes, diagnosticLabel)
                        .Select(static (offset, ordinal) => new McnkOffset(ordinal, offset))
                        .ToList();
                }
            }
        }
        else
        {
            // Flat chunk stream (e.g. Cataclysm / MoP split _tex0.adt and _obj0.adt)
            offsets = ReadMcnkOffsetsByChunkScan(adtBytes, diagnosticLabel)
                .Select(static (offset, ordinal) => new McnkOffset(ordinal, offset))
                .ToList();
        }

        if (offsets.Count == 0)
            return false;

        int entryCount = Math.Min(256, offsets.Count);
        var texParseOptions = new Mcnk.ParseOptions
        {
            UseHeaderAlphaSize = parseOptions.UseHeaderAlphaSize,
            UseHeaderShadowSize = parseOptions.UseHeaderShadowSize,
            SkipHeader = true
        };

        for (int ci = 0; ci < entryCount; ci++)
        {
            McnkOffset indexedOffset = offsets[ci];
            int off = indexedOffset.Offset;
            if (off <= 0 || off + 8 > adtBytes.Length)
                continue;

            string sig = Encoding.ASCII.GetString(adtBytes, off, 4);
            if (sig != "KNCM")
                continue;

            int mcnkSize = BitConverter.ToInt32(adtBytes, off + 4);
            if (mcnkSize <= 0 || off + 8 + mcnkSize > adtBytes.Length || mcnkSize < 4)
                continue;

            var mcnkData = new byte[mcnkSize];
            Array.Copy(adtBytes, off + 8, mcnkData, 0, mcnkSize);

            var mcnk = new Mcnk(mcnkData, texParseOptions);
            mcnkByIndex[(indexedOffset.Slot % 16, indexedOffset.Slot / 16)] = mcnk;
        }

        return mcnkByIndex.Count > 0;
    }

    private readonly record struct McnkOffset(int Slot, int Offset);

    /// <summary>
    /// Walk the top-level chunk stream and collect MCNK record offsets, following the native
    /// 5.0.1 rule exactly: <c>next = payload + size</c>, <c>remaining -= 8 + size</c>, with no
    /// alignment padding (<c>MapAdtFileData.cpp</c> FUN_00bb6f10). The native parser asserts
    /// two invariants the caller cannot otherwise observe — the walk consumes the buffer
    /// exactly (<c>dataSize == 0</c>) and yields exactly 0x100 MCNK records — so a desynced or
    /// truncated walk is reported here instead of silently producing a partial tile.
    /// </summary>
    private static List<int> ReadMcnkOffsetsByChunkScan(byte[] adtBytes)
        => ReadMcnkOffsetsByChunkScan(adtBytes, null);

    private static List<int> ReadMcnkOffsetsByChunkScan(byte[] adtBytes, string? diagnosticLabel)
    {
        var offsets = new List<int>(256);
        int position = 0;
        bool desynced = false;

        while (position + 8 <= adtBytes.Length)
        {
            string signature = Encoding.ASCII.GetString(adtBytes, position, 4);
            int size = BitConverter.ToInt32(adtBytes, position + 4);
            if (size < 0)
            {
                desynced = true;
                break;
            }

            if (signature == "KNCM")
                offsets.Add(position);

            // Native 5.0.1 walks chunks as next = payload + size with no alignment padding
            // (MapAdtFileData.cpp FUN_00bb6f10, MapChunk.cpp FUN_00ba3050: remaining -= 8 + size,
            // then assert dataSize == 0). Padding an odd-sized chunk desyncs the walk and
            // silently truncates the tile.
            int next = position + 8 + size;
            if (next <= position || next > adtBytes.Length)
            {
                desynced = next <= position || next > adtBytes.Length;
                break;
            }

            position = next;
        }

        if (diagnosticLabel != null && (desynced || position != adtBytes.Length || offsets.Count != 256))
        {
            Build335Diagnostics.Increment("AdtChunkWalkDesyncCount");
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[StandardADT] Chunk walk did not satisfy the native invariants for {diagnosticLabel}: " +
                $"consumed {position}/{adtBytes.Length} bytes, found {offsets.Count} MCNK records (native asserts 256 and a fully consumed buffer)");
        }

        return offsets;
    }

    private Vector3[] ExtractNormals(byte[]? mcnrData)
    {
        var normals = new Vector3[145];

        if (mcnrData != null && mcnrData.Length >= 435)
        {
            // LK MCNR is already interleaved (9-8-9-8 pattern)
            // MCNR component order is ERA-SCOPED, and this was measured, not inherited.
            // Ground truth: normals central-differenced from the tile's own heightmap, which
            // assumes no byte-order convention. Mean |dot| against that surface, per candidate:
            //
            //   client                 raw (b0,b1,b2)   swapped (b0,b2,b1)
            //   0.5.3.3368 alpha WDT       0.1809            0.9976     <- alpha path, unchanged
            //   4.0.0.12025 Cataclysm      0.9766            0.0303
            //   5.0.1.15464 MoP            0.8366            0.2068
            //
            // So the long-standing "MCNR stores X, Z, Y (WoW convention)" comment is correct for
            // the ALPHA reader and wrong for this one. The swap it performed put the up component
            // into the horizontal axis, leaving the renderer's normals up to ~70 degrees off the
            // real surface and collapsing max(dot(N, L), 0) toward ambient -- the reported
            // "terrain is too dark on non-0.5.3 eras". Alpha keeps its own decode in
            // AlphaWdtReader/AlphaTerrainAdapter, so this change cannot regress 0.5.3.
            // Verified via: inspect adt terrain-shading --client <dir> --build <version>
            for (int i = 0; i < 145; i++)
            {
                int off = i * 3;
                float nx = (sbyte)mcnrData[off] / 127f;
                float ny = (sbyte)mcnrData[off + 1] / 127f;
                float nz = (sbyte)mcnrData[off + 2] / 127f;
                normals[i] = TerrainNormalGeometry.TransformAdtNormalToRenderer(new Vector3(nx, ny, nz));
            }
            return normals;
        }

        for (int i = 0; i < 145; i++)
            normals[i] = Vector3.UnitZ;
        return normals;
    }

    private static TerrainLayer[] ExtractLayers(List<MclyEntry>? mclyEntries)
    {
        if (mclyEntries == null || mclyEntries.Count == 0)
            return Array.Empty<TerrainLayer>();

        var layers = new TerrainLayer[mclyEntries.Count];
        for (int i = 0; i < mclyEntries.Count; i++)
        {
            var e = mclyEntries[i];
            layers[i] = new TerrainLayer
            {
                TextureIndex = (int)e.TextureId,
                Flags = (uint)e.Flags,
                AlphaOffset = e.AlphaMapOffset,
                EffectId = e.EffectId
            };
        }
        return layers;
    }

    private static byte[]? SelectMccvData(Mcnk rootChunk, Mcnk? textureChunk)
    {
        const int minimumMccvSize = 145 * 4;

        if (rootChunk.MccvData is { Length: >= minimumMccvSize })
            return rootChunk.MccvData;

        if (textureChunk is not null && textureChunk.MccvData is { Length: >= minimumMccvSize })
            return textureChunk.MccvData;

        return null;
    }

    private static int[] ReadReferenceIndices(byte[]? payload)
    {
        if (payload == null || payload.Length < sizeof(int))
            return Array.Empty<int>();

        int count = payload.Length / sizeof(int);
        var references = new int[count];
        for (int index = 0; index < count; index++)
            references[index] = BitConverter.ToInt32(payload, index * sizeof(int));

        return references;
    }

    private static Dictionary<int, byte[]> ExtractAlphaMaps(Mcnk mcnk, TerrainAlphaDecodeMode decodeMode, bool useBigAlpha, bool doNotFixAlphaMap = false)
    {
        var maps = new Dictionary<int, byte[]>();
        if (mcnk.TextureLayers == null || mcnk.TextureLayers.Count <= 1)
            return maps;

        if (decodeMode == TerrainAlphaDecodeMode.LichKingStrict || decodeMode == TerrainAlphaDecodeMode.Cataclysm400)
        {
            if (mcnk.McalRawData == null || mcnk.McalRawData.Length == 0)
                return maps;

            bool isCataclysm400 = decodeMode == TerrainAlphaDecodeMode.Cataclysm400;
            bool chunkBigAlpha = InferBigAlphaForChunk(mcnk, useBigAlpha);
            int mcalLength = mcnk.McalRawData.Length;

            for (int i = 1; i < mcnk.TextureLayers.Count && i < 4; i++)
            {
                var layer = mcnk.TextureLayers[i];
                byte[]? alpha = null;
                int offset = unchecked((int)layer.AlphaMapOffset);
                bool hasValidOffset = offset >= 0 && offset < mcalLength;

                int? nextOffset = GetNextLayerAlphaOffset(mcnk.TextureLayers, i, mcalLength);
                bool layerBigAlpha = ResolveLayerBigAlpha(
                    offset,
                    nextOffset,
                    mcalLength,
                    chunkBigAlpha);

                if (!nextOffset.HasValue && hasValidOffset)
                {
                    int expectedSpan = layerBigAlpha ? 4096 : 2048;
                    int expectedNext = offset + expectedSpan;
                    if (expectedNext > offset && expectedNext <= mcalLength)
                        nextOffset = expectedNext;
                }

                if (hasValidOffset)
                {
                    try
                    {
                        int maxLength = nextOffset.HasValue
                            ? Math.Max(0, nextOffset.Value - offset)
                            : Math.Max(0, mcalLength - offset);

                        uint effectiveFlags = (uint)layer.Flags;
                        if (!isCataclysm400 && ShouldForceCompressedAlpha(effectiveFlags, maxLength))
                            effectiveFlags |= (uint)MclyFlags.CompressedAlpha;

                        alpha = isCataclysm400
                            ? DecodeCataclysm400Alpha(
                                mcnk.McalRawData,
                                offset,
                                effectiveFlags,
                                layerBigAlpha,
                                doNotFixAlphaMap,
                                maxLength)
                            : AlphaMapService.ReadAlpha(
                                mcnk.McalRawData,
                                offset,
                                effectiveFlags,
                                layerBigAlpha,
                                doNotFixAlphaMap,
                                maxLength);
                    }
                    catch
                    {
                        // Keep parse resilient and continue with remaining layers.
                    }
                }

                if (alpha != null && alpha.Length > 0)
                    maps[i] = alpha;
            }

            return maps;
        }

        bool chunkBigAlphaLegacy = InferBigAlphaForChunk(mcnk, useBigAlpha);
        if (mcnk.AlphaMaps != null && mcnk.McalRawData != null && mcnk.McalRawData.Length > 0)
        {
            int mcalLength = mcnk.McalRawData.Length;
            int layerCount = mcnk.TextureLayers.Count;
            for (int i = 1; i < layerCount && i < 4; i++)
            {
                int offset = unchecked((int)mcnk.TextureLayers[i].AlphaMapOffset);
                if (offset < 0 || offset >= mcalLength)
                    continue;

                int? nextOffset = GetNextLayerAlphaOffset(mcnk.TextureLayers, i, mcalLength);
                if (!nextOffset.HasValue)
                {
                    int expectedSpan = chunkBigAlphaLegacy ? 4096 : 2048;
                    int expectedNext = offset + expectedSpan;
                    if (expectedNext > offset && expectedNext <= mcalLength)
                        nextOffset = expectedNext;
                }

                byte[]? alpha = null;
                try
                {
                    alpha = mcnk.AlphaMaps.GetAlphaMapForLayerRelaxed(
                        mcnk.TextureLayers[i],
                        nextOffset,
                        chunkBigAlphaLegacy,
                        doNotFixAlphaMap);
                }
                catch
                {
                    alpha = DecodeLayerBySpan(mcnk.McalRawData, offset, nextOffset, chunkBigAlphaLegacy, doNotFixAlphaMap);
                }

                if (alpha != null && alpha.Length > 0)
                    maps[i] = alpha;
            }
        }

        if (maps.Count == 0 && mcnk.McalRawData != null && mcnk.McalRawData.Length > 0)
        {
            int offset = 0;
            int nLayers = mcnk.TextureLayers.Count;
            for (int layer = 1; layer < nLayers && layer < 4; layer++)
            {
                int alphaSize = chunkBigAlphaLegacy ? 4096 : 2048;
                if (offset + alphaSize > mcnk.McalRawData.Length)
                {
                    alphaSize = mcnk.McalRawData.Length - offset;
                    if (alphaSize <= 0)
                        break;
                }

                byte[]? alpha = chunkBigAlphaLegacy
                    ? DecodeLayerBySpan(mcnk.McalRawData, offset, offset + alphaSize, true, doNotFixAlphaMap)
                    : DecodeLayerBySpan(mcnk.McalRawData, offset, offset + alphaSize, false, doNotFixAlphaMap);

                if (alpha == null || alpha.Length == 0)
                    break;

                maps[layer] = alpha;
                offset += alphaSize;
            }
        }

        return maps;
    }

    private static void PostProcessCataclysm400AlphaMaps(List<TerrainChunkData> chunks, bool useBigAlpha)
    {
        if (!useBigAlpha || chunks.Count == 0)
            return;

        var chunkLookup = new Dictionary<(int chunkX, int chunkY), TerrainChunkData>();
        foreach (var chunk in chunks)
            chunkLookup[(chunk.ChunkX, chunk.ChunkY)] = chunk;

        foreach (var chunk in chunks)
            SynthesizeCataclysm400ResidualAlpha(chunk, useBigAlpha);

        foreach (var chunk in chunks)
        {
            bool doNotFixAlphaMap = (((uint)chunk.AlphaSourceFlags) & 0x8000u) != 0;
            if (!doNotFixAlphaMap)
                StitchCataclysm400ChunkEdges(chunk, chunkLookup);
        }
    }

    private static void SynthesizeCataclysm400ResidualAlpha(TerrainChunkData chunk, bool useBigAlpha)
    {
        if (!useBigAlpha || chunk.Layers.Length <= 1)
            return;

        int layerCount = Math.Min(4, chunk.Layers.Length);
        for (int targetLayerIndex = 1; targetLayerIndex < layerCount; targetLayerIndex++)
        {
            if (chunk.AlphaMaps.ContainsKey(targetLayerIndex))
                continue;

            if ((chunk.Layers[targetLayerIndex].Flags & 0x100u) != 0)
                continue;

            var residualAlpha = new byte[64 * 64];
            for (int pixelIndex = 0; pixelIndex < residualAlpha.Length; pixelIndex++)
            {
                int residual = 255;
                for (int layerIndex = 1; layerIndex < layerCount; layerIndex++)
                {
                    if (layerIndex == targetLayerIndex)
                        continue;

                    if (!chunk.AlphaMaps.TryGetValue(layerIndex, out var otherAlpha) || otherAlpha.Length <= pixelIndex)
                        continue;

                    residual -= otherAlpha[pixelIndex];
                }

                residualAlpha[pixelIndex] = (byte)Math.Clamp(residual, 0, 255);
            }

            chunk.AlphaMaps[targetLayerIndex] = residualAlpha;
        }
    }

    private static void StitchCataclysm400ChunkEdges(
        TerrainChunkData chunk,
        IReadOnlyDictionary<(int chunkX, int chunkY), TerrainChunkData> chunkLookup)
    {
        int layerCount = Math.Min(4, chunk.Layers.Length);
        for (int layerIndex = 1; layerIndex < layerCount; layerIndex++)
        {
            if (!chunk.AlphaMaps.TryGetValue(layerIndex, out var alpha) || alpha.Length < 64 * 64)
                continue;

            int textureIndex = chunk.Layers[layerIndex].TextureIndex;

            if (TryGetMatchingNeighborAlpha(chunkLookup, chunk.ChunkX, chunk.ChunkY + 1, textureIndex, out var eastAlpha))
            {
                for (int row = 0; row < 64; row++)
                    alpha[row * 64 + 63] = eastAlpha[row * 64];
            }

            if (TryGetMatchingNeighborAlpha(chunkLookup, chunk.ChunkX + 1, chunk.ChunkY, textureIndex, out var southAlpha))
                Buffer.BlockCopy(southAlpha, 0, alpha, 63 * 64, 64);

            if (TryGetMatchingNeighborAlpha(chunkLookup, chunk.ChunkX + 1, chunk.ChunkY + 1, textureIndex, out var diagonalAlpha))
                alpha[(64 * 64) - 1] = diagonalAlpha[0];
        }
    }

    private static bool TryGetMatchingNeighborAlpha(
        IReadOnlyDictionary<(int chunkX, int chunkY), TerrainChunkData> chunkLookup,
        int neighborChunkX,
        int neighborChunkY,
        int textureIndex,
        out byte[] alpha)
    {
        alpha = Array.Empty<byte>();

        if (!chunkLookup.TryGetValue((neighborChunkX, neighborChunkY), out var neighborChunk))
            return false;

        int layerCount = Math.Min(4, neighborChunk.Layers.Length);
        for (int layerIndex = 1; layerIndex < layerCount; layerIndex++)
        {
            if (neighborChunk.Layers[layerIndex].TextureIndex != textureIndex)
                continue;

            if (!neighborChunk.AlphaMaps.TryGetValue(layerIndex, out alpha) || alpha.Length < 64 * 64)
                continue;

            return true;
        }

        alpha = Array.Empty<byte>();
        return false;
    }

    private static int? GetNextLayerAlphaOffset(List<MclyEntry> textureLayers, int currentLayerIndex, int mcalLength)
    {
        int currentOffset = unchecked((int)textureLayers[currentLayerIndex].AlphaMapOffset);
        int? nextOffset = null;

        for (int layerIndex = currentLayerIndex + 1; layerIndex < textureLayers.Count; layerIndex++)
        {
            int candidateOffset = unchecked((int)textureLayers[layerIndex].AlphaMapOffset);
            if (candidateOffset <= currentOffset || candidateOffset > mcalLength)
                continue;

            nextOffset = candidateOffset;
            break;
        }

        return nextOffset;
    }

    private static bool ResolveLayerBigAlpha(int currentOffset, int? nextOffset, int mcalLength, bool defaultBigAlpha)
    {
        if (currentOffset < 0 || currentOffset >= mcalLength)
            return defaultBigAlpha;

        int span = nextOffset.HasValue
            ? nextOffset.Value - currentOffset
            : mcalLength - currentOffset;

        if (span >= 4096)
            return true;

        if (span > 0 && span <= 2048)
            return false;

        return defaultBigAlpha;
    }

    private static bool ShouldForceCompressedAlpha(uint flags, int maxLength)
    {
        if ((flags & (uint)MclyFlags.CompressedAlpha) != 0)
            return false;

        // Real 3.x chunks can omit the compressed flag while still storing tiny RLE payloads.
        // Any positive span smaller than the 2048-byte packed format cannot be valid 4-bit or 8-bit alpha.
        return maxLength > 0 && maxLength < 2048;
    }

    private static byte[]? DecodeCataclysm400Alpha(byte[] mcalRawData, int offset, uint flags, bool useBigAlpha, bool doNotFixAlphaMap, int maxLength)
    {
        if ((flags & (uint)MclyFlags.CompressedAlpha) != 0)
        {
            return AlphaMapService.ReadAlpha(
                mcalRawData,
                offset,
                flags,
                useBigAlpha,
                doNotFixAlphaMap,
                maxLength);
        }

        if (!useBigAlpha)
        {
            return AlphaMapService.ReadAlpha(
                mcalRawData,
                offset,
                flags,
                false,
                doNotFixAlphaMap,
                maxLength);
        }

        int available = Math.Max(0, Math.Min(maxLength, mcalRawData.Length - offset));
        if (available <= 0)
            return null;

        if (!doNotFixAlphaMap && available >= 63 * 63 && available < 64 * 64)
            return ExpandFixedBigAlpha(mcalRawData, offset, available);

        var alpha = new byte[64 * 64];
        Buffer.BlockCopy(mcalRawData, offset, alpha, 0, Math.Min(alpha.Length, available));
        return alpha;
    }

    private static byte[] ExpandFixedBigAlpha(byte[] source, int offset, int available)
    {
        var alpha = new byte[64 * 64];
        int readPos = offset;
        int sourceEnd = Math.Min(source.Length, offset + available);

        for (int y = 0; y < 63 && readPos < sourceEnd; y++)
        {
            int rowStart = y * 64;
            int rowBytes = Math.Min(63, sourceEnd - readPos);
            Buffer.BlockCopy(source, readPos, alpha, rowStart, rowBytes);
            readPos += rowBytes;

            int edgeSource = rowStart + Math.Max(0, rowBytes - 1);
            alpha[rowStart + 63] = alpha[edgeSource];
        }

        Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);
        return alpha;
    }

    private static byte[]? DecodeLayerBySpan(byte[] mcalRawData, int offset, int? nextOffset, bool useBigAlpha, bool doNotFixAlphaMap)
    {
        if (offset < 0 || offset >= mcalRawData.Length)
            return null;

        int remaining = mcalRawData.Length - offset;
        if (remaining <= 0)
            return null;

        int span = nextOffset.HasValue
            ? nextOffset.Value - offset
            : remaining;

        bool decodeAsBigAlpha = useBigAlpha;
        if (span >= 4096)
            decodeAsBigAlpha = true;
        else if (span > 0 && span <= 2048)
            decodeAsBigAlpha = false;

        if (decodeAsBigAlpha)
        {
            int bytesToCopy = Math.Min(4096, remaining);
            var alpha = new byte[4096];
            Buffer.BlockCopy(mcalRawData, offset, alpha, 0, bytesToCopy);
            return alpha;
        }

        int packedCount = Math.Min(2048, remaining);
        if (packedCount <= 0)
            return null;

        var decoded = new byte[4096];
        for (int i = 0; i < packedCount; i++)
        {
            byte packed = mcalRawData[offset + i];
            decoded[i * 2] = (byte)((packed & 0x0F) * 17);
            decoded[i * 2 + 1] = (byte)(((packed >> 4) & 0x0F) * 17);
        }

        return doNotFixAlphaMap ? decoded : ApplyLegacyEdgeFix(decoded);
    }

    private static bool InferBigAlphaForChunk(Mcnk mcnk, bool wdtBigAlpha)
    {
        if (wdtBigAlpha)
            return true;

        int layerCount = mcnk.TextureLayers?.Count ?? (int)mcnk.Header.Layers;
        int alphaLayerCount = Math.Max(0, layerCount - 1);
        if (alphaLayerCount == 0)
            return false;

        if (mcnk.TextureLayers != null)
        {
            for (int i = 1; i < mcnk.TextureLayers.Count && i < 4; i++)
            {
                if ((mcnk.TextureLayers[i].Flags & MclyFlags.CompressedAlpha) != 0)
                    return true;
            }
        }

        uint sizeAlpha = mcnk.Header.SizeMcal >= 8
            ? mcnk.Header.SizeMcal - 8
            : (uint)(mcnk.McalRawData?.Length ?? 0);
        if (sizeAlpha != 0)
        {
            if (sizeAlpha % 4096u == 0 && (sizeAlpha / 4096u) >= (uint)alphaLayerCount)
                return true;

            if (sizeAlpha % 2048u == 0 && (sizeAlpha / 2048u) == (uint)alphaLayerCount)
                return false;

            if (sizeAlpha >= 4096u)
                return true;
        }

        int mcalLen = mcnk.McalRawData?.Length ?? 0;
        if (mcalLen >= 4096 && (mcalLen % 4096) == 0 && (mcalLen / 4096) >= alphaLayerCount)
            return true;

        return false;
    }

    private static byte[] ApplyLegacyEdgeFix(byte[] alpha)
    {
        if (alpha.Length != 64 * 64)
            return alpha;

        for (int y = 0; y < 64; y++)
            alpha[y * 64 + 63] = alpha[y * 64 + 62];

        Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);
        return alpha;
    }

    private static byte[]? ExtractShadowMap(byte[]? mcshData)
    {
        if (mcshData == null || mcshData.Length < 8) return null;

        // MCSH: 64 rows × 8 bytes = 512 bytes of shadow bits
        var shadow = new byte[64 * 64];
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int byteIndex = y * 8 + (x / 8);
                int bitIndex = x % 8;
                if (byteIndex < mcshData.Length)
                {
                    bool isShadowed = (mcshData[byteIndex] & (1 << bitIndex)) != 0;
                    shadow[y * 64 + x] = isShadowed ? (byte)255 : (byte)0;
                }
            }
        }
        return shadow;
    }

    /// <summary>
    /// Extract MCLQ inline liquid data from raw bytes.
    /// Supports both 3.3.5 single-instance format and 0.6.0 packed multi-instance format.
    /// 0.6.0: payload is packed instances (0x2D4 bytes each), one per liquid flag bit
    /// (0x04=River, 0x08=Ocean, 0x10=Magma, 0x20=Slime). Instance layout:
    ///   +0x000: minHeight (float), +0x004: maxHeight (float),
    ///   +0x008: 81 vertices × 8 bytes (648), +0x290: 64 tile flags, +0x2D0: trailing uint32.
    /// 3.3.5: single instance (8 + 81×8 + 64 = 720 bytes).
    /// </summary>
    private static LiquidChunkData? ExtractMclq(byte[] mclqData, uint mcnkFlags,
        int tileX, int tileY, int chunkX, int chunkY, Vector3 worldPos, float baseHeight,
        int layerStride, int tileFlagsOffset)
    {
        if (mclqData == null || mclqData.Length < 8)
            return null;

        // Determine which liquid types are present from MCNK flags
        uint[] liquidBits = { 0x04, 0x08, 0x10, 0x20 };
        LiquidType[] liquidTypes = { LiquidType.Water, LiquidType.Ocean, LiquidType.Magma, LiquidType.Slime };

        // Check if this is a packed multi-instance format (0.6.0)
        // by counting how many liquid flag bits are set
        int instanceCount = 0;
        for (int b = 0; b < 4; b++)
            if ((mcnkFlags & liquidBits[b]) != 0) instanceCount++;

        if (instanceCount == 0)
            return null;

        // Try to parse packed instances (profile-specific layer stride)
        // If data is large enough for packed format, use it; otherwise fall back to single-instance
        int packedSize = instanceCount * layerStride;
        bool usePacked = mclqData.Length >= packedSize && instanceCount > 0;

        // For single-instance (3.3.5 or single liquid type), also accept 720-byte format
        if (!usePacked && mclqData.Length >= 720)
            usePacked = false; // use legacy single-instance path below

        int offset = 0;
        LiquidChunkData? result = null;

        for (int b = 0; b < 4; b++)
        {
            if ((mcnkFlags & liquidBits[b]) == 0) continue;

            if (offset + 8 > mclqData.Length) break;

            float minHeight = BitConverter.ToSingle(mclqData, offset + 0);
            float maxHeight = BitConverter.ToSingle(mclqData, offset + 4);

            if (float.IsNaN(minHeight) || float.IsNaN(maxHeight))
            {
                if (usePacked) { offset += 0x2D4; continue; }
                else break;
            }

            LiquidType liquidType = liquidTypes[b];

            // Build 9×9 height grid from per-vertex data.
            // Each vertex entry is 8 bytes: first 4 = flow/depth union data,
            // second 4 = float height (absolute world Z).
            // Per-vertex heights can slope for waterfalls and adjacent water at different Z levels.
            var heights = new float[81];

            if (offset + 8 + 81 * 8 <= mclqData.Length)
            {
                for (int i = 0; i < 81; i++)
                {
                    int voff = offset + 8 + i * 8 + 4; // height is second float in 8-byte vertex
                    float h = BitConverter.ToSingle(mclqData, voff);
                    if (float.IsNaN(h) || MathF.Abs(h) > 50000f)
                        h = maxHeight; // fallback to maxHeight for bad data
                    heights[i] = h;
                }
            }
            else
            {
                // Not enough data for per-vertex grid — flat plane at maxHeight
                Array.Fill(heights, maxHeight);
            }

            // Read 64 tile flags at profile tile-flags offset (packed) or legacy packed tail.
            int tileFlagsOff = usePacked ? (offset + tileFlagsOffset) : (offset + 8 + 81 * 8);
            byte[] tileFlags = null;
            if (tileFlagsOff + 64 <= mclqData.Length)
            {
                tileFlags = new byte[64];
                Array.Copy(mclqData, tileFlagsOff, tileFlags, 0, 64);
            }

            // Check if all tiles hidden
            bool anyVisible = true;
            if (tileFlags != null)
            {
                anyVisible = false;
                for (int i = 0; i < 64; i++)
                {
                    if ((tileFlags[i] & 0x0F) != 0x0F) { anyVisible = true; break; }
                }
            }

            if (anyVisible && result == null)
            {
                result = new LiquidChunkData
                {
                    MinHeight = minHeight,
                    MaxHeight = maxHeight,
                    Heights = heights,
                    TileFlags = tileFlags,
                    Type = liquidType,
                    WorldPosition = worldPos,
                    TileX = tileX,
                    TileY = tileY,
                    ChunkX = chunkX,
                    ChunkY = chunkY
                };
            }

            if (usePacked) offset += layerStride;
            else break; // single instance, done
        }

        return result;
    }

    private static byte[] StripMclqChunkHeaderIfPresent(byte[] mclqData)
    {
        // Some builds/paths provide MCLQ as a full chunk: [FourCC][uint32 size][payload...]
        // 0.6.0 client code (Ghidra) treats MCLQ like a normal chunk and uses payload at +8.
        if (mclqData.Length < 8)
            return mclqData;

        bool isMclq = mclqData[0] == (byte)'M' && mclqData[1] == (byte)'C' && mclqData[2] == (byte)'L' && mclqData[3] == (byte)'Q';
        bool isReversed = mclqData[0] == (byte)'Q' && mclqData[1] == (byte)'L' && mclqData[2] == (byte)'C' && mclqData[3] == (byte)'M';
        if (!isMclq && !isReversed)
            return mclqData;

        uint size = BitConverter.ToUInt32(mclqData, 4);
        if (size == 0)
            return Array.Empty<byte>();

        // Basic sanity: size must fit in the provided buffer (ignore optional padding byte).
        int available = mclqData.Length - 8;
        if (size > (uint)available)
            return mclqData;

        var payload = new byte[size];
        Buffer.BlockCopy(mclqData, 8, payload, 0, (int)size);
        return payload;
    }

    /// <summary>
    /// Parse MH2O liquid data via MHDR offset +0x28 (Ghidra-verified FUN_007d6ef0).
    /// MH2O header: 256 entries × 12 bytes (one per MCNK chunk).
    /// Each entry: { uint32 ofsInformation, uint32 layerCount, uint32 ofsRender }
    /// SMLiquidInstance: { uint16 liquidType, uint16 liquidObject, float min, float max,
    ///   uint8 xOfs, uint8 yOfs, uint8 width, uint8 height, uint32 ofsMask, uint32 ofsHeightmap }
    /// </summary>
    private void ParseMh2o(byte[] adt, int mhdrStart, GillijimProject.WowFiles.Mhdr mhdr,
        int tileX, int tileY, float chunkSmall, TileLoadResult result)
    {
        int mh2oOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.Mh2oOffset);
        if (mh2oOff == 0)
        {
            if (ViewerLog.Verbose)
                ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) MH2O: MHDR offset is 0 (no MH2O)");
            return;
        }

        int mh2oAbs = mhdrStart + mh2oOff;
        if (mh2oAbs + 8 > adt.Length)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"  Tile({tileX},{tileY}) MH2O: abs pos {mh2oAbs} beyond file ({adt.Length})");
            return;
        }

        // Check FourCC at the position
        string mh2oSig = Encoding.ASCII.GetString(adt, mh2oAbs, 4);
        int mh2oSize = BitConverter.ToInt32(adt, mh2oAbs + 4);
        if (ViewerLog.Verbose)
            ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) MH2O: mhdrOff={mh2oOff}, absPos={mh2oAbs}, sig='{mh2oSig}', size={mh2oSize}");

        // MH2O chunk: skip FourCC(4) + size(4) to get data start
        int mh2oDataStart = mh2oAbs + 8;
        if (mh2oSize <= 0 || mh2oDataStart + mh2oSize > adt.Length) return;

        var mh2oPayload = new byte[mh2oSize];
        Buffer.BlockCopy(adt, mh2oDataStart, mh2oPayload, 0, mh2oSize);

        Mh2oChunk mh2o;
        try
        {
            mh2o = Mh2oChunk.Parse(mh2oPayload, _liquidVertexFormatChain);
        }
        catch (Exception ex)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain,
                $"  Tile({tileX},{tileY}) MH2O parse failed: {ex.Message}");
            return;
        }

        int liquidCount = 0;
        for (int ci = 0; ci < 256; ci++)
        {
            var instances = mh2o.GetInstancesForChunk(ci).ToArray();
            if (instances.Length == 0)
                continue;

            var matchingChunk = result.Chunks.FirstOrDefault(c => c.McinIndex == ci);
            int chunkX = matchingChunk?.ChunkX ?? (ci % 16);
            int chunkY = matchingChunk?.ChunkY ?? (ci / 16);
            Vector3 worldPos = matchingChunk?.WorldPosition
                ?? new Vector3(
                    WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall,
                    WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall,
                    0f);

            var liquid = BuildMh2oLiquid(instances, tileX, tileY, chunkX, chunkY, worldPos);
            if (liquid == null)
                continue;

            if (matchingChunk != null && matchingChunk.Liquid == null)
            {
                matchingChunk.Liquid = liquid;
                LegacyLiquidSoundEmitterFactory.Append(result.SoundEmitters, matchingChunk);
                liquidCount++;
            }
        }

        ReportUnresolvedVertexFormats(tileX, tileY);

        if (liquidCount > 0 && ViewerLog.Verbose)
            ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) MH2O: {liquidCount} liquid chunks");
    }

    private LiquidChunkData? BuildMh2oLiquid(
        IEnumerable<Mh2oInstance> instances,
        int tileX,
        int tileY,
        int chunkX,
        int chunkY,
        Vector3 worldPos)
    {
        var instanceList = instances
            .Where(static instance => instance.Width is > 0 and <= 8 && instance.Height is > 0 and <= 8)
            .ToArray();
        if (instanceList.Length == 0)
            return null;

        var heights = new float[81];
        var vertexRanks = new int[81];
        var tileFlags = new byte[64];
        var tileRanks = new int[64];
        Array.Fill(vertexRanks, int.MaxValue);
        Array.Fill(tileFlags, (byte)0x0F);
        Array.Fill(tileRanks, int.MaxValue);

        LiquidType chunkType = LiquidType.Water;
        int chunkTypeRank = int.MaxValue;
        float minHeight = float.PositiveInfinity;
        float maxHeight = float.NegativeInfinity;
        float fallbackHeight = 0f;
        bool hasVisibleTile = false;
        bool hasHeight = false;

        foreach (var instance in instanceList)
        {
            LiquidType liquidType = MapMh2oLiquidType(instance.LiquidTypeId);
            int precedence = GetMh2oLiquidPrecedence(liquidType);
            if (precedence < chunkTypeRank)
            {
                chunkType = liquidType;
                chunkTypeRank = precedence;
            }

            float instanceFallback = SelectMh2oFallbackHeight(instance);
            if (!hasHeight)
                fallbackHeight = instanceFallback;

            for (int localY = 0; localY < instance.Height; localY++)
            {
                for (int localX = 0; localX < instance.Width; localX++)
                {
                    if (!instance.TileExists(localX, localY))
                        continue;

                    int tileIndex = (instance.YOffset + localY) * 8 + (instance.XOffset + localX);
                    if ((uint)tileIndex >= 64u || precedence >= tileRanks[tileIndex])
                        continue;

                    tileRanks[tileIndex] = precedence;
                    tileFlags[tileIndex] = 0;
                    hasVisibleTile = true;
                }
            }

            for (int localY = 0; localY <= instance.Height; localY++)
            {
                for (int localX = 0; localX <= instance.Width; localX++)
                {
                    int vertexIndex = (instance.YOffset + localY) * 9 + (instance.XOffset + localX);
                    if ((uint)vertexIndex >= 81u || precedence >= vertexRanks[vertexIndex])
                        continue;

                    float vertexHeight = ReadMh2oVertexHeight(instance, localX, localY, instanceFallback);
                    heights[vertexIndex] = vertexHeight;
                    vertexRanks[vertexIndex] = precedence;
                    minHeight = MathF.Min(minHeight, vertexHeight);
                    maxHeight = MathF.Max(maxHeight, vertexHeight);
                    hasHeight = true;
                }
            }
        }

        if (!hasVisibleTile)
            return null;

        if (!hasHeight)
        {
            minHeight = fallbackHeight;
            maxHeight = fallbackHeight;
            Array.Fill(heights, fallbackHeight);
        }
        else
        {
            for (int i = 0; i < heights.Length; i++)
            {
                if (vertexRanks[i] == int.MaxValue)
                    heights[i] = fallbackHeight;
            }
        }

        return new LiquidChunkData
        {
            MinHeight = minHeight,
            MaxHeight = maxHeight,
            Heights = heights,
            TileFlags = tileFlags,
            Type = chunkType,
            WorldPosition = worldPos,
            TileX = tileX,
            TileY = tileY,
            ChunkX = chunkX,
            ChunkY = chunkY
        };
    }

    private static float ReadMh2oVertexHeight(Mh2oInstance instance, int localX, int localY, float fallbackHeight)
    {
        if (instance.HeightMap == null)
            return fallbackHeight;

        int width = instance.Width + 1;
        int index = localY * width + localX;
        if ((uint)index >= (uint)instance.HeightMap.Length)
            return fallbackHeight;

        float value = instance.HeightMap[index];
        if (float.IsNaN(value) || float.IsInfinity(value))
            return fallbackHeight;

        return value;
    }

    private static float SelectMh2oFallbackHeight(Mh2oInstance instance)
    {
        if (!float.IsNaN(instance.MaxHeightLevel) && !float.IsInfinity(instance.MaxHeightLevel))
            return instance.MaxHeightLevel;
        if (!float.IsNaN(instance.MinHeightLevel) && !float.IsInfinity(instance.MinHeightLevel))
            return instance.MinHeightLevel;
        return 0f;
    }

    /// <summary>
    /// Load the LiquidObject -&gt; LiquidType -&gt; LiquidMaterial chain that resolves MH2O vertex
    /// formats. A failure here is reported and leaves the chain empty; it never throws the map load.
    /// </summary>
    private void LoadLiquidVertexFormatChain(IDBCProvider dbcProvider, string dbdDir, string build)
    {
        try
        {
            _liquidVertexFormatChain = LiquidVertexFormatChain.Load(
                dbcProvider, dbdDir, build, out IReadOnlyList<string> diagnostics);

            foreach (string line in diagnostics)
                ViewerLog.Important(ViewerLog.Category.Dbc, $"[MH2O] {line}");

            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"[MH2O] Liquid vertex-format chain for build {build}: "
                + $"{_liquidVertexFormatChain.LiquidObjectRowCount} LiquidObject rows, "
                + $"{_liquidVertexFormatChain.LiquidTypeMaterialRowCount} LiquidType material rows, "
                + $"{_liquidVertexFormatChain.LiquidMaterialRowCount} LiquidMaterial rows.");
        }
        catch (Exception ex)
        {
            _liquidVertexFormatChain = LiquidVertexFormatChain.Empty;
            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"[MH2O] Failed to load the liquid vertex-format chain for build {build}: {ex.Message}. "
                + "Liquid layers carrying a LiquidObject id will keep the flat-plane fallback.");
        }
    }

    /// <summary>
    /// Report each distinct unresolved <c>liquid_object_or_lvf</c> once. Spec 205 FR-004: an
    /// encoding the reader cannot decode must be visible, not silently rendered as a flat plane.
    /// </summary>
    private void ReportUnresolvedVertexFormats(int tileX, int tileY)
    {
        foreach ((ushort rawValue, int count) in _liquidVertexFormatChain.UnresolvedCounts)
        {
            if (!_reportedUnresolvedVertexFormats.Add(rawValue))
                continue;

            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[MH2O] Tile({tileX},{tileY}): liquid_object_or_lvf {rawValue} did not resolve to a "
                + $"vertex format ({count} layers so far); those layers keep the flat fallback at the "
                + "header height. This is correct for depth-only water and wrong for anything sloped.");
        }
    }

    private LiquidType MapMh2oLiquidType(ushort liquidTypeId)
    {
        if (_mh2oLiquidTypesById.TryGetValue(liquidTypeId, out var liquidType))
            return liquidType;

        liquidType = MapMh2oLiquidTypeFallback();
        if (_mh2oLiquidTypeLookupAttempted && _reportedUnknownMh2oLiquidTypeIds.Add(liquidTypeId))
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[MH2O] LiquidType.dbc ID {liquidTypeId} was not present in the loaded LiquidType table for build {_buildVersion ?? "unknown"}; using the native missing-row default as {liquidType}.");
        }

        return liquidType;
    }

    private void LoadMh2oLiquidTypeLookup(IDBCProvider dbcProvider, string dbdDir, string build)
    {
        _mh2oLiquidTypeLookupAttempted = true;

        try
        {
            var dbdProvider = new FilesystemDBDProvider(dbdDir);
            var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

            IDBCDStorage storage;
            try
            {
                storage = dbcd.Load("LiquidType", build, Locale.EnUS);
            }
            catch
            {
                storage = dbcd.Load("LiquidType", build, Locale.None);
            }

            var availableColumns = new HashSet<string>(storage.AvailableColumns, StringComparer.OrdinalIgnoreCase);
            string? typeColumn = DetectColumn(availableColumns, "Type", "LiquidType", "TypeID");
            string? idColumn = DetectColumn(availableColumns, "ID", "Id", "LiquidTypeID", "LiquidTypeId");
            string? nameColumn = DetectColumn(availableColumns, "Name");
            string classificationRoute;

            if (!string.IsNullOrWhiteSpace(typeColumn))
            {
                foreach (var row in storage.Values)
                {
                    int rowId = SafeField<int>(row, idColumn, row.ID);
                    if ((uint)rowId > ushort.MaxValue)
                        continue;

                    int liquidClass = SafeField<int>(row, typeColumn, -1);
                    if (!TryMapLiquidTypeClass(liquidClass, out var mappedLiquidType))
                        continue;

                    _mh2oLiquidTypesById[(ushort)rowId] = mappedLiquidType;
                }

                classificationRoute = $"class column '{typeColumn}'";
                ViewerLog.Important(ViewerLog.Category.Dbc,
                    $"[MH2O] Resolved {_mh2oLiquidTypesById.Count} LiquidType.dbc rows for build {build} via {classificationRoute}; idColumn='{idColumn ?? "<row.ID>"}' nameColumn='{nameColumn ?? "<none>"}'.");
                return;
            }

            if (string.IsNullOrWhiteSpace(typeColumn))
            {
                ViewerLog.Important(ViewerLog.Category.Dbc,
                    $"[MH2O] LiquidType.dbc loaded for build {build}, but no recognizable family column was found. Resolving family from DBC row names.");
            }

            foreach (var row in storage.Values)
            {
                int rowId = SafeField<int>(row, idColumn, row.ID);
                if ((uint)rowId > ushort.MaxValue)
                    continue;

                ushort id = (ushort)rowId;
                _mh2oLiquidTypesById[id] = ClassifyLiquidTypeFromDbcRow(row, nameColumn);
            }

            classificationRoute = string.IsNullOrWhiteSpace(nameColumn)
                ? "DBC row records without a family column"
                : $"DBC row-name heuristics ('{nameColumn}')";

            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"[MH2O] Resolved {_mh2oLiquidTypesById.Count} LiquidType.dbc rows for build {build} via {classificationRoute}; idColumn='{idColumn ?? "<row.ID>"}' nameColumn='{nameColumn ?? "<none>"}'.");
        }
        catch (Exception ex)
        {
            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"[MH2O] Failed to load LiquidType.dbc for build {build}: {ex.Message}");
        }
    }

    private static bool TryMapLiquidTypeClass(int liquidClass, out LiquidType liquidType)
    {
        liquidType = liquidClass switch
        {
            0 => LiquidType.Water,
            1 => LiquidType.Ocean,
            2 => LiquidType.Magma,
            3 => LiquidType.Slime,
            _ => LiquidType.Water,
        };

        return liquidClass is >= 0 and <= 3;
    }

    private static LiquidType MapMh2oLiquidTypeFallback() =>
        // The native missing-row behavior is the water family. Do not infer
        // lava/slime from a guessed numeric ID; exact DBC rows own that data.
        LiquidType.Water;

    private static LiquidType ClassifyLiquidTypeFromDbcRow(dynamic row, string? nameColumn)
    {
        string name = SafeField<string>(row, nameColumn, string.Empty) ?? string.Empty;
        if (!string.IsNullOrWhiteSpace(name))
        {
            if (ContainsAnyInsensitive(name, "slime", "ooze"))
                return LiquidType.Slime;
            if (ContainsAnyInsensitive(name, "magma", "lava"))
                return LiquidType.Magma;
            if (ContainsAnyInsensitive(name, "ocean", "sea"))
                return LiquidType.Ocean;
            if (ContainsAnyInsensitive(name, "river", "water", "lake", "fast", "slow"))
                return LiquidType.Water;
        }

        return MapMh2oLiquidTypeFallback();
    }

    private static bool ContainsAnyInsensitive(string text, params string[] needles)
    {
        foreach (string needle in needles)
        {
            if (text.Contains(needle, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    private static int GetMh2oLiquidPrecedence(LiquidType liquidType)
    {
        return liquidType switch
        {
            LiquidType.Magma => 0,
            LiquidType.Slime => 1,
            LiquidType.Water => 2,
            LiquidType.Ocean => 3,
            _ => int.MaxValue,
        };
    }

    private static string? DetectColumn(ISet<string> availableColumns, params string[] candidates)
    {
        foreach (var col in candidates)
        {
            if (availableColumns.Contains(col))
                return col;
        }

        return null;
    }

    private static T SafeField<T>(dynamic row, string? col, T fallback)
    {
        if (string.IsNullOrWhiteSpace(col))
            return fallback;

        try { return (T)row[col]; }
        catch { return fallback; }
    }

    /// <summary>
    /// Ghidra-verified (FUN_007d6ef0): locate MDDF/MODF via MHDR offsets.
    /// Resolve names via MMID/MWID → MMDX/MWMO byte offsets.
    /// </summary>
    private void CollectPlacementsViaMhdr(byte[] adt, int mhdrStart,
        GillijimProject.WowFiles.Mhdr mhdr, int tileX, int tileY, TileLoadResult result)
    {
        // Read MHDR offsets (all relative to mhdrStart)
        int mmdxOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MmdxOffset);
        int mmidOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MmidOffset);
        int mwmoOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MwmoOffset);
        int mwidOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MwidOffset);
        int mddfOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MddfOffset);
        int modfOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.ModfOffset);

        // Resolve absolute positions (MHDR offset + 8 to skip chunk header = data start)
        // Ghidra: pbVar1 + *(pbVar1 + offset) + 8
        int mmdxAbs = mmdxOff != 0 ? mhdrStart + mmdxOff : -1;
        int mmidAbs = mmidOff != 0 ? mhdrStart + mmidOff : -1;
        int mwmoAbs = mwmoOff != 0 ? mhdrStart + mwmoOff : -1;
        int mwidAbs = mwidOff != 0 ? mhdrStart + mwidOff : -1;
        int mddfAbs = mddfOff != 0 ? mhdrStart + mddfOff : -1;
        int modfAbs = modfOff != 0 ? mhdrStart + modfOff : -1;

        // Parse MMDX string block (null-terminated strings)
        byte[]? mmdxData = null;
        if (mmdxAbs >= 0 && mmdxAbs + 8 <= adt.Length)
        {
            int mmdxSize = BitConverter.ToInt32(adt, mmdxAbs + 4);
            int mmdxDataStart = mmdxAbs + 8;
            if (mmdxSize > 0 && mmdxDataStart + mmdxSize <= adt.Length)
                mmdxData = new ReadOnlySpan<byte>(adt, mmdxDataStart, mmdxSize).ToArray();
        }

        // Parse MMID (array of uint32 offsets into MMDX)
        List<uint>? mmidEntries = null;
        if (mmidAbs >= 0 && mmidAbs + 8 <= adt.Length)
        {
            int mmidSize = BitConverter.ToInt32(adt, mmidAbs + 4);
            int mmidDataStart = mmidAbs + 8;
            if (mmidSize > 0 && mmidDataStart + mmidSize <= adt.Length)
            {
                int count = mmidSize / 4;
                mmidEntries = new List<uint>(count);
                for (int i = 0; i < count; i++)
                    mmidEntries.Add(BitConverter.ToUInt32(adt, mmidDataStart + i * 4));
            }
        }

        // Parse MWMO string block
        byte[]? mwmoData = null;
        if (mwmoAbs >= 0 && mwmoAbs + 8 <= adt.Length)
        {
            int mwmoSize = BitConverter.ToInt32(adt, mwmoAbs + 4);
            int mwmoDataStart = mwmoAbs + 8;
            if (mwmoSize > 0 && mwmoDataStart + mwmoSize <= adt.Length)
                mwmoData = new ReadOnlySpan<byte>(adt, mwmoDataStart, mwmoSize).ToArray();
        }

        // Parse MWID (array of uint32 offsets into MWMO)
        List<uint>? mwidEntries = null;
        if (mwidAbs >= 0 && mwidAbs + 8 <= adt.Length)
        {
            int mwidSize = BitConverter.ToInt32(adt, mwidAbs + 4);
            int mwidDataStart = mwidAbs + 8;
            if (mwidSize > 0 && mwidDataStart + mwidSize <= adt.Length)
            {
                int count = mwidSize / 4;
                mwidEntries = new List<uint>(count);
                for (int i = 0; i < count; i++)
                    mwidEntries.Add(BitConverter.ToUInt32(adt, mwidDataStart + i * 4));
            }
        }

        // Parse MDDF
        if (mddfAbs >= 0 && mddfAbs + 8 <= adt.Length)
        {
            int mddfSize = BitConverter.ToInt32(adt, mddfAbs + 4);
            int mddfDataStart = mddfAbs + 8;
            if (mddfSize >= _adtProfile.MddfRecordSize && mddfDataStart + mddfSize <= adt.Length)
                ParseMddfViaMmid(adt, mddfDataStart, mddfSize, mmdxData, mmidEntries, result, _adtProfile.MddfRecordSize);
        }

        // Parse MODF
        if (modfAbs >= 0 && modfAbs + 8 <= adt.Length)
        {
            int modfSize = BitConverter.ToInt32(adt, modfAbs + 4);
            int modfDataStart = modfAbs + 8;
            if (modfSize >= _adtProfile.ModfRecordSize && modfDataStart + modfSize <= adt.Length)
                ParseModfViaMwid(adt, modfDataStart, modfSize, mwmoData, mwidEntries, result, _adtProfile.ModfRecordSize);
        }

        if (ViewerLog.Verbose)
            ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) MHDR placements: mddf={mddfOff != 0}, modf={modfOff != 0}, mmid={mmidEntries?.Count ?? 0}, mwid={mwidEntries?.Count ?? 0}");
    }

    private void CollectPlacementsFlat(byte[] objBytes, int tileX, int tileY, TileLoadResult result)
    {
        int mmdxOff = FindChunk(objBytes, "MMDX");
        int mmidOff = FindChunk(objBytes, "MMID");
        int mwmoOff = FindChunk(objBytes, "MWMO");
        int mwidOff = FindChunk(objBytes, "MWID");
        int mddfOff = FindChunk(objBytes, "MDDF");
        int modfOff = FindChunk(objBytes, "MODF");

        byte[]? mmdxData = null;
        if (mmdxOff >= 0 && mmdxOff + 8 <= objBytes.Length)
        {
            int sz = BitConverter.ToInt32(objBytes, mmdxOff + 4);
            if (sz > 0 && mmdxOff + 8 + sz <= objBytes.Length)
                mmdxData = new ReadOnlySpan<byte>(objBytes, mmdxOff + 8, sz).ToArray();
        }

        List<uint>? mmidEntries = null;
        if (mmidOff >= 0 && mmidOff + 8 <= objBytes.Length)
        {
            int sz = BitConverter.ToInt32(objBytes, mmidOff + 4);
            if (sz > 0 && mmidOff + 8 + sz <= objBytes.Length)
            {
                int count = sz / 4;
                mmidEntries = new List<uint>(count);
                for (int i = 0; i < count; i++)
                    mmidEntries.Add(BitConverter.ToUInt32(objBytes, mmidOff + 8 + i * 4));
            }
        }

        byte[]? mwmoData = null;
        if (mwmoOff >= 0 && mwmoOff + 8 <= objBytes.Length)
        {
            int sz = BitConverter.ToInt32(objBytes, mwmoOff + 4);
            if (sz > 0 && mwmoOff + 8 + sz <= objBytes.Length)
                mwmoData = new ReadOnlySpan<byte>(objBytes, mwmoOff + 8, sz).ToArray();
        }

        List<uint>? mwidEntries = null;
        if (mwidOff >= 0 && mwidOff + 8 <= objBytes.Length)
        {
            int sz = BitConverter.ToInt32(objBytes, mwidOff + 4);
            if (sz > 0 && mwidOff + 8 + sz <= objBytes.Length)
            {
                int count = sz / 4;
                mwidEntries = new List<uint>(count);
                for (int i = 0; i < count; i++)
                    mwidEntries.Add(BitConverter.ToUInt32(objBytes, mwidOff + 8 + i * 4));
            }
        }

        if (mddfOff >= 0 && mddfOff + 8 <= objBytes.Length)
        {
            int sz = BitConverter.ToInt32(objBytes, mddfOff + 4);
            if (sz >= _adtProfile.MddfRecordSize && mddfOff + 8 + sz <= objBytes.Length)
                ParseMddfViaMmid(objBytes, mddfOff + 8, sz, mmdxData, mmidEntries, result, _adtProfile.MddfRecordSize);
        }

        if (modfOff >= 0 && modfOff + 8 <= objBytes.Length)
        {
            int sz = BitConverter.ToInt32(objBytes, modfOff + 4);
            if (sz >= _adtProfile.ModfRecordSize && modfOff + 8 + sz <= objBytes.Length)
                ParseModfViaMwid(objBytes, modfOff + 8, sz, mwmoData, mwidEntries, result, _adtProfile.ModfRecordSize);
        }

        if (ViewerLog.Verbose)
            ViewerLog.Trace($"[Terrain] Tile({tileX},{tileY}) flat placements: mddf={mddfOff >= 0}, modf={modfOff >= 0}, mmid={mmidEntries?.Count ?? 0}, mwid={mwidEntries?.Count ?? 0}");
    }

    /// <summary>
    /// Resolve a name from xID[index] → byte offset into string block.
    /// </summary>
    private static string ResolveNameViaXid(uint nameId, List<uint>? xidEntries, byte[]? stringBlock)
    {
        if (xidEntries == null || stringBlock == null || nameId >= xidEntries.Count)
            return $"unknown_{nameId}";

        uint byteOffset = xidEntries[(int)nameId];
        if (byteOffset >= stringBlock.Length)
            return $"unknown_{nameId}";

        // Read null-terminated string
        int end = (int)byteOffset;
        while (end < stringBlock.Length && stringBlock[end] != 0)
            end++;
        return Encoding.ASCII.GetString(stringBlock, (int)byteOffset, end - (int)byteOffset);
    }

    private void ParseMddfViaMmid(byte[] data, int offset, int size,
        byte[]? mmdxData, List<uint>? mmidEntries, TileLoadResult result, int entrySize)
    {
        int count = size / entrySize;

        for (int i = 0; i < count; i++)
        {
            int pos = offset + i * entrySize;
            if (pos + entrySize > data.Length) break;

            uint nameId = BitConverter.ToUInt32(data, pos);
            uint uniqueId = BitConverter.ToUInt32(data, pos + 4);
            // MDDF position is (X, Z, Y): X=North, Z=Up(height), Y=West
            // Confirmed: LkToAlphaConverter passes positions unchanged → same layout as Alpha
            float rawX = BitConverter.ToSingle(data, pos + 8);   // North
            float rawZ = BitConverter.ToSingle(data, pos + 12);  // Up (height)
            float rawY = BitConverter.ToSingle(data, pos + 16);  // West
            // Rotation stored as (X, Z, Y) — same layout as position
            float rotX = BitConverter.ToSingle(data, pos + 20);
            float rotZ = BitConverter.ToSingle(data, pos + 24);
            float rotY = BitConverter.ToSingle(data, pos + 28);
            ushort scale = BitConverter.ToUInt16(data, pos + 32);

            lock (_placementLock)
            {
                string name = ResolveNameViaXid(nameId, mmidEntries, mmdxData);
                int nameIdx = GetOrAddMdxName(name);

                // Convert WoW coords to renderer: rendererX=MapOrigin-wowY, rendererY=MapOrigin-wowX, rendererZ=wowZ
                var placement = new MddfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(
                        WoWConstants.MapOrigin - rawY,
                        WoWConstants.MapOrigin - rawX,
                        rawZ),
                    Rotation = new Vector3(rotX, rotY, rotZ),
                    Scale = scale / 1024f
                };

                MddfPlacements.Add(placement);
                result.MddfPlacements.Add(placement);
            }
        }
    }

    private void ParseModfViaMwid(byte[] data, int offset, int size,
        byte[]? mwmoData, List<uint>? mwidEntries, TileLoadResult result, int entrySize)
    {
        int count = size / entrySize;

        for (int i = 0; i < count; i++)
        {
            int pos = offset + i * entrySize;
            if (pos + entrySize > data.Length) break;

            uint nameId = BitConverter.ToUInt32(data, pos);
            uint uniqueId = BitConverter.ToUInt32(data, pos + 4);
            // MODF position is (X, Z, Y): X=North, Z=Up(height), Y=West
            // Confirmed: LkToAlphaConverter passes positions unchanged → same layout as Alpha
            float rawX = BitConverter.ToSingle(data, pos + 8);   // North
            float rawZ = BitConverter.ToSingle(data, pos + 12);  // Up (height)
            float rawY = BitConverter.ToSingle(data, pos + 16);  // West
            // Rotation stored as (X, Z, Y)
            float rotX = BitConverter.ToSingle(data, pos + 20);
            float rotZ = BitConverter.ToSingle(data, pos + 24);
            float rotY = BitConverter.ToSingle(data, pos + 28);

            // Bounds stored as (X, Z, Y) pairs
            float bbMinX = BitConverter.ToSingle(data, pos + 32);
            float bbMinZ = BitConverter.ToSingle(data, pos + 36);
            float bbMinY = BitConverter.ToSingle(data, pos + 40);
            float bbMaxX = BitConverter.ToSingle(data, pos + 44);
            float bbMaxZ = BitConverter.ToSingle(data, pos + 48);
            float bbMaxY = BitConverter.ToSingle(data, pos + 52);
            ushort flags = BitConverter.ToUInt16(data, pos + 56);

            lock (_placementLock)
            {
                string name = ResolveNameViaXid(nameId, mwidEntries, mwmoData);
                int nameIdx = GetOrAddWmoName(name);

                // Convert WoW coords to renderer: rendererX=MapOrigin-wowY, rendererY=MapOrigin-wowX, rendererZ=wowZ
                // Note: MapOrigin-min > MapOrigin-max, so swap min/max after conversion
                var placement = new ModfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(
                        WoWConstants.MapOrigin - rawY,
                        WoWConstants.MapOrigin - rawX,
                        rawZ),
                    Rotation = new Vector3(rotX, rotY, rotZ),
                    BoundsMin = new Vector3(
                        WoWConstants.MapOrigin - bbMaxY,
                        WoWConstants.MapOrigin - bbMaxX,
                        bbMinZ),
                    BoundsMax = new Vector3(
                        WoWConstants.MapOrigin - bbMinY,
                        WoWConstants.MapOrigin - bbMinX,
                        bbMaxZ),
                    Flags = flags
                };

                ModfPlacements.Add(placement);
                result.ModfPlacements.Add(placement);
            }
        }
    }

    private int GetOrAddMdxName(string name)
    {
        if (_mdxNameIndex.TryGetValue(name, out int idx)) return idx;
        idx = _mdxNames.Count;
        _mdxNames.Add(name);
        _mdxNameIndex[name] = idx;
        return idx;
    }

    private int GetOrAddWmoName(string name)
    {
        if (_wmoNameIndex.TryGetValue(name, out int idx)) return idx;
        idx = _wmoNames.Count;
        _wmoNames.Add(name);
        _wmoNameIndex[name] = idx;
        return idx;
    }

    // ── WDT Parsing ──

    /// <summary>
    /// Parse MWMO + MODF from the WDT for global WMO placements.
    /// 0.6.0 WDTs use the same layout as Alpha WDTs:
    ///   MWMO = null-terminated WMO filename string block
    ///   MODF = 64-byte placement entries (nameIndex is a WDT-level WMO table index)
    /// </summary>
    private void ParseWdtWmoPlacement(byte[] wdt, bool useRawWorldCoordinates)
    {
        // Find MWMO chunk — WMO name string block
        int mwmoOff = FindChunk(wdt, "MWMO");
        if (mwmoOff < 0)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                useRawWorldCoordinates ? "  WMO-only: no MWMO chunk found in WDT" : "  WDT global WMO: no MWMO chunk found in WDT");
            return;
        }
        int mwmoSize = BitConverter.ToInt32(wdt, mwmoOff + 4);
        int mwmoData = mwmoOff + 8;
        if (mwmoSize <= 0 || mwmoData + mwmoSize > wdt.Length) return;

        // Parse null-terminated strings from MWMO block
        var wmoNames = ParseNullStrings(wdt, mwmoData, mwmoSize);
        foreach (var name in wmoNames)
            GetOrAddWmoName(name);

        if (_wmoNames.Count == 0)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                useRawWorldCoordinates ? "  WMO-only: MWMO contained no names" : "  WDT global WMO: MWMO contained no names");
            return;
        }

        ViewerLog.Important(ViewerLog.Category.Terrain,
            useRawWorldCoordinates
                ? $"  WMO-only: MWMO has {wmoNames.Count} names: {string.Join(", ", wmoNames.Select(Path.GetFileName))}"
                : $"  WDT global WMO: MWMO has {wmoNames.Count} names: {string.Join(", ", wmoNames.Select(Path.GetFileName))}");

        // Find MODF chunk — 64-byte placement entries
        int modfOff = FindChunk(wdt, "MODF");
        if (modfOff < 0)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                useRawWorldCoordinates ? "  WMO-only: no MODF chunk found in WDT" : "  WDT global WMO: no MODF chunk found in WDT");
            return;
        }
        int modfSize = BitConverter.ToInt32(wdt, modfOff + 4);
        int modfData = modfOff + 8;
        if (modfSize < 64 || modfData + modfSize > wdt.Length) return;

        int entryCount = modfSize / 64;
        for (int i = 0; i < entryCount; i++)
        {
            int pos = modfData + i * 64;
            if (pos + 64 > wdt.Length) break;

            uint nameId = BitConverter.ToUInt32(wdt, pos);
            uint uniqueId = BitConverter.ToUInt32(wdt, pos + 4);
            float rawX = BitConverter.ToSingle(wdt, pos + 8);
            float rawZ = BitConverter.ToSingle(wdt, pos + 12); // height
            float rawY = BitConverter.ToSingle(wdt, pos + 16);
            float rotX = BitConverter.ToSingle(wdt, pos + 20);
            float rotZ = BitConverter.ToSingle(wdt, pos + 24);
            float rotY = BitConverter.ToSingle(wdt, pos + 28);
            float bbMinX = BitConverter.ToSingle(wdt, pos + 32);
            float bbMinZ = BitConverter.ToSingle(wdt, pos + 36);
            float bbMinY = BitConverter.ToSingle(wdt, pos + 40);
            float bbMaxX = BitConverter.ToSingle(wdt, pos + 44);
            float bbMaxZ = BitConverter.ToSingle(wdt, pos + 48);
            float bbMaxY = BitConverter.ToSingle(wdt, pos + 52);
            ushort flags = BitConverter.ToUInt16(wdt, pos + 56);

            // For WDT-level MODF, nameId is an index into the WDT MWMO table.
            int nameIdx = (int)nameId;
            if (nameIdx < 0 || nameIdx >= _wmoNames.Count)
                nameIdx = 0;

            ModfPlacement placement;
            if (useRawWorldCoordinates)
            {
                // WMO-only maps keep raw WoW world coordinates.
                placement = new ModfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(rawX, rawY, rawZ),
                    Rotation = new Vector3(rotX, rotY, rotZ),
                    BoundsMin = new Vector3(
                        MathF.Min(bbMinX, bbMaxX), MathF.Min(bbMinY, bbMaxY), MathF.Min(bbMinZ, bbMaxZ)),
                    BoundsMax = new Vector3(
                        MathF.Max(bbMinX, bbMaxX), MathF.Max(bbMinY, bbMaxY), MathF.Max(bbMinZ, bbMaxZ)),
                    Flags = flags
                };
            }
            else
            {
                // Terrain maps use the same renderer-space conversion as ADT MODF placements.
                placement = new ModfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(
                        WoWConstants.MapOrigin - rawY,
                        WoWConstants.MapOrigin - rawX,
                        rawZ),
                    Rotation = new Vector3(rotX, rotY, rotZ),
                    BoundsMin = new Vector3(
                        WoWConstants.MapOrigin - bbMaxY,
                        WoWConstants.MapOrigin - bbMaxX,
                        bbMinZ),
                    BoundsMax = new Vector3(
                        WoWConstants.MapOrigin - bbMinY,
                        WoWConstants.MapOrigin - bbMinX,
                        bbMaxZ),
                    Flags = flags
                };
            }

            ModfPlacements.Add(placement);
            ViewerLog.Important(ViewerLog.Category.Terrain,
                useRawWorldCoordinates
                    ? $"  WMO-only MODF[{i}]: name={_wmoNames[nameIdx]} pos=({rawX:F1},{rawY:F1},{rawZ:F1}) rot=({rotX:F1},{rotY:F1},{rotZ:F1})"
                    : $"  WDT global MODF[{i}]: name={_wmoNames[nameIdx]} pos=({placement.Position.X:F1},{placement.Position.Y:F1},{placement.Position.Z:F1}) rot=({rotX:F1},{rotY:F1},{rotZ:F1}) flags=0x{flags:X4}");
        }
    }

    private static uint ReadMphdFlags(byte[] wdtBytes)
    {
        int off = FindChunk(wdtBytes, "MPHD");
        if (off >= 0 && off + 12 < wdtBytes.Length)
            return BitConverter.ToUInt32(wdtBytes, off + 8);
        return 0;
    }

    private static List<int> ReadMainChunk(byte[] wdtBytes)
    {
        var tiles = new List<int>();
        int mainOff = FindChunk(wdtBytes, "MAIN");
        if (mainOff < 0) return tiles;

        int mainSize = BitConverter.ToInt32(wdtBytes, mainOff + 4);
        int mainData = mainOff + 8;

        if (mainSize < 64 * 64 * 8) return tiles;

        for (int i = 0; i < 64 * 64; i++)
        {
            int entryOff = mainData + (i * 8);
            if (entryOff + 8 > wdtBytes.Length) break;

            uint flags = BitConverter.ToUInt32(wdtBytes, entryOff);
            if (flags != 0)
            {
                // Ghidra-verified (FUN_007b5950): MAIN is row-major [y][x].
                // Store raw index i = y*64+x. TerrainManager decodes:
                //   tx = i/64 = y (row), ty = i%64 = x (col).
                // This matches Alpha convention where tx=row, ty=col.
                tiles.Add(i);
            }
        }
        return tiles;
    }

    /// <summary>
    /// Find a chunk in LK format (reversed FourCC on disk).
    /// </summary>
    private static int FindChunk(byte[] bytes, string fourCC)
    {
        string reversed = new string(fourCC.Reverse().ToArray());

        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            if (size < 0) break;

            if (fcc == reversed)
                return i;

            // Native 5.0.1 walks chunks as next = payload + size with no alignment
            // padding (MapAdtFileData.cpp FUN_00bb6f10, MapChunk.cpp FUN_00ba3050:
            // remaining -= 8 + size, then assert dataSize == 0). Padding an odd-sized
            // chunk desyncs the walk and silently truncates the tile.
            int next = i + 8 + size;
            if (next <= i) break;
            i = next;
        }
        return -1;
    }

    private static List<string> ParseMtexViaMhdr(byte[] adtBytes, int mhdrStart, GillijimProject.WowFiles.Mhdr mhdr)
    {
        int mtexOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MtexOffset);
        if (mtexOff == 0)
            return new List<string>();

        int mtexAbs = mhdrStart + mtexOff;
        if (mtexAbs + 8 > adtBytes.Length)
            return new List<string>();

        string mtexSig = Encoding.ASCII.GetString(adtBytes, mtexAbs, 4);
        if (mtexSig != "XETM")
            return new List<string>();

        int mtexSize = BitConverter.ToInt32(adtBytes, mtexAbs + 4);
        int mtexData = mtexAbs + 8;
        if (mtexSize <= 0 || mtexData + mtexSize > adtBytes.Length)
            return new List<string>();

        return ParseNullStrings(adtBytes, mtexData, mtexSize);
    }

    private static List<string> ParseNullStrings(byte[] data, int offset, int size)
    {
        var result = new List<string>();
        int start = offset;
        int end = offset + size;
        for (int i = offset; i < end; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                    result.Add(Encoding.ASCII.GetString(data, start, i - start));
                start = i + 1;
            }
        }
        return result;
    }
}
