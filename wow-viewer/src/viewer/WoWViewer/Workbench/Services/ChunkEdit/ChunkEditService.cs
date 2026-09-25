using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Terrain chunk tools: selection, copy/paste clipboard (single and set), height inversion, dirty-tile tracking and heightmap output saving.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class ChunkEditService
{
    private readonly IViewerAppHost _host;

    internal ChunkEditService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private ref ChunkClipboard? _chunkClipboard => ref _host.ChunkClipboard;
    private ref (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardCopiedKey => ref _host.ChunkClipboardCopiedKey;
    private ref (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardLockedTargetKey => ref _host.ChunkClipboardLockedTargetKey;
    private ref ChunkClipboardSet? _chunkClipboardSet => ref _host.ChunkClipboardSet;
    private ref bool _chunkClipboardShowOverlay => ref _host.ChunkClipboardShowOverlay;
    private ref string _chunkClipboardStatus => ref _host.ChunkClipboardStatus;
    private ref bool _chunkToolEnabled => ref _host.ChunkToolEnabled;
    private HashSet<(int tileX, int tileY, int chunkX, int chunkY)> _selectedChunks => _host.SelectedChunks;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private TerrainTileIoService _terrainTileIo => _host.TerrainTileIo;
    private TerrainWeakSignalRestoreService _terrainWeakSignalRestore => _host.TerrainWeakSignalRestore;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private string EnsureEditorProjectOutputDirectory(bool forceNew = false) => _host.EnsureEditorProjectOutputDirectory(forceNew);
    private string GetEditorProjectName(string? fallbackName = null) => _host.GetEditorProjectName(fallbackName);
    private string? GetEditorProjectSourceKey() => _host.GetEditorProjectSourceKey();
    private bool TryPickTerrainChunkUnderMouse(TerrainRenderer renderer, out TerrainRenderer.TerrainChunkInfo info) => _host.TryPickTerrainChunkUnderMouse(renderer, out info);

    private bool _chunkClipboardUseMouse;
    private bool _chunkClipboardPasteRelativeHeights = true;
    private bool _chunkClipboardIncludeAlphaShadow;
    private bool _chunkClipboardIncludeTextures;
    private int _chunkClipboardSelectionRotation;
    private readonly Dictionary<(int tileX, int tileY), HashSet<(int chunkX, int chunkY)>> _chunkClipboardDirtyTileChunks = new();
    private string _chunkClipboardLastSaveFolder = string.Empty;

    internal sealed class HeightmapMetadata
    {
        public int Version { get; set; } = 1;
        public int Resolution { get; set; } = TerrainHeightmapIo.TileHeightmapSize;
        public float MinHeight { get; set; }
        public float MaxHeight { get; set; }
        public string Normalization { get; set; } = "per_tile";
    }

    private sealed class ChunkToolHeightmapSaveManifest
    {
        public int Version { get; set; } = 1;
        public string Format { get; set; } = "heightmap_257_per_tile";
        public string ProjectName { get; set; } = string.Empty;
        public string SourceKey { get; set; } = string.Empty;
        public string GeneratedUtc { get; set; } = string.Empty;
        public List<ChunkToolHeightmapSaveTile> Tiles { get; set; } = new();
    }

    private sealed class ChunkToolHeightmapSaveTile
    {
        public int TileX { get; set; }
        public int TileY { get; set; }
        public List<string> EditedChunks { get; set; } = new();
        public string HeightmapPng { get; set; } = string.Empty;
        public string HeightmapMetadataJson { get; set; } = string.Empty;
        public float MinHeight { get; set; }
        public float MaxHeight { get; set; }
    }

    private TerrainRenderer.TerrainChunkInfo? GetChunkClipboardTarget(TerrainRenderer renderer)
    {
        if (_chunkClipboardUseMouse && TryPickTerrainChunkUnderMouse(renderer, out var mouseChunk))
            return mouseChunk;
        return renderer.GetChunkInfoAt(_camera.Position.X, _camera.Position.Y);
    }

    internal bool TryLockChunkPasteTarget(TerrainRenderer renderer)
    {
        if (!TryPickTerrainChunkUnderMouse(renderer, out var info))
            return false;

        _chunkClipboardLockedTargetKey = (info.TileX, info.TileY, info.ChunkX, info.ChunkY);
        _chunkClipboardStatus = $"Locked paste target: tile({info.TileX},{info.TileY}) chunk({info.ChunkX},{info.ChunkY})";
        return true;
    }

    internal void ExecuteChunkClipboardCopy(TerrainRenderer renderer)
    {
        if (_selectedChunks.Count > 0)
        {
            CopySelectedChunks(renderer);
            return;
        }

        CopyChunkAtTarget(renderer);
    }

    internal void ExecuteChunkClipboardPaste(TerrainRenderer renderer)
    {
        if (_chunkClipboardLockedTargetKey == null)
        {
            _chunkClipboardStatus = "Paste blocked: lock a paste target with Ctrl+LMB.";
            return;
        }

        if (_chunkClipboardSet != null)
            PasteClipboardSetAtTarget(renderer);
        else
            PasteChunkAtTarget(renderer);
    }

    private void InvertSelectedChunkHeights(TerrainRenderer renderer)
    {
        List<(int tileX, int tileY, int chunkX, int chunkY)> targets = GetChunkEditTargets(renderer);
        if (targets.Count == 0)
        {
            _chunkClipboardStatus = "Invert Z failed: select chunk(s) or point at a loaded chunk.";
            return;
        }

        var groupedTargets = targets
            .GroupBy(target => (target.tileX, target.tileY))
            .OrderBy(group => group.Key.tileX)
            .ThenBy(group => group.Key.tileY);

        int inverted = 0;
        int skipped = 0;

        foreach (var group in groupedTargets)
        {
            if (!TryGetTileChunksForEdit(group.Key.tileX, group.Key.tileY, out var chunks))
            {
                skipped += group.Count();
                continue;
            }

            var newChunks = chunks.ToList();
            var editedChunks = new List<(int chunkX, int chunkY)>();

            foreach (var target in group)
            {
                int idx = newChunks.FindIndex(chunk => chunk.ChunkX == target.chunkX && chunk.ChunkY == target.chunkY);
                if (idx < 0)
                {
                    skipped++;
                    continue;
                }

                var sourceChunk = newChunks[idx];
                var invertedHeights = new float[sourceChunk.Heights.Length];
                for (int i = 0; i < sourceChunk.Heights.Length; i++)
                    invertedHeights[i] = -sourceChunk.Heights[i];

                var invertedNormals = TerrainChunkMath.GenerateNormalsForChunk(sourceChunk, invertedHeights, sourceChunk.HoleMask);
                newChunks[idx] = TerrainChunkMath.CloneTerrainChunk(
                    sourceChunk,
                    heights: invertedHeights,
                    normals: invertedNormals);
                editedChunks.Add((target.chunkX, target.chunkY));
                inverted++;
            }

            if (editedChunks.Count > 0)
                ApplyEditedTileChunks(group.Key.tileX, group.Key.tileY, newChunks, editedChunks);
        }

        _chunkClipboardStatus = $"Inverted Z for {inverted} chunk(s)"
            + (skipped > 0 ? $" (skipped {skipped})" : string.Empty)
            + ". Save edited heightmaps from the chunk tool when you want reusable outputs.";
    }

    private void CopyChunkAtTarget(TerrainRenderer renderer)
    {
        var targetChunk = GetChunkClipboardTarget(renderer);
        if (!targetChunk.HasValue)
        {
            _chunkClipboardStatus = "Copy failed: no loaded chunk at target.";
            return;
        }

        var key = targetChunk.Value;

        if (!TryGetChunkData(key.TileX, key.TileY, key.ChunkX, key.ChunkY, out var chunk))
        {
            _chunkClipboardStatus = $"Copy failed: chunk data not available for tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY}).";
            return;
        }

        _chunkClipboard = new ChunkClipboard(chunk);
        _chunkClipboardSet = null;
        _chunkClipboardCopiedKey = (key.TileX, key.TileY, key.ChunkX, key.ChunkY);
        _chunkClipboardStatus = $"Copied: tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY})";
    }

    private List<(int tileX, int tileY, int chunkX, int chunkY)> GetChunkEditTargets(TerrainRenderer renderer)
    {
        if (_selectedChunks.Count > 0)
        {
            return _selectedChunks
                .OrderBy(chunk => chunk.tileX)
                .ThenBy(chunk => chunk.tileY)
                .ThenBy(chunk => chunk.chunkX)
                .ThenBy(chunk => chunk.chunkY)
                .ToList();
        }

        var targetChunk = GetChunkClipboardTarget(renderer);
        if (!targetChunk.HasValue)
            return new List<(int tileX, int tileY, int chunkX, int chunkY)>();

        var chunk = targetChunk.Value;
        return new List<(int tileX, int tileY, int chunkX, int chunkY)>
        {
            (chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY)
        };
    }

    internal bool TryHandleChunkSelectionClick(TerrainRenderer renderer, bool shift)
    {
        if (!TryPickTerrainChunkUnderMouse(renderer, out var info))
            return false;

        var key = (info.TileX, info.TileY, info.ChunkX, info.ChunkY);
        if (shift)
        {
            if (!_selectedChunks.Add(key))
                _selectedChunks.Remove(key);
        }
        else
        {
            _selectedChunks.Clear();
            _selectedChunks.Add(key);
        }

        _chunkClipboardStatus = $"Selected {_selectedChunks.Count} chunk(s)";
        _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
        return true;
    }

    private void CopySelectedChunks(TerrainRenderer renderer)
    {
        if (_selectedChunks.Count == 0)
            return;

        int minGlobalX = int.MaxValue;
        int minGlobalY = int.MaxValue;
        foreach (var (tx, ty, cx, cy) in _selectedChunks)
        {
            int gx = tx * 16 + cx;
            int gy = ty * 16 + cy;
            minGlobalX = Math.Min(minGlobalX, gx);
            minGlobalY = Math.Min(minGlobalY, gy);
        }

        var set = new ChunkClipboardSet(minGlobalX, minGlobalY);
        int copied = 0;

        foreach (var (stx, sty, scx, scy) in _selectedChunks)
        {
            if (!TryGetChunkData(stx, sty, scx, scy, out var chunk))
                continue;

            int gx = stx * 16 + scx;
            int gy = sty * 16 + scy;
            set.Chunks[(gx - minGlobalX, gy - minGlobalY)] = new ChunkClipboard(chunk);
            copied++;
        }

        if (copied == 0)
        {
            _chunkClipboardStatus = "Copy failed: selection chunks not available.";
            return;
        }

        _chunkClipboardSet = set;
        _chunkClipboard = null;
        _chunkClipboardCopiedKey = _selectedChunks.First();
        _chunkClipboardStatus = $"Copied selection: {copied} chunk(s).";
    }

    private void PasteClipboardSetAtTarget(TerrainRenderer renderer)
    {
        if (_chunkClipboardSet == null)
            return;

        if (_chunkClipboardLockedTargetKey == null)
        {
            _chunkClipboardStatus = "Paste blocked: lock a paste target with Ctrl+LMB.";
            return;
        }

        int targetGlobalX = _chunkClipboardLockedTargetKey.Value.tileX * 16 + _chunkClipboardLockedTargetKey.Value.chunkX;
        int targetGlobalY = _chunkClipboardLockedTargetKey.Value.tileY * 16 + _chunkClipboardLockedTargetKey.Value.chunkY;

        int maxDx = 0;
        int maxDy = 0;
        foreach (var key in _chunkClipboardSet.Chunks.Keys)
        {
            maxDx = Math.Max(maxDx, key.dx);
            maxDy = Math.Max(maxDy, key.dy);
        }
        int width = maxDx + 1;
        int height = maxDy + 1;

        int srcGridW = width * 16 + 1;
        int srcGridH = height * 16 + 1;
        var sum = new float[srcGridW * srcGridH];
        var count = new ushort[srcGridW * srcGridH];

        foreach (var kvp in _chunkClipboardSet.Chunks)
        {
            int baseX = kvp.Key.dx * 16;
            int baseY = kvp.Key.dy * 16;
            var clip = kvp.Value;
            if (clip.Heights == null || clip.Heights.Length < 145)
                continue;

            for (int i = 0; i < 145; i++)
            {
                TerrainChunkMath.GetChunkVertexPosition(i, out int row, out int col, out bool isInner);

                int hx;
                int hy;
                if (!isInner)
                {
                    hx = col * 2;
                    hy = (row / 2) * 2;
                }
                else
                {
                    hx = col * 2 + 1;
                    hy = (row / 2) * 2 + 1;
                }

                int px = baseX + hx;
                int py = baseY + hy;
                if ((uint)px >= (uint)srcGridW || (uint)py >= (uint)srcGridH)
                    continue;

                int idx = py * srcGridW + px;
                sum[idx] += clip.Heights[i];
                if (count[idx] != ushort.MaxValue)
                    count[idx]++;
            }
        }

        var srcGrid = new float[srcGridW * srcGridH];
        for (int i = 0; i < srcGrid.Length; i++)
            srcGrid[i] = count[i] > 0 ? (sum[i] / count[i]) : float.NaN;

        var rotatedGrid = TerrainChunkMath.RotateFloatGrid(srcGrid, srcGridW, srcGridH, _chunkClipboardSelectionRotation, out int rotGridW, out int rotGridH);

        float heightDelta = 0f;
        if (_chunkClipboardPasteRelativeHeights)
        {
            var sourceClip = _chunkClipboardSet.Chunks.TryGetValue((0, 0), out var origin) ? origin : _chunkClipboardSet.Chunks.Values.First();
            float sourceRef = TerrainChunkMath.ComputeAverageHeight(sourceClip.Heights);
            if (TryGetChunkData(_chunkClipboardLockedTargetKey.Value.tileX, _chunkClipboardLockedTargetKey.Value.tileY,
                    _chunkClipboardLockedTargetKey.Value.chunkX, _chunkClipboardLockedTargetKey.Value.chunkY, out var targetChunkData))
            {
                float targetRef = TerrainChunkMath.ComputeAverageHeight(targetChunkData.Heights);
                heightDelta = targetRef - sourceRef;
            }
        }

        static (int dx, int dy) RotateInBox(int dx, int dy, int width, int height, int rot)
        {
            rot = ((rot % 4) + 4) % 4;
            return rot switch
            {
                0 => (dx, dy),
                1 => (height - 1 - dy, dx),
                2 => (width - 1 - dx, height - 1 - dy),
                3 => (dy, width - 1 - dx),
                _ => (dx, dy)
            };
        }

        var perTile = new Dictionary<(int tileX, int tileY), List<(int chunkX, int chunkY, int rdx, int rdy, ChunkClipboard clip)>>();
        foreach (var kvp in _chunkClipboardSet.Chunks)
        {
            var (rdx, rdy) = RotateInBox(kvp.Key.dx, kvp.Key.dy, width, height, _chunkClipboardSelectionRotation);
            int destGlobalX = targetGlobalX + rdx;
            int destGlobalY = targetGlobalY + rdy;
            if (destGlobalX < 0 || destGlobalX >= 64 * 16 || destGlobalY < 0 || destGlobalY >= 64 * 16)
                continue;

            int tileX = destGlobalX / 16;
            int tileY = destGlobalY / 16;
            int chunkX = destGlobalX % 16;
            int chunkY = destGlobalY % 16;

            var tkey = (tileX, tileY);
            if (!perTile.TryGetValue(tkey, out var list))
            {
                list = new List<(int, int, int, int, ChunkClipboard)>();
                perTile[tkey] = list;
            }

            list.Add((chunkX, chunkY, rdx, rdy, kvp.Value));
        }

        int pasted = 0;
        int skipped = 0;

        foreach (var entry in perTile)
        {
            var (tileX, tileY) = entry.Key;
            if (!TryGetTileChunksForEdit(tileX, tileY, out var chunks))
            {
                skipped += entry.Value.Count;
                continue;
            }

            var newChunks = chunks.ToList();

            foreach (var (chunkX, chunkY, rdx, rdy, clip) in entry.Value)
            {
                int idx = newChunks.FindIndex(c => c.ChunkX == chunkX && c.ChunkY == chunkY);
                if (idx < 0)
                {
                    skipped++;
                    continue;
                }

                var target = newChunks[idx];
                bool layersMatch = TerrainChunkMath.AreLayersCompatible(target.Layers, clip.Layers);

                int baseX = rdx * 16;
                int baseY = rdy * 16;
                var heights = new float[145];
                for (int i = 0; i < 145; i++)
                {
                    TerrainChunkMath.GetChunkVertexPosition(i, out int row, out int col, out bool isInner);

                    int hx;
                    int hy;
                    if (!isInner)
                    {
                        hx = col * 2;
                        hy = (row / 2) * 2;
                    }
                    else
                    {
                        hx = col * 2 + 1;
                        hy = (row / 2) * 2 + 1;
                    }

                    int px = baseX + hx;
                    int py = baseY + hy;
                    if ((uint)px >= (uint)rotGridW || (uint)py >= (uint)rotGridH)
                    {
                        heights[i] = target.Heights[i];
                        continue;
                    }

                    float v = rotatedGrid[py * rotGridW + px];
                    heights[i] = float.IsNaN(v) ? target.Heights[i] : (v + heightDelta);
                }

                int holeMask = TerrainChunkMath.RotateHoleMask(clip.HoleMask, _chunkClipboardSelectionRotation);
                var normals = TerrainChunkMath.GenerateNormalsForChunk(target, heights, holeMask);

                var layersToUse = target.Layers;
                var alphaToUse = target.AlphaMaps;
                byte[]? shadowToUse = target.ShadowMap;

                if (_chunkClipboardIncludeTextures)
                {
                    layersToUse = clip.Layers;
                    if (_chunkClipboardIncludeAlphaShadow)
                    {
                        alphaToUse = TerrainChunkMath.CloneAlphaMaps(clip.AlphaMaps);
                        shadowToUse = clip.ShadowMap != null ? (byte[])clip.ShadowMap.Clone() : null;
                    }
                }
                else if (_chunkClipboardIncludeAlphaShadow && layersMatch)
                {
                    alphaToUse = TerrainChunkMath.CloneAlphaMaps(clip.AlphaMaps);
                    shadowToUse = clip.ShadowMap != null ? (byte[])clip.ShadowMap.Clone() : null;
                }

                var pastedChunk = TerrainChunkMath.CloneTerrainChunk(
                    target,
                    heights: heights,
                    normals: normals,
                    holeMask: holeMask,
                    layers: layersToUse,
                    alphaMaps: alphaToUse,
                    shadowMap: shadowToUse,
                    mccvColors: target.MccvColors);

                newChunks[idx] = pastedChunk;
                pasted++;
            }

            ApplyEditedTileChunks(tileX, tileY, newChunks, entry.Value.Select(value => (value.chunkX, value.chunkY)));
        }

        _chunkClipboardStatus = $"Pasted {pasted} chunk(s)" + (skipped > 0 ? $" (skipped {skipped})" : "") + $". Rotation={_chunkClipboardSelectionRotation * 90}°";
    }

    private void ApplyEditedTileChunks(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> newChunks,
        IEnumerable<(int chunkX, int chunkY)> editedChunks)
    {
        if (_terrainManager != null)
            _terrainManager.ReplaceTileChunksAndRebuild(tileX, tileY, newChunks);
        else
            _vlmTerrainManager?.ReplaceTileChunksAndRebuild(tileX, tileY, newChunks);

        MarkChunkToolTileDirty(tileX, tileY, editedChunks);
    }

    private void MarkChunkToolTileDirty(int tileX, int tileY, IEnumerable<(int chunkX, int chunkY)> editedChunks)
    {
        if (!_chunkClipboardDirtyTileChunks.TryGetValue((tileX, tileY), out var chunkSet))
        {
            chunkSet = new HashSet<(int chunkX, int chunkY)>();
            _chunkClipboardDirtyTileChunks[(tileX, tileY)] = chunkSet;
        }

        foreach (var editedChunk in editedChunks)
            chunkSet.Add(editedChunk);
    }

    internal int GetChunkToolDirtyTileCount() => _chunkClipboardDirtyTileChunks.Count;

    internal int GetChunkToolDirtyChunkCount()
        => _chunkClipboardDirtyTileChunks.Values.Sum(chunks => chunks.Count);

    private string CreateChunkToolHeightmapOutputDirectory()
    {
        string root = Path.Combine(EnsureEditorProjectOutputDirectory(), "chunk-tool-heightmaps");
        Directory.CreateDirectory(root);

        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string candidate = Path.Combine(root, timestamp);
        int suffix = 1;
        while (Directory.Exists(candidate))
        {
            candidate = Path.Combine(root, $"{timestamp}_{suffix:D2}");
            suffix++;
        }

        Directory.CreateDirectory(candidate);
        return candidate;
    }

    private void SaveChunkToolHeightmapOutputs()
    {
        if (_chunkClipboardDirtyTileChunks.Count == 0)
        {
            _chunkClipboardStatus = "No edited chunk tiles are tracked yet.";
            return;
        }

        string outputDir = CreateChunkToolHeightmapOutputDirectory();
        var manifest = new ChunkToolHeightmapSaveManifest
        {
            ProjectName = GetEditorProjectName(),
            SourceKey = GetEditorProjectSourceKey() ?? string.Empty,
            GeneratedUtc = DateTime.UtcNow.ToString("O"),
        };

        int written = 0;
        int skipped = 0;

        foreach (var entry in _chunkClipboardDirtyTileChunks
            .OrderBy(item => item.Key.tileX)
            .ThenBy(item => item.Key.tileY))
        {
            var (tileX, tileY) = entry.Key;
            var chunks = _terrainTileIo.LoadTileChunksForExport(tileX, tileY);
            if (chunks == null || chunks.Count == 0)
            {
                skipped++;
                continue;
            }

            var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, tile.MinHeight, tile.MaxHeight);

            string pngName = $"tile_{tileX}_{tileY}_height_257.png";
            string jsonName = $"tile_{tileX}_{tileY}_height_257.json";
            string pngPath = Path.Combine(outputDir, pngName);
            string jsonPath = Path.Combine(outputDir, jsonName);

            using (var fs = File.Create(pngPath))
                img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

            var meta = new HeightmapMetadata
            {
                MinHeight = tile.MinHeight,
                MaxHeight = tile.MaxHeight,
                Normalization = "per_tile",
            };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));

            manifest.Tiles.Add(new ChunkToolHeightmapSaveTile
            {
                TileX = tileX,
                TileY = tileY,
                EditedChunks = entry.Value
                    .OrderBy(chunk => chunk.chunkX)
                    .ThenBy(chunk => chunk.chunkY)
                    .Select(chunk => $"{chunk.chunkX},{chunk.chunkY}")
                    .ToList(),
                HeightmapPng = pngName,
                HeightmapMetadataJson = jsonName,
                MinHeight = tile.MinHeight,
                MaxHeight = tile.MaxHeight,
            });
            written++;
        }

        if (written == 0)
        {
            _chunkClipboardStatus = "Chunk-tool save failed: no edited tiles could be exported.";
            return;
        }

        string manifestPath = Path.Combine(outputDir, "chunk_tool_heightmap_manifest.json");
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));

        _chunkClipboardLastSaveFolder = outputDir;
        _chunkClipboardStatus = $"Saved {written} edited tile heightmap output(s) to {outputDir}"
            + (skipped > 0 ? $" (skipped {skipped})" : string.Empty)
            + ". Source terrain files were left untouched.";
    }

    private void ClearChunkToolDirtyTracking()
    {
        _chunkClipboardDirtyTileChunks.Clear();
        _chunkClipboardLastSaveFolder = string.Empty;
        _chunkClipboardStatus = "Cleared chunk-tool dirty tracking.";
    }

    private void PasteChunkAtTarget(TerrainRenderer renderer)
    {
        if (_chunkClipboard == null)
        {
            _chunkClipboardStatus = "Paste failed: clipboard is empty.";
            return;
        }

        if (_chunkClipboardLockedTargetKey == null)
        {
            _chunkClipboardStatus = "Paste blocked: lock a paste target with Ctrl+LMB.";
            return;
        }

        var key = (TileX: _chunkClipboardLockedTargetKey.Value.tileX,
            TileY: _chunkClipboardLockedTargetKey.Value.tileY,
            ChunkX: _chunkClipboardLockedTargetKey.Value.chunkX,
            ChunkY: _chunkClipboardLockedTargetKey.Value.chunkY);

        if (!TryGetTileChunksForEdit(key.TileX, key.TileY, out var chunks))
        {
            _chunkClipboardStatus = $"Paste failed: tile data not available for tile({key.TileX},{key.TileY}).";
            return;
        }

        int idx = chunks.FindIndex(c => c.ChunkX == key.ChunkX && c.ChunkY == key.ChunkY);
        if (idx < 0)
        {
            _chunkClipboardStatus = $"Paste failed: chunk not found in tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY}).";
            return;
        }

        var target = chunks[idx];
        bool layersMatch = TerrainChunkMath.AreLayersCompatible(target.Layers, _chunkClipboard.Layers);

        float[] heights = (float[])_chunkClipboard.Heights.Clone();
        Vector3[] normals = (Vector3[])_chunkClipboard.Normals.Clone();
        if (_chunkClipboardPasteRelativeHeights)
        {
            float sourceRef = TerrainChunkMath.ComputeAverageHeight(_chunkClipboard.Heights);
            float targetRef = TerrainChunkMath.ComputeAverageHeight(target.Heights);
            float delta = targetRef - sourceRef;
            for (int i = 0; i < heights.Length; i++)
                heights[i] += delta;
        }

        var layersToUse = target.Layers;
        var alphaToUse = target.AlphaMaps;
        byte[]? shadowToUse = target.ShadowMap;

        if (_chunkClipboardIncludeTextures)
        {
            layersToUse = _chunkClipboard.Layers;
            if (_chunkClipboardIncludeAlphaShadow)
            {
                alphaToUse = TerrainChunkMath.CloneAlphaMaps(_chunkClipboard.AlphaMaps);
                shadowToUse = _chunkClipboard.ShadowMap != null ? (byte[])_chunkClipboard.ShadowMap.Clone() : null;
            }
        }
        else if (_chunkClipboardIncludeAlphaShadow && layersMatch)
        {
            alphaToUse = TerrainChunkMath.CloneAlphaMaps(_chunkClipboard.AlphaMaps);
            shadowToUse = _chunkClipboard.ShadowMap != null ? (byte[])_chunkClipboard.ShadowMap.Clone() : null;
        }

        var pasted = TerrainChunkMath.CloneTerrainChunk(
            target,
            heights: heights,
            normals: normals,
            holeMask: _chunkClipboard.HoleMask,
            layers: layersToUse,
            alphaMaps: alphaToUse,
            shadowMap: shadowToUse,
            mccvColors: (target.MccvColors != null && _chunkClipboard.MccvColors != null)
                ? (byte[])_chunkClipboard.MccvColors.Clone()
                : target.MccvColors);

        var newChunks = chunks.ToList();
        newChunks[idx] = pasted;

        ApplyEditedTileChunks(key.TileX, key.TileY, newChunks, new[] { (key.ChunkX, key.ChunkY) });

        bool didTextures = _chunkClipboardIncludeTextures;
        bool didAlpha = _chunkClipboardIncludeAlphaShadow && (didTextures || layersMatch);

        _chunkClipboardStatus = $"Pasted heights" +
                       (didTextures ? " + textures" : "") +
                       (didAlpha ? " + alpha/shadow" : "") +
                       $" into tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY})" +
                       (!didTextures && _chunkClipboardIncludeAlphaShadow && !layersMatch ? " (alpha skipped: layer mismatch)" : "");
    }

    private bool TryGetChunkData(int tileX, int tileY, int chunkX, int chunkY, out Terrain.TerrainChunkData chunk)
    {
        chunk = new Terrain.TerrainChunkData();

        if (!TryGetTileChunksForEdit(tileX, tileY, out var chunks))
            return false;

        var found = chunks.FirstOrDefault(c => c != null && c.ChunkX == chunkX && c.ChunkY == chunkY);
        if (found == null || found.Heights == null || found.Heights.Length == 0)
            return false;

        chunk = found;
        return true;
    }

    private bool TryGetTileChunksForEdit(int tileX, int tileY, out List<Terrain.TerrainChunkData> chunks)
    {
        chunks = new List<Terrain.TerrainChunkData>();

        if (_terrainManager != null)
        {
            var tile = _terrainManager.GetOrLoadTileLoadResult(tileX, tileY);
            chunks = tile.Chunks;
            return chunks.Count > 0;
        }

        if (_vlmTerrainManager != null)
        {
            if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
            {
                chunks = tile.Chunks;
                return chunks.Count > 0;
            }

            tile = _vlmTerrainManager.Loader.LoadTile(tileX, tileY);
            chunks = tile.Chunks;
            return chunks.Count > 0;
        }

        return false;
    }

    internal void DrawChunkClipboardContent(TerrainRenderer renderer)
    {
        ImGui.Checkbox("Enable Chunk Tool", ref _chunkToolEnabled);
        ImGui.SameLine();
        ImGui.Checkbox("Show Overlay", ref _chunkClipboardShowOverlay);

        ImGui.TextDisabled("Shift+LMB: toggle selection | Ctrl+LMB: lock paste target | Ctrl+C/Ctrl+V: copy/paste");

        ImGui.Checkbox("Copy Target: Use Mouse", ref _chunkClipboardUseMouse);
        ImGui.Checkbox("Paste Relative Heights", ref _chunkClipboardPasteRelativeHeights);
        ImGui.Checkbox("Include Alpha/Shadow", ref _chunkClipboardIncludeAlphaShadow);
        ImGui.Checkbox("Include Textures", ref _chunkClipboardIncludeTextures);

        ImGui.SetNextItemWidth(160f);
        string[] rotLabels = { "0°", "90°", "180°", "270°" };
        ImGui.Combo("Paste Rotation", ref _chunkClipboardSelectionRotation, rotLabels, rotLabels.Length);

        ImGui.SameLine();
        if (ImGui.SmallButton("Clear Locked Target##chunkTargetClear"))
        {
            _chunkClipboardLockedTargetKey = null;
            _chunkClipboardStatus = "Cleared locked paste target.";
        }

        ImGui.TextDisabled($"Selected: {_selectedChunks.Count}");
        if (_selectedChunks.Count > 0)
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("Clear##chunkSelClear"))
                _selectedChunks.Clear();
        }

        if (_chunkClipboardLockedTargetKey is { } locked)
            ImGui.Text($"Locked Paste Target: tile({locked.tileX},{locked.tileY}) chunk({locked.chunkX},{locked.chunkY})");
        else
            ImGui.TextDisabled("Locked Paste Target: (none)  (Ctrl+LMB to set)");

        var targetChunk = GetChunkClipboardTarget(renderer);
        bool hasChunk = targetChunk.HasValue;
        string targetLabel = _chunkClipboardUseMouse ? "Mouse" : "Camera";
        if (targetChunk is { } c)
        {
            ImGui.TextDisabled($"Copy Target ({targetLabel}): tile({c.TileX},{c.TileY}) chunk({c.ChunkX},{c.ChunkY})");
        }
        else
        {
            ImGui.TextDisabled($"Copy Target ({targetLabel}): (none loaded)");
        }

        if (!hasChunk) ImGui.BeginDisabled();
        if (ImGui.Button(_selectedChunks.Count > 0 ? "Copy Selection" : "Copy Chunk"))
        {
            if (_selectedChunks.Count > 0)
                CopySelectedChunks(renderer);
            else
                CopyChunkAtTarget(renderer);
        }
        if (!hasChunk) ImGui.EndDisabled();

        ImGui.SameLine();
        bool canPaste = (_chunkClipboardSet != null || _chunkClipboard != null);
        if (!canPaste) ImGui.BeginDisabled();
        if (ImGui.Button(_chunkClipboardSet != null ? "Paste Selection" : "Paste Chunk"))
        {
            if (_chunkClipboardSet != null)
                PasteClipboardSetAtTarget(renderer);
            else
                PasteChunkAtTarget(renderer);
        }
        if (!canPaste) ImGui.EndDisabled();

        ImGui.SameLine();
        bool canInvert = _selectedChunks.Count > 0 || hasChunk;
        if (!canInvert) ImGui.BeginDisabled();
        if (ImGui.Button(_selectedChunks.Count > 0 ? "Invert Z Selection" : "Invert Z Chunk"))
            InvertSelectedChunkHeights(renderer);
        if (!canInvert) ImGui.EndDisabled();

        ImGui.TextDisabled($"Edited tiles: {GetChunkToolDirtyTileCount()}  Edited chunks: {GetChunkToolDirtyChunkCount()}");
        ImGui.TextDisabled("Saves reusable 257x257 L16 heightmaps plus a manifest under the editor project output folder. Source terrain files stay untouched.");

        bool canSaveEdited = GetChunkToolDirtyTileCount() > 0;
        if (!canSaveEdited) ImGui.BeginDisabled();
        if (ImGui.Button("Save Edited Heightmaps"))
            SaveChunkToolHeightmapOutputs();
        if (!canSaveEdited) ImGui.EndDisabled();

        ImGui.SameLine();
        if (!canSaveEdited) ImGui.BeginDisabled();
        if (ImGui.SmallButton("Clear Dirty##chunkToolDirtyClear"))
            ClearChunkToolDirtyTracking();
        if (!canSaveEdited) ImGui.EndDisabled();

        if (!string.IsNullOrWhiteSpace(_chunkClipboardLastSaveFolder))
            ImGui.TextWrapped($"Last heightmap output: {_chunkClipboardLastSaveFolder}");

        if (!string.IsNullOrWhiteSpace(_chunkClipboardStatus))
            ImGui.TextWrapped(_chunkClipboardStatus);
    }
}
