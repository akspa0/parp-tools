using System.Numerics;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.UI;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Terrain;
using WowViewer.Core.Maps;

namespace WoWViewer.Workbench.Services;

/// <summary>
/// Spec 234 US3: New Map creator in the Editor tab.
/// Synthesizes and writes real client-loadable maps (templated procedural or blank flat)
/// and loads them directly into the viewer for immediate editing and save.
/// </summary>
public sealed class NewMapCreatorService
{
    private string _mapName = "CustomMap";
    private int _modeIdx = 0; // 0 = Templated Procedural, 1 = Blank Flat
    private int _themeIdx = 0;
    private int _baseTileX = 32;
    private int _baseTileY = 32;
    private int _tileRows = 2;
    private int _tileCols = 2;
    private int _plazaSpacing = 2;
    private float _flatHeight = 0f;
    private string _outputDir = Path.Combine(AppContext.BaseDirectory, "project_output", "maps");
    private string _statusMessage = string.Empty;
    private Vector4 _statusColor = new(0.2f, 0.9f, 0.2f, 1f);

    public void Draw(Action<string, string, int, int, int, int>? onLoadMapRequested)
    {
        ImGui.Text("New Map Creator (Spec 234 US3)");
        ImGui.TextDisabled("Synthesize a new client-loadable map (LK v18 WDT/ADT) and load it directly into the viewer for editing.");
        ImGui.Spacing();

        ImGui.InputText("Map Name", ref _mapName, 64);

        string[] modes = ["Templated Procedural Map", "Blank Flat Map"];
        ImGui.Combo("Creation Mode", ref _modeIdx, modes, modes.Length);

        if (_modeIdx == 0)
        {
            string[] themes = ["Garden Museum", "Elwynn Forest", "Cobblestone City", "Dun Morogh", "Barrens", "Ashenvale"];
            ImGui.Combo("Biome Theme", ref _themeIdx, themes, themes.Length);
            ImGui.SliderInt("Plaza Spacing (Chunks)", ref _plazaSpacing, 1, 4);
        }
        else
        {
            ImGui.SliderFloat("Base Ground Height", ref _flatHeight, -100f, 500f, "%.1f yd");
        }

        ImGui.SliderInt("Base Tile X (Col)", ref _baseTileX, 0, 63);
        ImGui.SliderInt("Base Tile Y (Row)", ref _baseTileY, 0, 63);
        ImGui.SliderInt("Tile Rows", ref _tileRows, 1, 8);
        ImGui.SliderInt("Tile Cols", ref _tileCols, 1, 8);

        ImGui.InputText("Output Directory", ref _outputDir, 260);

        int totalTiles = _tileRows * _tileCols;
        ImGui.TextDisabled($"Will create {totalTiles} tiles ({totalTiles * 256} chunks) at grid ({_baseTileX},{_baseTileY}) .. ({_baseTileX + _tileCols - 1},{_baseTileY + _tileRows - 1}).");
        ImGui.Spacing();

        if (ImGui.Button("Create & Load Map", new Vector2(200, 30)))
        {
            CreateAndLoadMap(onLoadMapRequested);
        }

        if (!string.IsNullOrEmpty(_statusMessage))
        {
            ImGui.Spacing();
            ImGui.TextColored(_statusColor, _statusMessage);
        }
    }

    private void CreateAndLoadMap(Action<string, string, int, int, int, int>? onLoadMapRequested)
    {
        string trimmedName = _mapName.Trim();
        if (string.IsNullOrWhiteSpace(trimmedName))
        {
            _statusMessage = "Error: Map name cannot be empty.";
            _statusColor = new Vector4(1f, 0.3f, 0.3f, 1f);
            return;
        }

        char[] invalidChars = Path.GetInvalidFileNameChars();
        if (trimmedName.IndexOfAny(invalidChars) >= 0)
        {
            _statusMessage = "Error: Map name contains invalid characters.";
            _statusColor = new Vector4(1f, 0.3f, 0.3f, 1f);
            return;
        }

        try
        {
            string mapDir = Path.Combine(_outputDir, "World", "Maps", trimmedName);
            if (Directory.Exists(mapDir) && Directory.EnumerateFiles(mapDir, "*.adt").Any())
            {
                _statusMessage = $"Error: Map '{trimmedName}' already exists at {mapDir}. Choose another name or output directory.";
                _statusColor = new Vector4(1f, 0.3f, 0.3f, 1f);
                return;
            }

            Directory.CreateDirectory(mapDir);

            var template = new TerrainMapTemplate
            {
                MapName = trimmedName,
                Theme = (BiomeTheme)_themeIdx,
                BaseTileX = _baseTileX,
                BaseTileY = _baseTileY,
                TileRows = _tileRows,
                TileCols = _tileCols,
                PlazaSpacingChunks = _plazaSpacing,
                Palette = BiomePalette.ForTheme((BiomeTheme)_themeIdx)
            };

            TemplatedMapResult result = TemplatedTerrainGenerator.GenerateMap(template);

            // If blank flat mode is selected, override vertex heights with flat ground height
            if (_modeIdx == 1)
            {
                foreach (var key in result.Tiles.Keys.ToList())
                {
                    var adt = result.Tiles[key];
                    var flatChunks = new List<LkMcnkData>(adt.Chunks.Count);
                    foreach (var chunk in adt.Chunks)
                    {
                        float[] heights = new float[145];
                        Array.Fill(heights, _flatHeight);
                        flatChunks.Add(new LkMcnkData
                        {
                            IndexX = chunk.IndexX,
                            IndexY = chunk.IndexY,
                            Flags = chunk.Flags,
                            AreaId = chunk.AreaId,
                            NLayers = chunk.NLayers,
                            HoleMask = chunk.HoleMask,
                            BaseHeight = _flatHeight,
                            Heights = heights,
                            Normals = chunk.Normals,
                            ShadowMap = chunk.ShadowMap,
                            AlphaMapData = chunk.AlphaMapData,
                            AlphaMapSize = chunk.AlphaMapSize,
                            Layers = chunk.Layers,
                            DoodadRefs = chunk.DoodadRefs,
                            WorldModelRefs = chunk.WorldModelRefs,
                            LiquidData = chunk.LiquidData,
                            MccvColors = chunk.MccvColors,
                            MclvLighting = chunk.MclvLighting,
                            PosX = chunk.PosX,
                            PosY = chunk.PosY,
                            PosZ = _flatHeight,
                        });
                    }

                    result.Tiles[key] = new LkAdtData
                    {
                        MapName = adt.MapName,
                        TileX = adt.TileX,
                        TileY = adt.TileY,
                        TextureNames = adt.TextureNames,
                        ModelNames = adt.ModelNames,
                        WorldModelNames = adt.WorldModelNames,
                        ModelPlacements = [],
                        WorldModelPlacements = [],
                        Chunks = flatChunks,
                        MhdrFlags = adt.MhdrFlags,
                        MfboFlightBounds = adt.MfboFlightBounds,
                    };
                }
            }

            foreach (var (coord, adtData) in result.Tiles)
            {
                string adtPath = Path.Combine(mapDir, $"{trimmedName}_{coord.Y}_{coord.X}.adt");
                LkAdtWriter.Write(adtPath, adtData);
            }

            var existingTiles = new HashSet<(int tileX, int tileY)>(result.Tiles.Keys.Select(k => (k.X, k.Y)));
            string wdtPath = Path.Combine(mapDir, $"{trimmedName}.wdt");
            LkWdtWriter.Write(wdtPath, existingTiles);

            _statusMessage = $"Successfully created map '{trimmedName}' with {result.Tiles.Count} tiles at '{mapDir}'.";
            _statusColor = new Vector4(0.2f, 0.9f, 0.2f, 1f);
            ViewerLog.Important(ViewerLog.Category.General, $"[NewMapCreator] {_statusMessage}");

            onLoadMapRequested?.Invoke(_outputDir, trimmedName, _baseTileX, _baseTileY, _tileRows, _tileCols);
        }
        catch (Exception ex)
        {
            _statusMessage = $"Creation failed: {ex.Message}";
            _statusColor = new Vector4(1f, 0.3f, 0.3f, 1f);
            ViewerLog.Error(ViewerLog.Category.General, $"[NewMapCreator] Error: {ex}");
        }
    }
}
