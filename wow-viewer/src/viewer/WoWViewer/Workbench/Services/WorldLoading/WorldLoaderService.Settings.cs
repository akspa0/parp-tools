using System;
using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.Rendering;
using WowViewer.Core.Maps;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// WorldLoaderService: members moved from ViewerApp_Settings.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class WorldLoaderService
{

    /// <summary>
    /// Apply saved fog defaults to terrain lighting after terrain loads.
    /// Call this after terrain manager creation.
    /// </summary>
    private void ApplyGlobalFogDefaults(TerrainLighting lighting)
    {
        (lighting.FogStart, lighting.FogEnd) = TerrainLightingMath.NormalizeFogRange(_defaultFogStart, _defaultFogEnd);
    }
}
