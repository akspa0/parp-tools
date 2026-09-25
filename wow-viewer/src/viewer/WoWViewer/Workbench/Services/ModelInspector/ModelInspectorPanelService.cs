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
using static WoWViewer.ViewerApp;
using static WoWViewer.NavigatorPanelService;

namespace WoWViewer;

/// <summary>
/// Model inspector: model info, animation controls and export, WMO/doodad-set controls, renderer visibility, framing and model action tabs.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class ModelInspectorPanelService
{
    private readonly IViewerAppHost _host;

    internal ModelInspectorPanelService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see ModelInspectorPanelService.Host.cs.

    private int _selectedStandaloneWmoDoodadIndex = -1;
    private int _selectedWorldWmoDoodadIndex = -1;
    private int _standaloneWmoDoodadGroupFilter = -1;
    private int _worldWmoDoodadGroupFilter = -1;
    private static readonly string[] WmoLiquidRotationLabels = { "0°", "90°", "180°", "270°" };

    private bool TryFrameStandaloneWmoDoodad(WmoRenderer wmoRenderer, WmoDoodadInfo doodad)
    {
        if (wmoRenderer.TryGetDoodadBounds(doodad.Index, Matrix4x4.Identity, out Vector3 boundsMin, out Vector3 boundsMax))
        {
            FrameBounds(boundsMin, boundsMax, mdxMirrorX: false);
            _statusMessage = $"Framed standalone WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
            return true;
        }

        FramePoint(doodad.LocalPosition, radius: 2f);
        _statusMessage = $"Framed standalone WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
        return true;
    }

    private bool TryFrameSelectedWorldWmoDoodad(WmoRenderer wmoRenderer, WmoDoodadInfo doodad)
    {
        if (_worldScene?.SelectedInstance is not ObjectInstance selectedInstance)
            return false;

        if (wmoRenderer.TryGetDoodadBounds(doodad.Index, selectedInstance.Transform, out Vector3 boundsMin, out Vector3 boundsMax))
        {
            FrameBounds(boundsMin, boundsMax, mdxMirrorX: false);
            _statusMessage = $"Framed world WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
            return true;
        }

        Vector3 worldPosition = Vector3.Transform(doodad.LocalPosition, selectedInstance.Transform);
        FramePoint(worldPosition, radius: 2f);
        _statusMessage = $"Framed world WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
        return true;
    }

    private bool TryGetStandaloneWmoAssetPath(out string assetPath)
    {
        assetPath = string.Empty;
        if (_renderer is not WmoRenderer || string.IsNullOrWhiteSpace(_lastVirtualPath))
            return false;

        assetPath = NormalizeAssetPathForUi(_lastVirtualPath);
        return !string.IsNullOrWhiteSpace(assetPath);
    }
}
