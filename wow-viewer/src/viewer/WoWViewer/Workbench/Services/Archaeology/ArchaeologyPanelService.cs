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
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Archaeology panels: UniqueId archaeology window, range/layers/playback/capture sub-tabs and archaeology playback.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class ArchaeologyPanelService
{
    private readonly IViewerAppHost _host;

    internal ArchaeologyPanelService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see ArchaeologyPanelService.Host.cs.


    // Experimental pages retain their existing internal selectors while the
    // top-level destination owns the only visible category switch.
    private int _activeArcheologyTabIndex = 0;
    private double _archeologyPlaybackAccumulator = 0.0; // for fractional uniqueId advancement
    private int _archeologyPlaybackRestoreMin = -1; // saved on Play, restored on Stop
    private int _archeologyPlaybackRestoreMax = -1;
    private bool _archeologyPlaybackRestoreFilter = false;

    internal void UpdateArcheologyPlayback(double dt)
    {
        if (!_archeologyPlaybackActive)
            return;

        if (_worldScene == null)
        {
            _archeologyPlaybackActive = false;
            _archeologyPlaybackAccumulator = 0;
            _statusMessage = "Archeology playback stopped because the world was unloaded.";
            return;
        }

        if (!_worldScene.TryGetUniqueIdFilterRange(out int minId, out int maxId, out _))
        {
            _archeologyPlaybackActive = false;
            _archeologyPlaybackAccumulator = 0;
            _statusMessage = "Archeology playback stopped because no scoped UniqueId range is available.";
            return;
        }

        _archeologyPlaybackAccumulator += dt * _archeologyPlaybackSpeed;
        int advance = (int)Math.Floor(_archeologyPlaybackAccumulator);
        if (advance <= 0) return;
        _archeologyPlaybackAccumulator -= advance;

        int currentMax = _worldScene.UniqueIdFilterMax;
        int newMax = currentMax + advance;
        if (newMax >= maxId)
        {
            if (_archeologyPlaybackLoop)
            {
                // Loop: snap back to min
                int restoreMin = _archeologyPlaybackRestoreMin >= 0 ? _archeologyPlaybackRestoreMin : minId;
                _worldScene.SetUniqueIdFilterRange(restoreMin, restoreMin);
                _archeologyPlaybackAccumulator = 0;
            }
            else
            {
                _worldScene.UniqueIdFilterMax = maxId;
                _archeologyPlaybackActive = false;
                _archeologyPlaybackAccumulator = 0;
                _statusMessage = "Archeology playback reached end of range.";
            }
        }
        else
        {
            _worldScene.UniqueIdFilterMax = newMax;
        }
    }
}
