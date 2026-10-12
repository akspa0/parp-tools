using System.Numerics;
using System.Diagnostics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Workbench;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;
using WoWViewer.Population;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// MainMenuBarService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class MainMenuBarService
{

    /// <summary>
    /// Legacy-sidebar adapter for View/Tools utility menu entries. It reveals
    /// the same utility dispatcher inside the existing right sidebar instead
    /// of creating a new floating window.
    /// </summary>
    private void OpenLegacyWorkbenchUtility(UtilitiesBottomTab tab)
    {
        _activeUtilitiesTabIndex = Math.Clamp((int)tab, 0, (int)UtilitiesBottomTab.Audio);
        _legacyUtilityPage = tab;
        _showRightSidebar = true;
        _workbenchOpen = true;
    }
}
