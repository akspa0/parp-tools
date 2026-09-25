using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// ArchaeologyPanelService: members moved from ViewerApp_Investigation.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ArchaeologyPanelService
{

    private string _secondaryOverlayMapInput = "";
    private string _secondaryOverlaySearchFilter = "";
}
