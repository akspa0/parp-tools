using System;
using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.Rendering;
using WowViewer.Core.Maps;

namespace WoWViewer;

public partial class ViewerApp
{
    ref bool IViewerAppHost.ShowSettingsWindow => ref _settingsWindow._showSettingsWindow;
    void IViewerAppHost.DrawAuthoritativeFogControls(bool showDescription) => _settingsWindow.DrawAuthoritativeFogControls(showDescription);
}
