using System.Numerics;
using ImGuiNET;
using WoWViewer.Logging;

namespace WoWViewer;

/// <summary>
/// Partial class containing the in-app log viewer UI.
/// </summary>
public partial class ViewerApp
{
    void IViewerAppHost.DrawLogViewerContent() => _logViewer.DrawLogViewerContent();
}
