namespace WoWViewer;

/// <summary>
/// Content-only Options, stats, workspace and shared inspection controls.
/// The UI manager owns chrome windows, placement, navigation and popup lifetime.
/// Existing viewer state and services are accessed through <see cref="IViewerAppHost"/>.
/// </summary>
internal sealed partial class ViewerChromeService
{
    private readonly IViewerAppHost _host;

    internal ViewerChromeService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see ViewerChromeService.Host.cs.

    private const float SceneViewportPreferredMinWidth = 420f;
}
