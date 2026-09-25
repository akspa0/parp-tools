using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using Silk.NET.Input;

namespace WoWViewer;

public partial class ViewerApp
{
    void IViewerAppHost.OpenCapturePanelTab(CameraPathsService.CapturePanelTab tab) => _cameraPaths.OpenCapturePanelTab(tab);
    void IViewerAppHost.DrawCapturePanelContent() => _cameraPaths.DrawCapturePanelContent();
    void IViewerAppHost.StopCameraPathPlayback() => _cameraPaths.StopCameraPathPlayback();
}
