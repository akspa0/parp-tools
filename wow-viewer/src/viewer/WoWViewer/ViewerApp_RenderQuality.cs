using System;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using Silk.NET.OpenGL;

namespace WoWViewer;

public partial class ViewerApp
{
    void IViewerAppHost.DrawRenderQualityContent() => _renderQuality.DrawRenderQualityContent();
}
