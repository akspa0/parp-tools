using System.Numerics;
using ImGuiNET;
using WowViewer.Core.Maps;

namespace WoWViewer;

public partial class ViewerApp
{
    ref bool IViewerAppHost.WantSelectDatasetCatalogRoot => ref _datasetCatalog._wantSelectDatasetCatalogRoot;
}
