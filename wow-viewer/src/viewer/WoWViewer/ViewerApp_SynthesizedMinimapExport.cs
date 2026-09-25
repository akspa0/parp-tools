using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using ImGuiNET;
using WoWViewer.DataSources;
using WowViewer.Core.IO.Casc;
using WowViewer.Core.Maps;

namespace WoWViewer;

public partial class ViewerApp
{
    void IViewerAppHost.PrepareSynthesizedMinimapExportDialogInputs() => _synthesizedMinimapExport.PrepareSynthesizedMinimapExportDialogInputs();
    void IViewerAppHost.DrawSynthesizedMinimapExportContent(bool showCloseButton) => _synthesizedMinimapExport.DrawSynthesizedMinimapExportContent(showCloseButton);
}
