using ImGuiNET;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// CameraPathsService: members moved from ViewerKeyBindings.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class CameraPathsService
{

    private bool IsCaptureKeyboardContextActive()
    {
        return GetActiveKeyContext() == ViewerKeyContext.Capture
            && (_showCameraPathWindow || _activeCapturePanelTabIndex == (int)CapturePanelTab.CameraPath);
    }
}
