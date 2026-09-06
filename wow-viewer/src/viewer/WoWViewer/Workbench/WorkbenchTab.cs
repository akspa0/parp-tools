namespace WoWViewer.Workbench;

/// <summary>
/// User-facing top-level destinations in the right-sidebar workbench.
/// These are task-oriented destinations rather than implementation/history
/// buckets. Older Model/World/Tools callers are adapted by ViewerApp.
/// </summary>
public enum WorkbenchTab
{
    Quick = 0,
    Inspect = 1,
    // 2-4 are retained for settings/caller compatibility. They are no
    // longer rendered as top-level tabs; ViewerApp adapts them to one of the
    // four canonical destinations before drawing the workbench.
    Scene = 2,
    Utilities = 3,
    Experimental = 4,
    Editor = 5,
    Archaeology = 6,
}
