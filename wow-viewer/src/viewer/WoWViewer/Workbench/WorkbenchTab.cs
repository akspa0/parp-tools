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
    Scene = 2,
    Utilities = 3,
    Experimental = 4,
}
