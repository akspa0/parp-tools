namespace WowViewer.Core.Editor.Plugins;

public enum EditorPluginState
{
    /// <summary>Registered but not active, and not faulted.</summary>
    Inactive,

    /// <summary>The currently active plugin.</summary>
    Active,

    /// <summary>Faulted; not re-invoked until an explicit reset succeeds.</summary>
    Faulted,
}

/// <summary>Host-owned per-plugin runtime state (availability cache, fault, lifecycle state).</summary>
public sealed class EditorPluginRuntimeState
{
    public EditorPluginState State { get; set; } = EditorPluginState.Inactive;

    public Exception? Fault { get; set; }

    public EditorPluginAvailability Availability { get; set; }

    public string? FaultMessage => Fault?.Message;
}