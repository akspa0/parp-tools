namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// Immutable snapshot of a plugin for the Editor list: identity, display metadata, lifecycle state,
/// and availability with a stated reason when unavailable.
/// </summary>
public readonly record struct EditorPluginCatalogEntry(
    string Id,
    string DisplayName,
    string Description,
    EditorPluginState State,
    EditorPluginAvailability Availability,
    string? FaultMessage);