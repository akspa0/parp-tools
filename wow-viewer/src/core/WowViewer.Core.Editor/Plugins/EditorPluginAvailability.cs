namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// A plugin's availability for the current build. Unavailable plugins must remain visible with a
/// stated reason (Spec 166 FR-001), never hidden, so the reason travels with the verdict.
/// </summary>
public readonly record struct EditorPluginAvailability(bool IsAvailable, string? UnavailableReason)
{
    public static EditorPluginAvailability Available => new(true, null);

    public static EditorPluginAvailability Unavailable(string reason)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reason);
        return new EditorPluginAvailability(false, reason);
    }

    public override string ToString()
        => IsAvailable ? "available" : $"unavailable: {UnavailableReason}";
}