namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// Thrown when registration is invalid — the only case today is a duplicate plugin identity
/// (Spec 166 FR-002), which must fail at startup rather than at first use.
/// </summary>
public sealed class EditorPluginRegistrationException : Exception
{
    public EditorPluginRegistrationException(string message)
        : base(message)
    {
    }

    public EditorPluginRegistrationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}