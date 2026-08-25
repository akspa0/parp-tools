using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Logging;

namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// What a plugin is handed for a frame or a lifecycle transition: the current build identity, the
/// host it belongs to (for dirty reporting and re-activation), and a log sink that the host has
/// already scoped with the plugin's identity.
/// </summary>
public sealed class EditorPluginContext
{
    public EditorPluginContext(EditorBuildVersion build, EditorHost host, IEditorLog log)
    {
        ArgumentNullException.ThrowIfNull(host);
        ArgumentNullException.ThrowIfNull(log);

        Build = build;
        Host = host;
        Log = log;
    }

    public EditorBuildVersion Build { get; }

    public EditorHost Host { get; }

    public IEditorLog Log { get; }
}