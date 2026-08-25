using WowViewer.Core.Editor.Eras;

namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// The single reference plugin that ships with the host. It proves the whole path — registration,
/// activation, draw, deactivation, disposal, and availability — without needing any game data, which
/// is exactly the boundary Spec 166's independent test draws: one plugin class per capability.
/// </summary>
public sealed class ReferenceEditorPlugin : IEditorPlugin
{
    public const string PluginId = "editor.platform.reference";

    private int _activationCount;
    private bool _faultOnDraw;

    public ReferenceEditorPlugin(bool faultOnDraw = false)
    {
        _faultOnDraw = faultOnDraw;
    }

    public EditorPluginDescriptor Descriptor { get; } = new(
        PluginId,
        "Reference plugin",
        "Proves editor registration and lifecycle with a single line of text.",
        EditorBuildEraRange.All);

    public int ActivationCount => _activationCount;

    public EditorPluginAvailability GetAvailability(EditorBuildVersion build)
        => EditorPluginAvailability.Available;

    public void OnActivated(EditorPluginContext context)
    {
        _activationCount++;
        context.Log.Info("Reference plugin activated.");
    }

    public void OnDeactivated(EditorPluginContext context)
        => context.Log.Info("Reference plugin deactivated.");

    public void OnUpdate(EditorPluginContext context)
    {
    }

    public void OnDraw(EditorPluginContext context)
    {
        if (_faultOnDraw)
            throw new InvalidOperationException("Deliberate draw fault.");

        context.Log.Trace("Reference plugin drawing one line of text.");
    }

    public void OnReset(EditorPluginContext context)
    {
        _faultOnDraw = false;
        context.Log.Info("Reference plugin reset.");
    }

    public void OnDisposed(EditorPluginContext context)
        => context.Log.Info("Reference plugin disposed.");
}