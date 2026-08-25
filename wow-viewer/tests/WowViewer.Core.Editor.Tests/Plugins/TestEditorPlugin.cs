using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Plugins;

namespace WowViewer.Core.Editor.Tests.Plugins;

/// <summary>Configurable fake plugin for lifecycle, availability, and fault testing.</summary>
public sealed class TestEditorPlugin : IEditorPlugin
{
    private readonly EditorBuildEraRange _eras;
    private readonly string? _unavailableReason;
    private readonly Func<bool>? _availabilityOverride;
    private readonly List<string> _lifecycleCalls = [];
    private bool _faultOnDraw;
    private bool _faultOnActivate;
    private bool _faultOnDeactivate;

    public TestEditorPlugin(
        string id,
        EditorBuildEraRange? eras = null,
        string? unavailableReason = null,
        Func<bool>? availabilityOverride = null)
    {
        _eras = eras ?? EditorBuildEraRange.All;
        _unavailableReason = unavailableReason;
        _availabilityOverride = availabilityOverride;
        Descriptor = new EditorPluginDescriptor(id, $"Display {id}", $"Desc {id}", _eras);
    }

    public EditorPluginDescriptor Descriptor { get; }

    public IReadOnlyList<string> LifecycleCalls => _lifecycleCalls;

    public bool FaultOnDraw
    {
        get => _faultOnDraw;
        set => _faultOnDraw = value;
    }

    public bool FaultOnActivate
    {
        get => _faultOnActivate;
        set => _faultOnActivate = value;
    }

    public bool FaultOnDeactivate
    {
        get => _faultOnDeactivate;
        set => _faultOnDeactivate = value;
    }

    public EditorPluginAvailability GetAvailability(EditorBuildVersion build)
    {
        _lifecycleCalls.Add("GetAvailability");
        if (_availabilityOverride is not null)
            return _availabilityOverride() ? EditorPluginAvailability.Available : EditorPluginAvailability.Unavailable("override");

        return _unavailableReason is null ? EditorPluginAvailability.Available : EditorPluginAvailability.Unavailable(_unavailableReason);
    }

    public void OnActivated(EditorPluginContext context)
    {
        _lifecycleCalls.Add("OnActivated");
        if (_faultOnActivate)
            throw new InvalidOperationException("deliberate activation fault");
    }

    public void OnDeactivated(EditorPluginContext context)
    {
        _lifecycleCalls.Add("OnDeactivated");
        if (_faultOnDeactivate)
            throw new InvalidOperationException("deliberate deactivation fault");
    }

    public void OnUpdate(EditorPluginContext context) => _lifecycleCalls.Add("OnUpdate");

    public void OnDraw(EditorPluginContext context)
    {
        _lifecycleCalls.Add("OnDraw");
        if (_faultOnDraw)
            throw new InvalidOperationException("deliberate draw fault");
    }

    public void OnReset(EditorPluginContext context)
    {
        _lifecycleCalls.Add("OnReset");
        _faultOnDraw = false;
    }

    public void OnDisposed(EditorPluginContext context) => _lifecycleCalls.Add("OnDisposed");
}