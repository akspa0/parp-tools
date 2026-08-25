using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Logging;
using WowViewer.Core.Editor.Plugins;

namespace WowViewer.Core.Editor;

public enum EditorActivationResult
{
    Activated,
    AlreadyActive,
    Unavailable,
    Faulted,
    NotFound,
}

/// <summary>
/// The in-process editor plugin host: one registered pool of plugins, one active plugin at a time,
/// availability recomputation on build change, and fault containment. This is library-only — the
/// viewer shell passes in a log and drives the frame loop.
/// </summary>
public sealed class EditorHost : IDisposable
{
    private readonly EditorPluginRegistry _registry = new();
    private readonly IEditorLog _log;
    private readonly Dictionary<string, EditorPluginRuntimeState> _runtime = new(StringComparer.Ordinal);

    private EditorBuildVersion _build;
    private IEditorPlugin? _active;

    public EditorHost(EditorBuildVersion build, IEditorLog log)
    {
        _build = build;
        _log = log ?? throw new ArgumentNullException(nameof(log));
    }

    public IReadOnlyList<IEditorPlugin> Plugins => _registry.Plugins;

    public IEditorPlugin? ActivePlugin => _active;

    public EditorBuildVersion Build
    {
        get => _build;
        set
        {
            if (value == _build)
                return;

            _build = value;
            RecomputeAvailability();
        }
    }

    /// <summary>Registers a plugin and initializes its host-side runtime state.</summary>
    public void Register(IEditorPlugin plugin)
    {
        ArgumentNullException.ThrowIfNull(plugin);

        _registry.Register(plugin);
        _runtime[plugin.Descriptor.Id] = new EditorPluginRuntimeState
        {
            Availability = ComputeAvailability(plugin, _build),
        };
    }

    /// <summary>Immutable snapshot of every plugin for the Editor list.</summary>
    public IReadOnlyList<EditorPluginCatalogEntry> Catalog
    {
        get
        {
            var entries = new EditorPluginCatalogEntry[_registry.Plugins.Count];
            for (int index = 0; index < _registry.Plugins.Count; index++)
            {
                IEditorPlugin plugin = _registry.Plugins[index];
                EditorPluginRuntimeState state = _runtime[plugin.Descriptor.Id];
                entries[index] = new EditorPluginCatalogEntry(
                    plugin.Descriptor.Id,
                    plugin.Descriptor.DisplayName,
                    plugin.Descriptor.Description,
                    state.State,
                    state.Availability,
                    state.FaultMessage);
            }

            return entries;
        }
    }

    public EditorActivationResult Activate(string id)
    {
        IEditorPlugin? plugin = _registry.Find(id);
        if (plugin is null)
            return EditorActivationResult.NotFound;

        EditorPluginRuntimeState state = _runtime[id];
        if (state.State == EditorPluginState.Faulted)
            return EditorActivationResult.Faulted;

        EditorPluginAvailability availability = ComputeAvailability(plugin, _build);
        state.Availability = availability;
        if (!availability.IsAvailable)
            return EditorActivationResult.Unavailable;

        if (ReferenceEquals(_active, plugin))
            return EditorActivationResult.AlreadyActive;

        Deactivate();

        _active = plugin;
        state.State = EditorPluginState.Active;
        try
        {
            plugin.OnActivated(CreateContext());
            return EditorActivationResult.Activated;
        }
        catch (Exception ex)
        {
            MarkFaulted(plugin, "activation", ex);
            return EditorActivationResult.Faulted;
        }
    }

    public void Deactivate()
    {
        if (_active is null)
            return;

        IEditorPlugin plugin = _active;
        _active = null;

        try
        {
            plugin.OnDeactivated(CreateContext());
        }
        catch (Exception ex)
        {
            MarkFaulted(plugin, "deactivation", ex);
            return;
        }

        if (_runtime.TryGetValue(plugin.Descriptor.Id, out EditorPluginRuntimeState? state)
            && state.State == EditorPluginState.Active)
        {
            state.State = EditorPluginState.Inactive;
        }
    }

    public void Update()
    {
        if (_active is null)
            return;

        InvokeSafely(_active, static (p, ctx) => p.OnUpdate(ctx), "update");
    }

    public void Draw()
    {
        if (_active is null)
            return;

        InvokeSafely(_active, static (p, ctx) => p.OnDraw(ctx), "draw");
    }

    public bool Reset(string id)
    {
        IEditorPlugin? plugin = _registry.Find(id);
        if (plugin is null)
            return false;

        EditorPluginRuntimeState state = _runtime[id];
        if (state.State != EditorPluginState.Faulted)
            return true;

        try
        {
            plugin.OnReset(CreateContext());
        }
        catch (Exception ex)
        {
            MarkFaulted(plugin, "reset", ex);
            return false;
        }

        state.State = EditorPluginState.Inactive;
        state.Fault = null;
        return true;
    }

    public void Dispose()
    {
        Deactivate();

        foreach (IEditorPlugin plugin in _registry.Plugins)
        {
            try
            {
                plugin.OnDisposed(CreateContext());
            }
            catch (Exception ex)
            {
                _log.Error($"Editor plugin '{plugin.Descriptor.Id}' faulted during dispose: {ex.Message}", ex);
            }
        }
    }

    public EditorPluginAvailability GetAvailability(IEditorPlugin plugin)
        => ComputeAvailability(plugin, _build);

    private void RecomputeAvailability()
    {
        foreach (IEditorPlugin plugin in _registry.Plugins)
        {
            if (_runtime.TryGetValue(plugin.Descriptor.Id, out EditorPluginRuntimeState? state))
                state.Availability = ComputeAvailability(plugin, _build);
        }
    }

    private static EditorPluginAvailability ComputeAvailability(IEditorPlugin plugin, EditorBuildVersion build)
    {
        if (!plugin.Descriptor.SupportedEras.Contains(build))
        {
            return EditorPluginAvailability.Unavailable(
                $"Build {build.Original} is outside supported eras {plugin.Descriptor.SupportedEras.ToString()}.");
        }

        try
        {
            return plugin.GetAvailability(build);
        }
        catch (Exception ex)
        {
            return EditorPluginAvailability.Unavailable($"Availability check failed: {ex.Message}");
        }
    }

    private EditorPluginContext CreateContext()
        => new(_build, this, new IdentityScopedLog(_log, _active?.Descriptor.Id));

    private void InvokeSafely(IEditorPlugin plugin, Action<IEditorPlugin, EditorPluginContext> action, string phase)
    {
        EditorPluginRuntimeState state = _runtime[plugin.Descriptor.Id];
        if (state.State == EditorPluginState.Faulted)
            return;

        try
        {
            action(plugin, CreateContext());
        }
        catch (Exception ex)
        {
            MarkFaulted(plugin, phase, ex);
        }
    }

    private void MarkFaulted(IEditorPlugin plugin, string phase, Exception exception)
    {
        if (_runtime.TryGetValue(plugin.Descriptor.Id, out EditorPluginRuntimeState? state))
        {
            state.State = EditorPluginState.Faulted;
            state.Fault = exception;
        }

        if (ReferenceEquals(_active, plugin))
            _active = null;

        _log.Error($"Editor plugin '{plugin.Descriptor.Id}' faulted during {phase}: {exception.Message}", exception);
    }

    /// <summary>Prefixes every log line with the owning plugin's identity.</summary>
    private sealed class IdentityScopedLog : IEditorLog
    {
        private readonly IEditorLog _inner;
        private readonly string? _pluginId;

        public IdentityScopedLog(IEditorLog inner, string? pluginId)
        {
            _inner = inner;
            _pluginId = pluginId;
        }

        private string Scope(string message)
            => _pluginId is null ? message : $"[plugin:{_pluginId}] {message}";

        public void Trace(string message) => _inner.Trace(Scope(message));
        public void Info(string message) => _inner.Info(Scope(message));
        public void Warn(string message) => _inner.Warn(Scope(message));
        public void Error(string message, Exception? exception = null) => _inner.Error(Scope(message), exception);
    }
}