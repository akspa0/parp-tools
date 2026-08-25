namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// Owns the set of compiled-in plugins. Registration is append-only and fails fast on duplicate
/// identity so the mistake is caught at startup, not when a user opens the second plugin.
/// </summary>
public sealed class EditorPluginRegistry
{
    private readonly List<IEditorPlugin> _plugins = [];
    private readonly Dictionary<string, IEditorPlugin> _byId = new(StringComparer.Ordinal);

    public void Register(IEditorPlugin plugin)
    {
        ArgumentNullException.ThrowIfNull(plugin);
        ArgumentNullException.ThrowIfNull(plugin.Descriptor);

        string id = plugin.Descriptor.Id;
        if (!_byId.TryAdd(id, plugin))
        {
            IEditorPlugin existing = _byId[id];
            throw new EditorPluginRegistrationException(
                $"Duplicate editor plugin identity '{id}'. First registered by '{existing.Descriptor.DisplayName}', then re-registered by '{plugin.Descriptor.DisplayName}'.");
        }

        _plugins.Add(plugin);
    }

    public IReadOnlyList<IEditorPlugin> Plugins => _plugins;

    public IEditorPlugin? Find(string id)
        => _byId.GetValueOrDefault(id);
}