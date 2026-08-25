using WowViewer.Core.Editor.Eras;

namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// The stable declaration of a plugin: a unique identity, a human display name, a description, and
/// the build eras it supports. The identity is the registration key; two plugins with the same
/// identity fail registration (Spec 166 FR-002).
/// </summary>
public sealed record EditorPluginDescriptor
{
    public EditorPluginDescriptor(
        string id,
        string displayName,
        string description,
        EditorBuildEraRange supportedEras)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentException.ThrowIfNullOrWhiteSpace(displayName);

        Id = id;
        DisplayName = displayName;
        Description = description ?? string.Empty;
        SupportedEras = supportedEras;
    }

    public string Id { get; init; }

    public string DisplayName { get; init; }

    public string Description { get; init; }

    public EditorBuildEraRange SupportedEras { get; init; }
}