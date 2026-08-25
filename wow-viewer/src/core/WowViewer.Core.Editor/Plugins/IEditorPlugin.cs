using WowViewer.Core.Editor.Eras;

namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// The plugin contract every editor capability implements. The host drives the lifecycle and owns
/// fault containment, undo/dirty state, and availability recomputation; a plugin only declares what
/// it is and draws/updates its own surface.
/// </summary>
/// <remarks>
/// Lifecycle (Spec 166 FR-003): register → availability query (repeatable) → activate (one
/// <c>OnActivated</c> before first draw) → update/draw → deactivate → dispose. A faulted plugin is
/// not re-invoked until <see cref="OnReset"/> succeeds.
/// </remarks>
public interface IEditorPlugin
{
    EditorPluginDescriptor Descriptor { get; }

    /// <summary>Reports availability for a build. Must be side-effect free and idempotent.</summary>
    EditorPluginAvailability GetAvailability(EditorBuildVersion build);

    void OnActivated(EditorPluginContext context);
    void OnDeactivated(EditorPluginContext context);
    void OnUpdate(EditorPluginContext context);
    void OnDraw(EditorPluginContext context);

    /// <summary>Explicit reset after a fault or a user request to clear retained state.</summary>
    void OnReset(EditorPluginContext context);

    void OnDisposed(EditorPluginContext context);
}