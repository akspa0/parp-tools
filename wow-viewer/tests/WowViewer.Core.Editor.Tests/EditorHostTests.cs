using WowViewer.Core.Editor;
using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Logging;
using WowViewer.Core.Editor.Plugins;
using WowViewer.Core.Editor.Tests.Plugins;

namespace WowViewer.Core.Editor.Tests;

public class EditorHostTests
{
    private static readonly EditorBuildVersion Build335 = EditorBuildVersion.Parse("3.3.5.12340");

    private static EditorHost CreateHost(out InMemoryEditorLog log)
    {
        log = new InMemoryEditorLog();
        return new EditorHost(Build335, log);
    }

    [Fact]
    public void Register_duplicate_identity_fails_at_startup()
    {
        EditorHost host = CreateHost(out _);
        host.Register(new TestEditorPlugin("a"));
        host.Register(new TestEditorPlugin("b"));

        var duplicate = new TestEditorPlugin("a");

        var exception = Assert.Throws<EditorPluginRegistrationException>(() => host.Register(duplicate));
        Assert.Contains("a", exception.Message);
    }

    [Fact]
    public void Catalog_lists_unavailable_plugins_with_stated_reason()
    {
        EditorHost host = CreateHost(out _);
        host.Register(new TestEditorPlugin("ok"));
        host.Register(new TestEditorPlugin("needs-data", unavailableReason: "Requires a loaded map"));

        var entries = host.Catalog;

        Assert.Equal(2, entries.Count);
        var unavailable = entries.Single(e => e.Id == "needs-data");
        Assert.False(unavailable.Availability.IsAvailable);
        Assert.Equal("Requires a loaded map", unavailable.Availability.UnavailableReason);
    }

    [Fact]
    public void Activate_calls_activation_exactly_once_and_sets_active()
    {
        EditorHost host = CreateHost(out _);
        var plugin = new TestEditorPlugin("p");
        host.Register(plugin);

        Assert.Equal(EditorActivationResult.Activated, host.Activate("p"));
        Assert.Equal(EditorActivationResult.AlreadyActive, host.Activate("p"));

        Assert.Same(plugin, host.ActivePlugin);
        Assert.Equal(1, plugin.LifecycleCalls.Count(c => c == "OnActivated"));
    }

    [Fact]
    public void Switch_away_and_back_retains_state()
    {
        EditorHost host = CreateHost(out _);
        var first = new TestEditorPlugin("first");
        var second = new TestEditorPlugin("second");
        host.Register(first);
        host.Register(second);

        host.Activate("first");
        Assert.Equal(1, first.LifecycleCalls.Count(c => c == "OnActivated"));

        host.Activate("second");
        Assert.Contains("OnDeactivated", first.LifecycleCalls);
        Assert.Null(host.ActivePlugin is null ? null : (host.ActivePlugin == first ? first : null));
        Assert.Same(second, host.ActivePlugin);

        host.Activate("first");
        Assert.Equal(2, first.LifecycleCalls.Count(c => c == "OnActivated"));
        Assert.Same(first, host.ActivePlugin);
    }

    [Fact]
    public void Fault_during_draw_marks_faulted_and_does_not_rethrow()
    {
        EditorHost host = CreateHost(out InMemoryEditorLog log);
        var plugin = new TestEditorPlugin("p") { FaultOnDraw = true };
        host.Register(plugin);
        host.Activate("p");

        host.Draw();

        var entry = host.Catalog.Single(e => e.Id == "p");
        Assert.Equal(EditorPluginState.Faulted, entry.State);
        Assert.NotNull(entry.FaultMessage);
        Assert.Contains("deliberate draw fault", entry.FaultMessage);
        Assert.Contains(log.Entries, e => e.Level == EditorLogLevel.Error && e.Message.Contains("'p' faulted during draw"));
    }

    [Fact]
    public void Reset_clears_fault_and_allows_reactivation()
    {
        EditorHost host = CreateHost(out _);
        var plugin = new TestEditorPlugin("p") { FaultOnDraw = true };
        host.Register(plugin);
        host.Activate("p");
        host.Draw();

        Assert.Equal(EditorActivationResult.Faulted, host.Activate("p"));

        Assert.True(host.Reset("p"));
        Assert.Equal(EditorPluginState.Inactive, host.Catalog.Single(e => e.Id == "p").State);
        Assert.Equal(EditorActivationResult.Activated, host.Activate("p"));
    }

    [Fact]
    public void Availability_recomputes_when_build_changes()
    {
        EditorHost host = CreateHost(out _);
        var plugin = new TestEditorPlugin(
            "era",
            eras: EditorBuildEraRange.AtLeast(EditorBuildVersion.Parse("1.0")));
        host.Register(plugin);

        Assert.True(host.Catalog.Single(e => e.Id == "era").Availability.IsAvailable);

        host.Build = EditorBuildVersion.Parse("0.5.3.3368");

        var entry = host.Catalog.Single(e => e.Id == "era");
        Assert.False(entry.Availability.IsAvailable);
        Assert.Contains("outside supported eras", entry.Availability.UnavailableReason);
    }

    [Fact]
    public void Activate_unavailable_plugin_returns_unavailable()
    {
        EditorHost host = CreateHost(out _);
        host.Register(new TestEditorPlugin("needs-data", unavailableReason: "no map"));

        Assert.Equal(EditorActivationResult.Unavailable, host.Activate("needs-data"));
        Assert.Null(host.ActivePlugin);
    }

    [Fact]
    public void Activate_unknown_plugin_returns_not_found()
    {
        EditorHost host = CreateHost(out _);

        Assert.Equal(EditorActivationResult.NotFound, host.Activate("missing"));
    }

    [Fact]
    public void Dispose_completes_teardown_even_when_one_plugin_faults()
    {
        EditorHost host = CreateHost(out _);
        var faulting = new TestEditorPlugin("faulty") { FaultOnDeactivate = true };
        var healthy = new TestEditorPlugin("healthy");
        host.Register(faulting);
        host.Register(healthy);
        host.Activate("faulty");

        host.Dispose();

        Assert.Contains("OnDisposed", healthy.LifecycleCalls);
        Assert.Contains("OnDisposed", faulting.LifecycleCalls);
    }

    [Fact]
    public void Reference_plugin_registers_draws_and_tears_down()
    {
        EditorHost host = CreateHost(out _);
        host.Register(new ReferenceEditorPlugin());

        Assert.Equal(EditorActivationResult.Activated, host.Activate(ReferenceEditorPlugin.PluginId));
        host.Draw();
        host.Deactivate();
        host.Dispose();
    }
}