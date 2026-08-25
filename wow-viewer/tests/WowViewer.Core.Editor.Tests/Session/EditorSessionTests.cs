using WowViewer.Core.Editor.Logging;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.Editor.Session;

namespace WowViewer.Core.Editor.Tests.Session;

public class EditorSessionTests
{
    private sealed class RecordingApplier : IEditorOperationApplier
    {
        public List<EditorOperation> Applied { get; } = [];

        public void Apply(EditorOperation operation) => Applied.Add(operation);
    }

    private static EditorSession CreateSession(RecordingApplier applier)
        => new(applier, new InMemoryEditorLog());

    private static PlaceholderOperation Op(string id, string plugin, bool undoable = true)
        => new(id, plugin, undoable);

    [Fact]
    public void Undo_reverses_in_reverse_order_across_plugins()
    {
        var applier = new RecordingApplier();
        EditorSession session = CreateSession(applier);

        session.RecordApplied(Op("a-1", "pluginA"));
        session.RecordApplied(Op("b-1", "pluginB"));
        session.RecordApplied(Op("a-2", "pluginA"));

        session.Undo();
        session.Undo();
        session.Undo();

        Assert.Equal(
            ["reverse:a-2", "reverse:b-1", "reverse:a-1"],
            applier.Applied.Select(op => ((PlaceholderOperation)op).Id).ToArray());
        Assert.False(session.CanUndo);
    }

    [Fact]
    public void Undo_applies_the_reverse_operation()
    {
        var applier = new RecordingApplier();
        EditorSession session = CreateSession(applier);

        session.RecordApplied(Op("a-1", "pluginA"));

        session.Undo();

        Assert.Single(applier.Applied);
        Assert.StartsWith("reverse:", ((PlaceholderOperation)applier.Applied[0]).Id);
    }

    [Fact]
    public void Dirty_state_clears_only_after_save()
    {
        var applier = new RecordingApplier();
        EditorSession session = CreateSession(applier);

        session.RecordApplied(Op("a-1", "pluginA"));
        session.RecordApplied(Op("b-1", "pluginB"));

        Assert.True(session.HasUnsavedChanges);
        Assert.Equal(2, session.DirtyPlugins.Count);
        Assert.Contains("pluginA", session.DirtyPlugins);
        Assert.Contains("pluginB", session.DirtyPlugins);

        IReadOnlyList<string> saved = session.SaveAll();
        Assert.Equal(2, saved.Count);
        Assert.False(session.HasUnsavedChanges);
    }

    [Fact]
    public void Non_undoable_operation_clears_undo_history()
    {
        var applier = new RecordingApplier();
        EditorSession session = CreateSession(applier);

        session.RecordApplied(Op("a-1", "pluginA"));
        session.RecordApplied(Op("b-1", "pluginB", undoable: false));

        Assert.False(session.CanUndo);
    }

    [Fact]
    public void Redo_replays_after_undo()
    {
        var applier = new RecordingApplier();
        EditorSession session = CreateSession(applier);

        session.RecordApplied(Op("a-1", "pluginA"));

        Assert.False(session.CanRedo);
        session.Undo();
        Assert.True(session.CanRedo);
        session.Redo();
        Assert.False(session.CanRedo);
        Assert.True(session.CanUndo);
    }

    [Fact]
    public void GuardWritablePath_refuses_game_install_and_container()
    {
        var applier = new RecordingApplier();
        EditorSession session = CreateSession(applier);
        session.AddProtectedRoot("C:/Clients/WoW335");

        Assert.Throws<EditorProtectedPathException>(() => session.GuardWritablePath("C:/Clients/WoW335/Data/world.MPQ"));
        Assert.Throws<EditorProtectedPathException>(() => session.GuardWritablePath("C:/Clients/WoW335/Data/enUS/patch-enUS-3.MPQ"));
        Assert.Throws<EditorProtectedPathException>(() => session.GuardWritablePath("C:/Output/out.mpq"));
    }

    [Fact]
    public void GuardWritablePath_allows_output_dir_paths()
    {
        var applier = new RecordingApplier();
        EditorSession session = CreateSession(applier);
        session.AddProtectedRoot("C:/Clients/WoW335");

        session.GuardWritablePath("C:/Output/Maps/Foo/Foo_1_0.adt");
    }

    [Fact]
    public void ResolveOutputPath_requires_output_directory_for_relative_paths()
    {
        var applier = new RecordingApplier();
        EditorSession session = new(applier, new InMemoryEditorLog());

        Assert.Throws<EditorSessionException>(() => session.ResolveOutputPath("Foo_1_0.adt"));
    }

    private sealed class PlaceholderOperation : EditorOperation
    {
        public string Id { get; }

        public PlaceholderOperation(string id, string plugin, bool undoable)
            : base(id, plugin, $"Op {id}", undoable)
        {
            Id = id;
        }

        public override IReadOnlyList<string> AffectedPaths => [];

        public override EditorOperation CreateReverse()
            => new PlaceholderOperation($"reverse:{Id}", OriginPluginId, Undoable);
    }
}