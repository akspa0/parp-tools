using WowViewer.Core.Editor.Logging;
using WowViewer.Core.Editor.Operations;

namespace WowViewer.Core.Editor.Session;

/// <summary>
/// One cross-plugin undo/redo history plus aggregated dirty state and the write-safety policy
/// (Spec 168). The session owns "what changed and can I reverse it"; the applier owns "how do files
/// and the scene react". No game-install or Blizzard-container path is ever a valid write target.
/// </summary>
public sealed class EditorSession
{
    private readonly IEditorOperationApplier _applier;
    private readonly IEditorLog _log;
    private readonly List<EditorOperation> _undo = [];
    private readonly List<EditorOperation> _redo = [];
    private readonly HashSet<string> _dirtyPlugins = new(StringComparer.Ordinal);
    private readonly List<string> _protectedRoots = [];

    private string? _outputDirectory;

    public EditorSession(IEditorOperationApplier applier, IEditorLog log, string? outputDirectory = null)
    {
        _applier = applier ?? throw new ArgumentNullException(nameof(applier));
        _log = log ?? throw new ArgumentNullException(nameof(log));
        OutputDirectory = outputDirectory;
    }

    public string? OutputDirectory
    {
        get => _outputDirectory;
        set => _outputDirectory = string.IsNullOrWhiteSpace(value) ? null : Path.GetFullPath(value);
    }

    public IReadOnlyList<string> ProtectedRoots => _protectedRoots;

    public IReadOnlySet<string> DirtyPlugins => _dirtyPlugins;

    public bool HasUnsavedChanges => _dirtyPlugins.Count > 0;

    public bool CanUndo => _undo.Count > 0;

    public bool CanRedo => _redo.Count > 0;

    public void AddProtectedRoot(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string full = Path.GetFullPath(path);
        if (!_protectedRoots.Contains(full, StringComparer.OrdinalIgnoreCase))
            _protectedRoots.Add(full);
    }

    /// <summary>Records that a plugin applied an operation, advancing undo history and dirty state.</summary>
    public void RecordApplied(EditorOperation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);

        if (operation.Undoable)
        {
            _undo.Add(operation);
            _redo.Clear();
        }
        else
        {
            // A non-undoable operation records a breakpoint: history before it is no longer
            // reachable, so it is cleared rather than offering a broken undo across the gap.
            _undo.Clear();
            _redo.Clear();
        }

        _dirtyPlugins.Add(operation.OriginPluginId);
        _log.Trace($"Session recorded operation '{operation.OperationId}' from plugin '{operation.OriginPluginId}'.");
    }

    public EditorOperation? Undo()
    {
        if (_undo.Count == 0)
            return null;

        EditorOperation operation = _undo[^1];
        _undo.RemoveAt(_undo.Count - 1);

        EditorOperation reverse = operation.CreateReverse();
        _applier.Apply(reverse);

        _redo.Add(operation);
        _dirtyPlugins.Add(operation.OriginPluginId);
        _log.Info($"Undid operation '{operation.OperationId}' from plugin '{operation.OriginPluginId}'.");
        return reverse;
    }

    public EditorOperation? Redo()
    {
        if (_redo.Count == 0)
            return null;

        EditorOperation operation = _redo[^1];
        _redo.RemoveAt(_redo.Count - 1);

        _applier.Apply(operation);

        _undo.Add(operation);
        _dirtyPlugins.Add(operation.OriginPluginId);
        _log.Info($"Redid operation '{operation.OperationId}' from plugin '{operation.OriginPluginId}'.");
        return operation;
    }

    /// <summary>Marks all plugins clean and reports which plugins were saved.</summary>
    public IReadOnlyList<string> SaveAll()
    {
        string[] saved = [.. _dirtyPlugins.OrderBy(id => id, StringComparer.Ordinal)];
        _dirtyPlugins.Clear();
        foreach (string pluginId in saved)
            _log.Info($"Plugin '{pluginId}' saved.");

        return saved;
    }

    public void MarkClean(string pluginId)
        => _dirtyPlugins.Remove(pluginId);

    public bool IsProtectedPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;

        string full = Path.GetFullPath(path);
        char separator = Path.DirectorySeparatorChar;

        foreach (string root in _protectedRoots)
        {
            string trimmed = root.TrimEnd(separator, Path.AltDirectorySeparatorChar);
            if (full.Equals(trimmed, StringComparison.OrdinalIgnoreCase))
                return true;

            if (full.StartsWith(trimmed + separator, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    public bool IsContainerPath(string path)
        => Path.GetExtension(path).Equals(".mpq", StringComparison.OrdinalIgnoreCase);

    /// <summary>Refuses a write into any protected game-install path or Blizzard container.</summary>
    public void GuardWritablePath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (IsProtectedPath(path))
            throw new EditorProtectedPathException(path, "the path is inside a configured game install (never write into a client).");

        if (IsContainerPath(path))
            throw new EditorProtectedPathException(path, "Blizzard containers (MPQ/CASC) are read-only inputs, never outputs.");
    }

    /// <summary>Resolves an output path under the configured output directory and guards it.</summary>
    public string ResolveOutputPath(string relativeOrFullPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(relativeOrFullPath);

        string full;
        if (Path.IsPathFullyQualified(relativeOrFullPath))
        {
            full = Path.GetFullPath(relativeOrFullPath);
        }
        else
        {
            if (_outputDirectory is null)
                throw new EditorSessionException("No output directory configured; cannot resolve a relative output path.");

            full = Path.GetFullPath(Path.Combine(_outputDirectory, relativeOrFullPath));
        }

        GuardWritablePath(full);
        return full;
    }
}