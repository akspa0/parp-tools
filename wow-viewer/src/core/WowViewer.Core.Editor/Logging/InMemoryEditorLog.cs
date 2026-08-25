namespace WowViewer.Core.Editor.Logging;

public enum EditorLogLevel
{
    Trace,
    Info,
    Warn,
    Error,
}

public readonly record struct EditorLogEntry(EditorLogLevel Level, string Message, Exception? Exception)
{
    public override string ToString()
        => Exception is null ? $"[{Level}] {Message}" : $"[{Level}] {Message} | {Exception.GetType().Name}: {Exception.Message}";
}

/// <summary>
/// Thread-safe in-memory log used by tests and headless hosts. Records entries so fault
/// containment and lifecycle transitions are observable without a UI dependency.
/// </summary>
public sealed class InMemoryEditorLog : IEditorLog
{
    private readonly object _gate = new();
    private readonly List<EditorLogEntry> _entries = [];

    public IReadOnlyList<EditorLogEntry> Entries
    {
        get
        {
            lock (_gate)
                return _entries.ToArray();
        }
    }

    public void Trace(string message) => Add(EditorLogLevel.Trace, message, null);
    public void Info(string message) => Add(EditorLogLevel.Info, message, null);
    public void Warn(string message) => Add(EditorLogLevel.Warn, message, null);
    public void Error(string message, Exception? exception = null) => Add(EditorLogLevel.Error, message, exception);

    private void Add(EditorLogLevel level, string message, Exception? exception)
    {
        lock (_gate)
            _entries.Add(new EditorLogEntry(level, message, exception));
    }
}