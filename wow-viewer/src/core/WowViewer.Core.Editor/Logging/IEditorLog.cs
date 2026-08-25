namespace WowViewer.Core.Editor.Logging;

/// <summary>
/// Minimal structured sink the editor host needs. The viewer implements this with its existing log
/// surface; tests implement it with an in-memory list. Keeping it interface-first means editor
/// library types never depend on the viewer's logging UI.
/// </summary>
public interface IEditorLog
{
    void Trace(string message);
    void Info(string message);
    void Warn(string message);
    void Error(string message, Exception? exception = null);
}