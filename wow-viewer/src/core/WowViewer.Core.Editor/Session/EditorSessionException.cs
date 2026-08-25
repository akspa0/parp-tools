namespace WowViewer.Core.Editor.Session;

public class EditorSessionException : Exception
{
    public EditorSessionException(string message) : base(message) { }
    public EditorSessionException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>Thrown when a write targets a protected game-install or Blizzard-container path.</summary>
public sealed class EditorProtectedPathException : EditorSessionException
{
    public EditorProtectedPathException(string path, string reason)
        : base($"Write refused for '{path}': {reason}")
    {
        Path = path;
    }

    public string Path { get; }
}