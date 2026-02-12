namespace MdxViewer.Logging;

/// <summary>
/// Simple categorized logging for the viewer. Each category can be independently toggled.
/// Default: only Important and Error are shown. Debug/Verbose hidden until toggled.
/// </summary>
public static class ViewerLog
{
    public enum Category
    {
        General,
        MpqData,
        Terrain,
        Mdx,
        Wmo,
        Dbc,
        Vlm,
        Export,
        Shader,
    }

    public enum Level
    {
        Debug,
        Info,
        Important,
        Error,
    }

    private static Level _minLevel = Level.Important;
    private static bool _verbose = false;
    private static readonly HashSet<Category> _mutedCategories = new() { Category.Mdx, Category.Dbc };

    /// <summary>
    /// When true, Info/Debug messages are printed to console.
    /// Set via --verbose CLI flag. Default: false (only Important/Error print).
    /// </summary>
    public static bool Verbose
    {
        get => _verbose;
        set
        {
            _verbose = value;
            _minLevel = value ? Level.Debug : Level.Important;
        }
    }
    private static readonly List<(DateTime Time, Category Cat, Level Lvl, string Message)> _history = new();
    private static readonly object _lock = new();
    private static int _maxHistory = 500;

    public static Level MinLevel
    {
        get => _minLevel;
        set => _minLevel = value;
    }

    public static void Mute(Category cat) { lock (_lock) _mutedCategories.Add(cat); }
    public static void Unmute(Category cat) { lock (_lock) _mutedCategories.Remove(cat); }
    public static bool IsMuted(Category cat) { lock (_lock) return _mutedCategories.Contains(cat); }

    public static void Log(Category cat, Level level, string message)
    {
        lock (_lock)
        {
            _history.Add((DateTime.Now, cat, level, message));
            if (_history.Count > _maxHistory)
                _history.RemoveRange(0, _history.Count - _maxHistory);
        }

        if (level < _minLevel) return;
        if (level != Level.Error && IsMuted(cat)) return;

        string prefix = $"[{cat}]";
        if (level == Level.Error)
            Console.WriteLine($"{prefix} ERROR: {message}");
        else
            Console.WriteLine($"{prefix} {message}");
    }

    public static void Debug(Category cat, string msg) => Log(cat, Level.Debug, msg);
    public static void Info(Category cat, string msg) => Log(cat, Level.Info, msg);
    public static void Important(Category cat, string msg) => Log(cat, Level.Important, msg);
    public static void Error(Category cat, string msg) => Log(cat, Level.Error, msg);

    /// <summary>Write to console only if --verbose is active. No category/history overhead.</summary>
    public static void Trace(string msg)
    {
        if (_verbose) Console.WriteLine(msg);
    }

    /// <summary>
    /// Get recent log history for UI display.
    /// </summary>
    public static List<(DateTime Time, Category Cat, Level Lvl, string Message)> GetHistory()
    {
        lock (_lock) return new(_history);
    }

    public static IReadOnlySet<Category> MutedCategories
    {
        get { lock (_lock) return new HashSet<Category>(_mutedCategories); }
    }
}
