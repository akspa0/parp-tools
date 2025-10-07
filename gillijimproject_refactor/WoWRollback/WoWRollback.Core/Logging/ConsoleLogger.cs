namespace WoWRollback.Core.Logging;

/// <summary>
/// Simple console logger for orchestrator output.
/// Provides structured logging with timestamps and severity levels.
/// </summary>
public static class ConsoleLogger
{
    /// <summary>
    /// Logs an informational message to console.
    /// </summary>
    public static void Info(string message)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        Console.WriteLine($"[INFO] {timestamp} {message}");
    }

    /// <summary>
    /// Logs an error message to console error stream.
    /// </summary>
    public static void Error(string message)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        Console.Error.WriteLine($"[ERROR] {timestamp} {message}");
    }

    /// <summary>
    /// Logs a warning message to console.
    /// </summary>
    public static void Warn(string message)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        Console.WriteLine($"[WARN] {timestamp} {message}");
    }

    /// <summary>
    /// Logs a debug message to console (only if verbose).
    /// </summary>
    public static void Debug(string message, bool verbose = false)
    {
        if (!verbose)
            return;

        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        Console.WriteLine($"[DEBUG] {timestamp} {message}");
    }

    /// <summary>
    /// Logs a success message with green color (if supported).
    /// </summary>
    public static void Success(string message)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        var originalColor = Console.ForegroundColor;
        
        try
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[SUCCESS] {timestamp} {message}");
        }
        finally
        {
            Console.ForegroundColor = originalColor;
        }
    }
}
