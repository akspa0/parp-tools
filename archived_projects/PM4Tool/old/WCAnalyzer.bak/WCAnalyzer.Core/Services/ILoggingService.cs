namespace WCAnalyzer.Core.Services;

/// <summary>
/// Interface for logging services.
/// </summary>
public interface ILoggingService
{
    /// <summary>
    /// Logs a debug message.
    /// </summary>
    /// <param name="message">The message to log.</param>
    void LogDebug(string message);

    /// <summary>
    /// Logs an informational message.
    /// </summary>
    /// <param name="message">The message to log.</param>
    void LogInfo(string message);

    /// <summary>
    /// Logs a warning message.
    /// </summary>
    /// <param name="message">The message to log.</param>
    void LogWarning(string message);

    /// <summary>
    /// Logs an error message.
    /// </summary>
    /// <param name="message">The message to log.</param>
    void LogError(string message);

    /// <summary>
    /// Logs a critical message.
    /// </summary>
    /// <param name="message">The message to log.</param>
    void LogCritical(string message);
}

/// <summary>
/// Defines the log level.
/// </summary>
public enum LogLevel
{
    /// <summary>
    /// Debug level for detailed diagnostic information.
    /// </summary>
    Debug,

    /// <summary>
    /// Information level for general information.
    /// </summary>
    Information,

    /// <summary>
    /// Warning level for potential issues.
    /// </summary>
    Warning,

    /// <summary>
    /// Error level for errors that don't stop the application.
    /// </summary>
    Error,

    /// <summary>
    /// Critical level for critical errors that stop the application.
    /// </summary>
    Critical
}