using System;
using System.IO;
using WCAnalyzer.Core.Services;
using Microsoft.Extensions.Logging;
using MsLogLevel = Microsoft.Extensions.Logging.LogLevel;

namespace WCAnalyzer.Core.Utilities
{
    /// <summary>
    /// Console implementation of the logging service.
    /// </summary>
    public class ConsoleLogger : ILoggingService
    {
        private readonly Services.LogLevel _minimumLevel;

        /// <summary>
        /// Creates a new instance of the ConsoleLogger class.
        /// </summary>
        /// <param name="minimumLevel">The minimum log level to display.</param>
        public ConsoleLogger(Services.LogLevel minimumLevel = Services.LogLevel.Information)
        {
            _minimumLevel = minimumLevel;
        }

        /// <inheritdoc/>
        public void LogDebug(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Debug)
            {
                var originalColor = Console.ForegroundColor;
                Console.ForegroundColor = ConsoleColor.Gray;
                Console.WriteLine($"[DEBUG] {message}");
                Console.ForegroundColor = originalColor;
            }
        }

        /// <inheritdoc/>
        public void LogInfo(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Information)
            {
                var originalColor = Console.ForegroundColor;
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine($"[INFO] {message}");
                Console.ForegroundColor = originalColor;
            }
        }

        /// <inheritdoc/>
        public void LogWarning(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Warning)
            {
                var originalColor = Console.ForegroundColor;
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[WARN] {message}");
                Console.ForegroundColor = originalColor;
            }
        }

        /// <inheritdoc/>
        public void LogError(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Error)
            {
                var originalColor = Console.ForegroundColor;
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] {message}");
                Console.ForegroundColor = originalColor;
            }
        }

        /// <inheritdoc/>
        public void LogCritical(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Critical)
            {
                var originalColor = Console.ForegroundColor;
                Console.ForegroundColor = ConsoleColor.DarkRed;
                Console.WriteLine($"[CRITICAL] {message}");
                Console.ForegroundColor = originalColor;
            }
        }
    }

    /// <summary>
    /// File implementation of the logging service.
    /// </summary>
    public class FileLogger : ILoggingService, IDisposable
    {
        private readonly Services.LogLevel _minimumLevel;
        private readonly StreamWriter _writer;
        private bool _disposed = false;

        /// <summary>
        /// Creates a new instance of the FileLogger class.
        /// </summary>
        /// <param name="filePath">The path to the log file.</param>
        /// <param name="minimumLevel">The minimum log level to log.</param>
        public FileLogger(string filePath, Services.LogLevel minimumLevel = Services.LogLevel.Information)
        {
            _minimumLevel = minimumLevel;

            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            _writer = new StreamWriter(filePath, true);
            _writer.AutoFlush = true;

            // Write header
            _writer.WriteLine($"--- Log started at {DateTime.Now:yyyy-MM-dd HH:mm:ss} ---");
        }

        /// <inheritdoc/>
        public void LogDebug(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Debug)
            {
                _writer.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [DEBUG] {message}");
            }
        }

        /// <inheritdoc/>
        public void LogInfo(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Information)
            {
                _writer.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [INFO] {message}");
            }
        }

        /// <inheritdoc/>
        public void LogWarning(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Warning)
            {
                _writer.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [WARN] {message}");
            }
        }

        /// <inheritdoc/>
        public void LogError(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Error)
            {
                _writer.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [ERROR] {message}");
            }
        }

        /// <inheritdoc/>
        public void LogCritical(string message)
        {
            if (_minimumLevel <= Services.LogLevel.Critical)
            {
                _writer.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [CRITICAL] {message}");
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the resources used by the FileLogger.
        /// </summary>
        /// <param name="disposing">Whether to dispose managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _writer.WriteLine($"--- Log ended at {DateTime.Now:yyyy-MM-dd HH:mm:ss} ---");
                    _writer.Dispose();
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer.
        /// </summary>
        ~FileLogger()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Composite implementation of the logging service that logs to multiple destinations.
    /// </summary>
    public class CompositeLogger : ILoggingService, IDisposable
    {
        private readonly ILoggingService[] _loggers;
        private bool _disposed = false;

        /// <summary>
        /// Creates a new instance of the CompositeLogger class.
        /// </summary>
        /// <param name="loggers">The loggers to use.</param>
        public CompositeLogger(params ILoggingService[] loggers)
        {
            _loggers = loggers ?? throw new ArgumentNullException(nameof(loggers));
        }

        /// <inheritdoc/>
        public void LogDebug(string message)
        {
            foreach (var logger in _loggers)
            {
                logger.LogDebug(message);
            }
        }

        /// <inheritdoc/>
        public void LogInfo(string message)
        {
            foreach (var logger in _loggers)
            {
                logger.LogInfo(message);
            }
        }

        /// <inheritdoc/>
        public void LogWarning(string message)
        {
            foreach (var logger in _loggers)
            {
                logger.LogWarning(message);
            }
        }

        /// <inheritdoc/>
        public void LogError(string message)
        {
            foreach (var logger in _loggers)
            {
                logger.LogError(message);
            }
        }

        /// <inheritdoc/>
        public void LogCritical(string message)
        {
            foreach (var logger in _loggers)
            {
                logger.LogCritical(message);
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the resources used by the CompositeLogger.
        /// </summary>
        /// <param name="disposing">Whether to dispose managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    foreach (var logger in _loggers)
                    {
                        if (logger is IDisposable disposable)
                        {
                            disposable.Dispose();
                        }
                    }
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer.
        /// </summary>
        ~CompositeLogger()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Implementation of <see cref="ILoggingService"/> using Microsoft.Extensions.Logging.
    /// </summary>
    public class LoggingService : ILoggingService
    {
        private readonly ILogger _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="LoggingService"/> class.
        /// </summary>
        /// <param name="logger">The logger to use.</param>
        public LoggingService(ILogger<LoggingService> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <inheritdoc/>
        public void LogDebug(string message)
        {
            _logger.LogDebug(message);
        }

        /// <inheritdoc/>
        public void LogError(string message)
        {
            _logger.LogError(message);
        }

        /// <inheritdoc/>
        public void LogInfo(string message)
        {
            _logger.LogInformation(message);
        }

        /// <inheritdoc/>
        public void LogWarning(string message)
        {
            _logger.LogWarning(message);
        }

        /// <inheritdoc/>
        public void LogCritical(string message)
        {
            _logger.LogCritical(message);
        }
    }
}