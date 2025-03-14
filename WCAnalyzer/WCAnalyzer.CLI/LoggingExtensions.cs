using System;
using System.IO;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.CLI
{
    public static class LoggingExtensions
    {
        public static ILoggingBuilder AddFile(this ILoggingBuilder builder, string filePath)
        {
            builder.AddProvider(new FileLoggerProvider(filePath));
            return builder;
        }
    }

    public class FileLoggerProvider : ILoggerProvider
    {
        private readonly string _filePath;

        public FileLoggerProvider(string filePath)
        {
            _filePath = filePath;
            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
        }

        public ILogger CreateLogger(string categoryName)
        {
            return new FileLogger(_filePath, categoryName);
        }

        public void Dispose()
        {
        }
    }

    public class FileLogger : ILogger
    {
        private readonly string _filePath;
        private readonly string _categoryName;
        private static readonly object _lock = new object();

        public FileLogger(string filePath, string categoryName)
        {
            _filePath = filePath;
            _categoryName = categoryName;
        }

        public IDisposable BeginScope<TState>(TState state)
        {
            return null;
        }

        public bool IsEnabled(LogLevel logLevel)
        {
            return logLevel != LogLevel.None;
        }

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            if (!IsEnabled(logLevel))
            {
                return;
            }

            var message = formatter(state, exception);
            var logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} [{logLevel}] {_categoryName}: {message}";
            if (exception != null)
            {
                logEntry += Environment.NewLine + exception.ToString();
            }

            lock (_lock)
            {
                File.AppendAllText(_filePath, logEntry + Environment.NewLine);
            }
        }
    }
} 