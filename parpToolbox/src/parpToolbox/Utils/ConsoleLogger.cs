using System;
using System.IO;

namespace ParpToolbox.Utils
{
    /// <summary>
    /// Simple console logger that writes to both console and a log file simultaneously.
    /// </summary>
    public static class ConsoleLogger
    {
        private static StreamWriter? _logWriter;
        private static string? _logFilePath;

        /// <summary>
        /// Initialize logging to a timestamped file in the specified directory.
        /// </summary>
        public static void Initialize(string outputDirectory)
        {
            try
            {
                Directory.CreateDirectory(outputDirectory);
                
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                _logFilePath = Path.Combine(outputDirectory, $"parpToolbox_log_{timestamp}.txt");
                
                _logWriter = new StreamWriter(_logFilePath, append: false);
                _logWriter.AutoFlush = true; // Ensure immediate writes
                
                WriteLine($"=== parpToolbox Log Started at {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===");
                WriteLine($"Log file: {_logFilePath}");
                WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to initialize log file: {ex.Message}");
                _logWriter = null;
            }
        }

        /// <summary>
        /// Write a line to both console and log file.
        /// </summary>
        public static void WriteLine(string message = "")
        {
            Console.WriteLine(message);
            _logWriter?.WriteLine(message);
        }

        /// <summary>
        /// Write text to both console and log file without newline.
        /// </summary>
        public static void Write(string message)
        {
            Console.Write(message);
            _logWriter?.Write(message);
        }

        /// <summary>
        /// Close the log file and clean up resources.
        /// </summary>
        public static void Close()
        {
            if (_logWriter != null)
            {
                WriteLine();
                WriteLine($"=== parpToolbox Log Ended at {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===");
                _logWriter.Close();
                _logWriter = null;
                
                if (!string.IsNullOrEmpty(_logFilePath))
                {
                    Console.WriteLine($"Log saved to: {_logFilePath}");
                }
            }
        }
    }
}
