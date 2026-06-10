using System;
using System.IO;

namespace WoWToolbox.Core.v2.Foundation.Utilities
{
    /// <summary>
    /// Centralized output management with timestamped directories for all tools and tests.
    /// Simplifies output organization across the entire WoWToolbox ecosystem.
    /// </summary>
    public static class OutputManager
    {
        private static string? _currentSessionId;
        private static string? _currentOutputRoot;

        /// <summary>
        /// Gets or sets the base output directory. Defaults to "output" in the current directory.
        /// </summary>
        public static string BaseOutputDirectory { get; set; } = "output";

        /// <summary>
        /// Gets the current session ID (timestamp). Creates one if it doesn't exist.
        /// </summary>
        public static string CurrentSessionId
        {
            get
            {
                if (_currentSessionId == null)
                {
                    _currentSessionId = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                }
                return _currentSessionId;
            }
        }

        /// <summary>
        /// Gets the current session output root directory.
        /// </summary>
        public static string CurrentOutputRoot
        {
            get
            {
                if (_currentOutputRoot == null)
                {
                    _currentOutputRoot = Path.Combine(BaseOutputDirectory, CurrentSessionId);
                    Directory.CreateDirectory(_currentOutputRoot);
                }
                return _currentOutputRoot;
            }
        }

        /// <summary>
        /// Creates and returns a subdirectory within the current session output.
        /// </summary>
        /// <param name="subdirectoryName">Name of the subdirectory to create</param>
        /// <returns>Full path to the created subdirectory</returns>
        public static string CreateSubdirectory(string subdirectoryName)
        {
            var fullPath = Path.Combine(CurrentOutputRoot, subdirectoryName);
            Directory.CreateDirectory(fullPath);
            return fullPath;
        }

        /// <summary>
        /// Gets a file path within the current session output directory.
        /// Creates the directory structure if it doesn't exist.
        /// </summary>
        /// <param name="fileName">Name of the file</param>
        /// <param name="subdirectory">Optional subdirectory within the session</param>
        /// <returns>Full path to the file location</returns>
        public static string GetOutputFilePath(string fileName, string? subdirectory = null)
        {
            string directory = subdirectory != null 
                ? CreateSubdirectory(subdirectory)
                : CurrentOutputRoot;
            
            return Path.Combine(directory, fileName);
        }

        /// <summary>
        /// Resets the session, creating a new timestamped directory for the next session.
        /// </summary>
        public static void StartNewSession()
        {
            _currentSessionId = null;
            _currentOutputRoot = null;
        }

        /// <summary>
        /// Gets a standardized output path for a specific tool.
        /// </summary>
        /// <param name="toolName">Name of the tool (e.g., "PM4Parsing", "BuildingExtraction")</param>
        /// <param name="fileName">Name of the output file</param>
        /// <param name="sourceFileName">Optional source file name for context</param>
        /// <returns>Full path to the output file</returns>
        public static string GetToolOutputPath(string toolName, string fileName, string? sourceFileName = null)
        {
            string subdirectory = sourceFileName != null 
                ? Path.Combine(toolName, Path.GetFileNameWithoutExtension(sourceFileName))
                : toolName;
            
            return GetOutputFilePath(fileName, subdirectory);
        }

        /// <summary>
        /// Logs the current output session information.
        /// </summary>
        public static void LogSessionInfo()
        {
            Console.WriteLine($"📁 Output Session: {CurrentSessionId}");
            Console.WriteLine($"📂 Output Directory: {CurrentOutputRoot}");
        }

        /// <summary>
        /// Cleans up old output directories, keeping only the most recent ones.
        /// </summary>
        /// <param name="keepCount">Number of recent sessions to keep (default: 10)</param>
        public static void CleanupOldSessions(int keepCount = 10)
        {
            if (!Directory.Exists(BaseOutputDirectory)) return;

            var directories = Directory.GetDirectories(BaseOutputDirectory)
                .Select(d => new DirectoryInfo(d))
                .Where(d => DateTime.TryParseExact(d.Name, "yyyyMMdd_HHmmss", null, 
                    System.Globalization.DateTimeStyles.None, out _))
                .OrderByDescending(d => d.CreationTime)
                .Skip(keepCount);

            foreach (var dir in directories)
            {
                try
                {
                    dir.Delete(true);
                    Console.WriteLine($"🗑️ Cleaned up old session: {dir.Name}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ Could not delete {dir.Name}: {ex.Message}");
                }
            }
        }
    }
} 