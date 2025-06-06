using System;
using System.Collections.Generic;
using System.IO;

namespace WoWToolbox.Tests.Utilities;

/// <summary>
/// Manages centralized output collection from all parts of the PM4Tool into timestamped folders
/// </summary>
public static class CentralizedOutputManager
{
    private static readonly string BaseOutputPath = Path.Combine("output");
    private static string? _currentSessionFolder;
    
    /// <summary>
    /// Gets or creates the current session output folder with timestamp
    /// </summary>
    public static string CurrentSessionFolder
    {
        get
        {
            if (_currentSessionFolder == null)
            {
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                _currentSessionFolder = Path.Combine(BaseOutputPath, $"pm4tool_session_{timestamp}");
                Directory.CreateDirectory(_currentSessionFolder);
            }
            return _currentSessionFolder;
        }
    }

    /// <summary>
    /// Resets the session folder - forces creation of a new timestamped folder
    /// </summary>
    public static void ResetSession()
    {
        _currentSessionFolder = null;
    }

    /// <summary>
    /// Creates a subfolder within the current session for specific component outputs
    /// </summary>
    /// <param name="componentName">Name of the component (e.g., "mprl_mesh", "gui_exports", "analysis")</param>
    /// <returns>Full path to the component subfolder</returns>
    public static string CreateComponentFolder(string componentName)
    {
        var componentPath = Path.Combine(CurrentSessionFolder, componentName);
        Directory.CreateDirectory(componentPath);
        return componentPath;
    }

    /// <summary>
    /// Copies a file to the centralized output location
    /// </summary>
    /// <param name="sourceFilePath">Source file to copy</param>
    /// <param name="componentName">Component that generated the file</param>
    /// <param name="destinationFileName">Optional custom destination filename</param>
    /// <returns>Path where the file was copied</returns>
    public static string CopyFileToSession(string sourceFilePath, string componentName, string? destinationFileName = null)
    {
        if (!File.Exists(sourceFilePath))
        {
            throw new FileNotFoundException($"Source file not found: {sourceFilePath}");
        }

        var componentFolder = CreateComponentFolder(componentName);
        var fileName = destinationFileName ?? Path.GetFileName(sourceFilePath);
        var destinationPath = Path.Combine(componentFolder, fileName);
        
        File.Copy(sourceFilePath, destinationPath, overwrite: true);
        return destinationPath;
    }

    /// <summary>
    /// Copies an entire directory to the centralized output location
    /// </summary>
    /// <param name="sourceDirectoryPath">Source directory to copy</param>
    /// <param name="componentName">Component that generated the directory</param>
    /// <param name="destinationFolderName">Optional custom destination folder name</param>
    /// <returns>Path where the directory was copied</returns>
    public static string CopyDirectoryToSession(string sourceDirectoryPath, string componentName, string? destinationFolderName = null)
    {
        if (!Directory.Exists(sourceDirectoryPath))
        {
            throw new DirectoryNotFoundException($"Source directory not found: {sourceDirectoryPath}");
        }

        var componentFolder = CreateComponentFolder(componentName);
        var folderName = destinationFolderName ?? Path.GetFileName(sourceDirectoryPath);
        var destinationPath = Path.Combine(componentFolder, folderName);
        
        CopyDirectoryRecursive(sourceDirectoryPath, destinationPath);
        return destinationPath;
    }

    /// <summary>
    /// Creates a report file in the session folder documenting all collected outputs
    /// </summary>
    /// <param name="additionalInfo">Additional information to include in the report</param>
    public static void GenerateSessionReport(Dictionary<string, object>? additionalInfo = null)
    {
        var reportPath = Path.Combine(CurrentSessionFolder, "session_report.md");
        var timestamp = DateTime.Now;
        
        using var writer = new StreamWriter(reportPath);
        
        writer.WriteLine("# PM4Tool Centralized Output Session Report");
        writer.WriteLine($"**Generated:** {timestamp:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine($"**Session Folder:** `{CurrentSessionFolder}`");
        writer.WriteLine();
        
        // List all components and their outputs
        writer.WriteLine("## Component Outputs");
        writer.WriteLine();
        
        var componentDirs = Directory.GetDirectories(CurrentSessionFolder);
        foreach (var componentDir in componentDirs)
        {
            var componentName = Path.GetFileName(componentDir);
            writer.WriteLine($"### {componentName}");
            
            var files = Directory.GetFiles(componentDir, "*", SearchOption.AllDirectories);
            var subdirs = Directory.GetDirectories(componentDir, "*", SearchOption.AllDirectories);
            
            writer.WriteLine($"- **Files:** {files.Length}");
            writer.WriteLine($"- **Subdirectories:** {subdirs.Length}");
            writer.WriteLine($"- **Location:** `{componentDir}`");
            writer.WriteLine();
            
            if (files.Length > 0)
            {
                writer.WriteLine("**Generated Files:**");
                foreach (var file in files.Take(20)) // Limit to first 20 files
                {
                    var relativePath = Path.GetRelativePath(CurrentSessionFolder, file);
                    var fileSize = new FileInfo(file).Length;
                    writer.WriteLine($"- `{relativePath}` ({FormatFileSize(fileSize)})");
                }
                if (files.Length > 20)
                {
                    writer.WriteLine($"- ... and {files.Length - 20} more files");
                }
                writer.WriteLine();
            }
        }
        
        // Add additional information if provided
        if (additionalInfo != null && additionalInfo.Count > 0)
        {
            writer.WriteLine("## Additional Information");
            writer.WriteLine();
            foreach (var kvp in additionalInfo)
            {
                writer.WriteLine($"**{kvp.Key}:** {kvp.Value}");
            }
            writer.WriteLine();
        }
        
        // Add summary statistics
        var allFiles = Directory.GetFiles(CurrentSessionFolder, "*", SearchOption.AllDirectories);
        var totalSize = allFiles.Sum(f => new FileInfo(f).Length);
        
        writer.WriteLine("## Session Summary");
        writer.WriteLine($"- **Total Files:** {allFiles.Length:N0}");
        writer.WriteLine($"- **Total Size:** {FormatFileSize(totalSize)}");
        writer.WriteLine($"- **Components:** {componentDirs.Length}");
        writer.WriteLine($"- **Session Duration:** Started at {timestamp:HH:mm:ss}");
        writer.WriteLine();
        writer.WriteLine("---");
        writer.WriteLine("*Generated by PM4Tool Centralized Output Manager*");
    }

    /// <summary>
    /// Collects outputs from common PM4Tool directories into the centralized session
    /// </summary>
    public static void CollectExistingOutputs()
    {
        var outputCollections = new List<(string SourcePattern, string ComponentName)>
        {
            ("output/combined_mprl_mesh*", "mprl_mesh"),
            ("output/comprehensive_pipeline_*", "comprehensive_pipeline"),
            ("output/**/PM4_BatchOutput", "pm4_batch"),
            ("output/**/CompleteGeometryExport", "geometry_export"),
            ("output/**/ChunkRelationshipAnalysis", "chunk_analysis"),
            ("output/**/UnknownFieldInvestigation", "field_investigation"),
            ("output/**/EnhancedObjExport", "enhanced_export"),
            ("output/test_*", "test_outputs")
        };

        foreach (var (sourcePattern, componentName) in outputCollections)
        {
            try
            {
                CollectOutputsByPattern(sourcePattern, componentName);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not collect outputs for pattern '{sourcePattern}': {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Collects all PM4Tool outputs and generates a comprehensive session report
    /// </summary>
    /// <param name="includeExisting">Whether to include existing output directories</param>
    /// <returns>Path to the session folder</returns>
    public static string CreateComprehensiveSession(bool includeExisting = true)
    {
        ResetSession(); // Start fresh session
        
        if (includeExisting)
        {
            CollectExistingOutputs();
        }
        
        // Generate comprehensive report
        var additionalInfo = new Dictionary<string, object>
        {
            {"PM4Tool Version", "1.0.0"},
            {"Collection Mode", includeExisting ? "Includes Existing Outputs" : "New Outputs Only"},
            {"Output Collection", "Centralized"},
            {"Z-Coordinate Fix", "Applied (negated for correct orientation)"}
        };
        
        GenerateSessionReport(additionalInfo);
        
        Console.WriteLine($"📁 Centralized output session created: {CurrentSessionFolder}");
        return CurrentSessionFolder;
    }

    #region Private Helper Methods

    private static void CopyDirectoryRecursive(string sourceDir, string destDir)
    {
        Directory.CreateDirectory(destDir);
        
        // Copy files
        foreach (var file in Directory.GetFiles(sourceDir))
        {
            var destFile = Path.Combine(destDir, Path.GetFileName(file));
            File.Copy(file, destFile, overwrite: true);
        }
        
        // Copy subdirectories
        foreach (var subDir in Directory.GetDirectories(sourceDir))
        {
            var destSubDir = Path.Combine(destDir, Path.GetFileName(subDir));
            CopyDirectoryRecursive(subDir, destSubDir);
        }
    }

    private static void CollectOutputsByPattern(string pattern, string componentName)
    {
        var basePath = pattern.Contains("*") ? Path.GetDirectoryName(pattern) ?? "." : pattern;
        var searchPattern = pattern.Contains("*") ? Path.GetFileName(pattern) : "*";
        
        if (Directory.Exists(basePath))
        {
            var matchingDirs = Directory.GetDirectories(basePath, searchPattern);
            foreach (var dir in matchingDirs)
            {
                var destName = $"{Path.GetFileName(dir)}_{DateTime.Now:HHmmss}";
                CopyDirectoryToSession(dir, componentName, destName);
            }
            
            var matchingFiles = Directory.GetFiles(basePath, searchPattern);
            foreach (var file in matchingFiles)
            {
                CopyFileToSession(file, componentName);
            }
        }
    }

    private static string FormatFileSize(long bytes)
    {
        string[] suffixes = { "B", "KB", "MB", "GB", "TB" };
        int suffixIndex = 0;
        double size = bytes;
        
        while (size >= 1024 && suffixIndex < suffixes.Length - 1)
        {
            size /= 1024;
            suffixIndex++;
        }
        
        return $"{size:F1} {suffixes[suffixIndex]}";
    }

    #endregion
} 