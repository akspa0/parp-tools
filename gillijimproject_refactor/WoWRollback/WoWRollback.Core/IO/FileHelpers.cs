namespace WoWRollback.Core.IO;

/// <summary>
/// Shared file system utilities for the orchestrator.
/// </summary>
public static class FileHelpers
{
    /// <summary>
    /// Recursively copies a directory and all its contents.
    /// </summary>
    /// <param name="sourceDir">Source directory path</param>
    /// <param name="destinationDir">Destination directory path</param>
    /// <param name="overwrite">Whether to overwrite existing files</param>
    public static void CopyDirectory(string sourceDir, string destinationDir, bool overwrite = true)
    {
        if (string.IsNullOrWhiteSpace(sourceDir))
            throw new ArgumentException("Source directory is required", nameof(sourceDir));
        if (string.IsNullOrWhiteSpace(destinationDir))
            throw new ArgumentException("Destination directory is required", nameof(destinationDir));

        if (!Directory.Exists(sourceDir))
            return;

        if (Directory.Exists(destinationDir) && overwrite)
        {
            Directory.Delete(destinationDir, recursive: true);
        }

        Directory.CreateDirectory(destinationDir);

        // Create all subdirectories
        foreach (var directory in Directory.EnumerateDirectories(sourceDir, "*", SearchOption.AllDirectories))
        {
            var relativeDir = Path.GetRelativePath(sourceDir, directory);
            var targetDir = Path.Combine(destinationDir, relativeDir);
            Directory.CreateDirectory(targetDir);
        }

        // Copy all files
        foreach (var file in Directory.EnumerateFiles(sourceDir, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(sourceDir, file);
            var targetPath = Path.Combine(destinationDir, relative);
            var targetDirectory = Path.GetDirectoryName(targetPath);
            
            if (!string.IsNullOrEmpty(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            
            File.Copy(file, targetPath, overwrite: true);
        }
    }

    /// <summary>
    /// Ensures a directory exists, creating it if necessary.
    /// </summary>
    /// <param name="path">Directory path to ensure exists</param>
    public static void EnsureDirectoryExists(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        Directory.CreateDirectory(path);
    }

    /// <summary>
    /// Safely deletes a directory if it exists.
    /// </summary>
    /// <param name="path">Directory path to delete</param>
    /// <param name="recursive">Whether to delete subdirectories and files</param>
    public static void DeleteDirectoryIfExists(string path, bool recursive = true)
    {
        if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path))
            return;

        try
        {
            Directory.Delete(path, recursive);
        }
        catch
        {
            // Best-effort deletion; ignore errors
        }
    }
}
