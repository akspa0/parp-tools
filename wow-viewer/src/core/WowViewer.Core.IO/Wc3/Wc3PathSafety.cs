namespace WowViewer.Core.IO.Wc3;

// Use Windows-safe virtual names even on other hosts: an extracted corpus may be
// copied to Windows later. Do not trim/repair names or expand historical aliases.
internal static class Wc3PathSafety
{
    internal static string VirtualKey(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string normalized = path.Replace('/', '\\');
        if (normalized.StartsWith('\\') || Path.IsPathRooted(path) || normalized.Contains(':'))
            throw new ArgumentException("Select a relative file name, not a rooted, drive, UNC, or alternate-stream path.", nameof(path));

        foreach (string part in normalized.Split('\\'))
        {
            if (part.Length == 0 || part is "." or ".." || part.EndsWith('.') || part.EndsWith(' ') ||
                part.Any(static c => c < 32 || c is '<' or '>' or '"' or '|' or '?' or '*'))
                throw new ArgumentException($"Unsafe file path '{path}': empty, traversal, or non-portable component.", nameof(path));

            string stem = part.Split('.')[0];
            if (stem.Equals("CON", StringComparison.OrdinalIgnoreCase) ||
                stem.Equals("PRN", StringComparison.OrdinalIgnoreCase) ||
                stem.Equals("AUX", StringComparison.OrdinalIgnoreCase) ||
                stem.Equals("NUL", StringComparison.OrdinalIgnoreCase) ||
                stem.Equals("CONIN$", StringComparison.OrdinalIgnoreCase) ||
                stem.Equals("CONOUT$", StringComparison.OrdinalIgnoreCase) ||
                (stem.Length == 4 && (stem.StartsWith("COM", StringComparison.OrdinalIgnoreCase) ||
                                     stem.StartsWith("LPT", StringComparison.OrdinalIgnoreCase)) &&
                 (stem[3] is >= '1' and <= '9' or '\u00b9' or '\u00b2' or '\u00b3')))
                throw new ArgumentException($"Unsafe Windows device name in '{path}'.", nameof(path));
        }

        return normalized;
    }

    internal static string FullPath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));
        // A colon is only valid in a Windows drive prefix, never in a component.
        string relative = full[(Path.GetPathRoot(full)?.Length ?? 0)..];
        if (relative.Contains(':'))
            throw new ArgumentException("Filesystem paths cannot contain alternate data streams.", nameof(path));
        return full;
    }

    internal static bool Contains(string root, string path)
    {
        return path.Equals(root, StringComparison.OrdinalIgnoreCase) ||
            path.StartsWith(Path.EndsInDirectorySeparator(root) ? root : root + Path.DirectorySeparatorChar,
                StringComparison.OrdinalIgnoreCase);
    }

    internal static string Combine(string root, string key)
    {
        string path = FullPath(Path.Combine(root, key.Replace('\\', Path.DirectorySeparatorChar)));
        if (!Contains(root, path) || path.Equals(root, StringComparison.OrdinalIgnoreCase))
            throw new IOException("The selected file would escape its root.");
        return path;
    }

    internal static FileAttributes? Attributes(string path)
    {
        try { return File.GetAttributes(path); }
        catch (FileNotFoundException) { return null; }
        catch (DirectoryNotFoundException) { return null; }
    }

    internal static void CheckAncestors(string path)
    {
        for (string? current = path; current is not null; current = Path.GetDirectoryName(current))
        {
            FileAttributes? attributes = Attributes(current);
            if (attributes.HasValue && (attributes.Value & FileAttributes.ReparsePoint) != 0)
                throw new IOException($"Refusing reparse-point/symbolic-link path '{current}'. Select a physical directory/file instead.");
        }
    }

    internal static void CreateDirectories(string path, Wc3PathGuard guard)
    {
        CheckAncestors(path);
        Stack<string> missing = new();
        for (string? current = path; current is not null && !Directory.Exists(current); current = Path.GetDirectoryName(current))
        {
            if (Attributes(current).HasValue)
                throw new IOException($"Output directory component '{current}' is an existing file.");
            missing.Push(current);
        }
        while (missing.TryPop(out string? directory))
        {
            CheckAncestors(directory);
            // Refuse ambiguous destination trees on case-sensitive hosts too.
            if (!Directory.Exists(directory)) RejectCaseCollision(directory);
            Directory.CreateDirectory(directory);
            guard.Add(directory);
        }
    }

    internal static void RejectCaseCollision(string path)
    {
        string parent = Path.GetDirectoryName(path)!;
        if (!Directory.Exists(parent))
            return;
        string name = Path.GetFileName(path);
        if (Directory.EnumerateFileSystemEntries(parent).Any(entry =>
            Path.GetFileName(entry).Equals(name, StringComparison.OrdinalIgnoreCase)))
            throw new IOException($"Output '{path}' already exists (including case-insensitive collisions). Choose another output root; overwrite is not allowed.");
    }
}
