namespace WoWViewer;

/// <summary>Reports completion only after the action returns a positive, verified count.</summary>
internal static class FileOperationReporter
{
    internal static bool Run(string operation, string path, Func<int> action, Action<string> report)
    {
        try
        {
            string fullPath = Path.GetFullPath(path);
            int count = action();
            if (count <= 0)
                throw new InvalidOperationException("No artifacts were written or applied.");
            report($"{operation}: {count} artifact(s) — {fullPath}");
            return true;
        }
        catch (Exception ex)
        {
            report($"{operation} failed — {path}: {ex.Message} Partial output/application may remain.");
            return false;
        }
    }

    internal static void RequireWrittenFile(string path)
    {
        if (!File.Exists(path) || new FileInfo(path).Length == 0)
            throw new IOException($"No non-empty output file was written: {path}");
    }
}
