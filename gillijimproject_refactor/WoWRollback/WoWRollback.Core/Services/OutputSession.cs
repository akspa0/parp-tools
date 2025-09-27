namespace WoWRollback.Core.Services;

public static class OutputSession
{
    public static string Create(string? root, string map)
    {
        var rootDir = string.IsNullOrWhiteSpace(root)
            ? Path.Combine(Directory.GetCurrentDirectory(), "rollback_outputs")
            : root!;
        Directory.CreateDirectory(rootDir);
        var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var session = Path.Combine(rootDir, $"session_{stamp}", map);
        Directory.CreateDirectory(session);
        return session;
    }
}
