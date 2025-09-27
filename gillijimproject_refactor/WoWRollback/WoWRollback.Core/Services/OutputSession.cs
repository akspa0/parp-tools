namespace WoWRollback.Core.Services;

public static class OutputSession
{
    public static string Create(string? root, string map, string buildTag)
    {
        var rootDir = string.IsNullOrWhiteSpace(root)
            ? Path.Combine(Directory.GetCurrentDirectory(), "rollback_outputs")
            : root!;

        var versionSegment = string.IsNullOrWhiteSpace(buildTag) ? "unknown_build" : buildTag;
        var session = Path.Combine(rootDir, versionSegment, map);
        Directory.CreateDirectory(session);
        return session;
    }
}
