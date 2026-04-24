namespace WowViewer.Core.Runtime.World;

public static class WorldSkyboxBackdropClassifier
{
    public static bool IsBackdropModelPath(string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return false;

        string normalized = modelPath.Replace('\\', '/').ToLowerInvariant();
        if (!normalized.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)
            && !normalized.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (normalized.Contains("skylight"))
            return false;

        return normalized.Contains("environments/stars/")
            || normalized.Contains("/skybox/")
            || normalized.Contains("skybox")
            || normalized.Contains("skybowl");
    }
}
