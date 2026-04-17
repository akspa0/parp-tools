using System.Text.Json;

namespace WowViewer.App;

internal sealed class WowViewerAppSettings
{
    public WowViewerSession Session { get; set; } = WowViewerSession.CreateDefault();

    public bool ShowAboutWindow { get; set; } = true;

    public bool ShowWorkspaceWindow { get; set; } = true;

    public bool ShowControlWindow { get; set; } = true;

    public bool ShowDiagnosticsWindow { get; set; } = true;

    public bool ShowBoundaryWindow { get; set; } = true;

    public bool ShowWorldStatusWindow { get; set; } = true;

    public bool ShowNavigatorWindow { get; set; } = true;

    public bool ShowInspectorWindow { get; set; } = true;
}

internal static class WowViewerAppSettingsStore
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    public static string SettingsDirectory => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "settings");

    public static string SettingsPath => Path.Combine(SettingsDirectory, "wowviewer_app_settings.json");

    public static WowViewerAppSettings Load()
    {
        try
        {
            if (!File.Exists(SettingsPath))
                return new WowViewerAppSettings();

            string json = File.ReadAllText(SettingsPath);
            WowViewerAppSettings? settings = JsonSerializer.Deserialize<WowViewerAppSettings>(json, JsonOptions);
            if (settings == null)
                return new WowViewerAppSettings();

            settings.Session ??= WowViewerSession.CreateDefault();
            settings.Session.Normalize();
            return settings;
        }
        catch
        {
            return new WowViewerAppSettings();
        }
    }

    public static void Save(WowViewerAppSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        Directory.CreateDirectory(SettingsDirectory);
        string json = JsonSerializer.Serialize(settings, JsonOptions);
        File.WriteAllText(SettingsPath, json);
    }
}