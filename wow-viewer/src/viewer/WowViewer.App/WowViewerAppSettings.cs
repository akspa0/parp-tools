using System.Text.Json;

namespace WowViewer.App;

internal sealed class WowViewerAppSettings
{
    public bool UseArchiveSource { get; set; } = true;

    public string ArchiveRoot { get; set; } = string.Empty;

    public string VirtualPath { get; set; } = string.Empty;

    public string InputPath { get; set; } = string.Empty;

    public int ProfileIndex { get; set; }

    public int SequenceIndex { get; set; }

    public int TimeMs { get; set; }

    public int VisualSize { get; set; } = 384;

    public bool ShowAboutWindow { get; set; } = true;

    public bool ShowControlWindow { get; set; } = true;

    public bool ShowDiagnosticsWindow { get; set; } = true;

    public bool ShowBoundaryWindow { get; set; } = true;
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

            settings.VisualSize = Math.Clamp(settings.VisualSize, 128, 1024);
            settings.ProfileIndex = Math.Clamp(settings.ProfileIndex, 0, 99);
            settings.SequenceIndex = Math.Max(0, settings.SequenceIndex);
            settings.TimeMs = Math.Max(0, settings.TimeMs);
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