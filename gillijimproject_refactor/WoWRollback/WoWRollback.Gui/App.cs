using System;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;

namespace WoWRollback.Gui;

public partial class App : Application
{
    public static string[] LaunchArgs { get; set; } = Array.Empty<string>();

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var (cache, presets) = ParseArgs(LaunchArgs);
            desktop.MainWindow = new MainWindow(cache, presets);
        }

        base.OnFrameworkInitializationCompleted();
    }

    private static (string CacheDir, string PresetsDir) ParseArgs(string[] args)
    {
        string cache = System.IO.Path.Combine(Environment.CurrentDirectory, "work", "cache");
        string presets = System.IO.Path.Combine(Environment.CurrentDirectory, "work", "presets");
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a == "--cache" && i + 1 < args.Length) cache = args[++i];
            else if (a == "--presets" && i + 1 < args.Length) presets = args[++i];
        }
        cache = System.IO.Path.GetFullPath(cache);
        presets = System.IO.Path.GetFullPath(presets);
        return (cache, presets);
    }
}
