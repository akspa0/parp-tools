using System;
using Avalonia;

namespace WoWRollback.Gui;

internal static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        App.LaunchArgs = args ?? Array.Empty<string>();
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(App.LaunchArgs);
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .LogToTrace();
}
