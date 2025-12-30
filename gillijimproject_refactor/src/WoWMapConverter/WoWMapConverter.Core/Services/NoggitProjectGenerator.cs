using System;
using System.IO;
using System.Text.Json;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// Generates Noggit Red project files for converted map data.
/// </summary>
public static class NoggitProjectGenerator
{
    /// <summary>
    /// Generate a Noggit project file in the export directory.
    /// </summary>
    /// <param name="exportDir">Root export directory containing World/Maps/...</param>
    /// <param name="projectName">Name for the project (e.g., map name)</param>
    /// <param name="clientPath">Path to LK client installation</param>
    public static void Generate(string exportDir, string projectName, string clientPath)
    {
        var payload = new NoggitProjectRoot(
            new NoggitProject(
                Bookmarks: Array.Empty<object>(),
                Client: new NoggitClient(clientPath, "Wrath Of The Lich King"),
                PinnedMaps: Array.Empty<object>(),
                ProjectName: projectName
            )
        );

        var options = new JsonSerializerOptions { WriteIndented = true };
        var json = JsonSerializer.Serialize(payload, options);
        
        Directory.CreateDirectory(exportDir);
        var outputPath = Path.Combine(exportDir, "project.noggitproj");
        File.WriteAllText(outputPath, json);
        
        Console.WriteLine($"[Noggit] Generated project: {outputPath}");
    }

    private sealed record NoggitProjectRoot(NoggitProject Project);
    
    private sealed record NoggitProject(
        object[] Bookmarks,
        NoggitClient Client,
        object[] PinnedMaps,
        string ProjectName
    );
    
    private sealed record NoggitClient(string ClientPath, string ClientVersion);
}
