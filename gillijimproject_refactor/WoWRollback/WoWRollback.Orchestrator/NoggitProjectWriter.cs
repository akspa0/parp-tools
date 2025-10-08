using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace WoWRollback.Orchestrator;

internal static class NoggitProjectWriter
{
    private sealed record NoggitProjectRoot(NoggitProject Project);

    private sealed record NoggitProject(
        IReadOnlyList<object> Bookmarks,
        NoggitClient Client,
        IReadOnlyList<object> PinnedMaps,
        string ProjectName);

    private sealed record NoggitClient(string ClientPath, string ClientVersion);

    public static void Write(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }
        if (adtResults is null || adtResults.Count == 0)
        {
            return;
        }

        var successfulVersions = adtResults
            .Where(r => r.Success)
            .Select(r => r.Version)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (successfulVersions.Length == 0)
        {
            return;
        }

        var clientPath = !string.IsNullOrWhiteSpace(session.Options.NoggitClientPath)
            ? session.Options.NoggitClientPath!
            : "<set this in cli!>";

        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        foreach (var version in successfulVersions)
        {
            var versionDir = Path.Combine(session.Paths.AdtDir, version);
            if (!Directory.Exists(versionDir))
            {
                continue;
            }

            var payload = new NoggitProjectRoot(
                new NoggitProject(
                    Bookmarks: Array.Empty<object>(),
                    Client: new NoggitClient(clientPath, "Wrath Of The Lich King"),
                    PinnedMaps: Array.Empty<object>(),
                    ProjectName: session.SessionId));

            var json = JsonSerializer.Serialize(payload, options);
            var outputPath = Path.Combine(versionDir, "project.noggitproject");
            File.WriteAllText(outputPath, json);
        }
    }
}
