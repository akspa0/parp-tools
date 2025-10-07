using System;
using System.IO;

namespace WoWRollback.Orchestrator;

internal static class SessionManager
{
    public static SessionContext CreateSession(PipelineOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var outputRoot = Path.GetFullPath(options.OutputRoot);
        var sharedRoot = Path.Combine(outputRoot, "shared_outputs");
        var sharedDbcRoot = Path.Combine(sharedRoot, "dbc");
        var sharedCrosswalkRoot = Path.Combine(sharedRoot, "crosswalks");

        var sessionRoot = Path.Combine(outputRoot, $"session_{timestamp}");
        var sessionAdtDir = Path.Combine(sessionRoot, "adt");
        var sessionAnalysisDir = Path.Combine(sessionRoot, "analysis");
        var sessionViewerDir = Path.Combine(sessionRoot, "viewer");
        var sessionLogsDir = Path.Combine(sessionRoot, "logs");
        var sessionManifest = Path.Combine(sessionRoot, "manifest.json");

        Directory.CreateDirectory(sharedRoot);
        Directory.CreateDirectory(sharedDbcRoot);
        Directory.CreateDirectory(sharedCrosswalkRoot);
        Directory.CreateDirectory(sessionRoot);
        Directory.CreateDirectory(sessionAdtDir);
        Directory.CreateDirectory(sessionAnalysisDir);
        Directory.CreateDirectory(sessionViewerDir);
        Directory.CreateDirectory(sessionLogsDir);

        var paths = new SessionPaths(
            Root: sessionRoot,
            SharedDbcRoot: sharedDbcRoot,
            SharedCrosswalkRoot: sharedCrosswalkRoot,
            AdtDir: sessionAdtDir,
            AnalysisDir: sessionAnalysisDir,
            ViewerDir: sessionViewerDir,
            LogsDir: sessionLogsDir,
            ManifestPath: sessionManifest);

        return new SessionContext(timestamp, paths, options);
    }
}

internal sealed record SessionContext(
    string SessionId,
    SessionPaths Paths,
    PipelineOptions Options)
{
    public string Root => Paths.Root;
    public string ManifestPath => Paths.ManifestPath;
    public string SharedDbcRoot => Paths.SharedDbcRoot;
    public string SharedCrosswalkRoot => Paths.SharedCrosswalkRoot;
}

internal sealed record SessionPaths(
    string Root,
    string SharedDbcRoot,
    string SharedCrosswalkRoot,
    string AdtDir,
    string AnalysisDir,
    string ViewerDir,
    string LogsDir,
    string ManifestPath);
