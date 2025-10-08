using System;
using System.IO;
using WoWRollback.Core.Services.Config;

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
        var sessionRoot = Path.Combine(outputRoot, $"session_{timestamp}");

        // Per spec: numbered directories inside session (01_, 02_, etc)
        var dbcDir = Path.Combine(sessionRoot, "01_dbcs");
        var crosswalkDir = Path.Combine(sessionRoot, "02_crosswalks");
        var adtDir = Path.Combine(sessionRoot, "03_adts");
        var analysisDir = Path.Combine(sessionRoot, "04_analysis");
        var viewerDir = Path.Combine(sessionRoot, "05_viewer");
        var logsDir = Path.Combine(sessionRoot, "logs");
        var manifestPath = Path.Combine(sessionRoot, "manifest.json");

        // Create all directories
        Directory.CreateDirectory(sessionRoot);
        Directory.CreateDirectory(dbcDir);
        Directory.CreateDirectory(crosswalkDir);
        Directory.CreateDirectory(adtDir);
        Directory.CreateDirectory(analysisDir);
        Directory.CreateDirectory(viewerDir);
        Directory.CreateDirectory(logsDir);

        var paths = new SessionPaths(
            Root: sessionRoot,
            DbcDir: dbcDir,
            CrosswalkDir: crosswalkDir,
            AdtDir: adtDir,
            AnalysisDir: analysisDir,
            ViewerDir: viewerDir,
            LogsDir: logsDir,
            ManifestPath: manifestPath);

        var overrideResolver = AreaOverrideLoader.LoadFromDirectory(options.AreaOverrideDirectory);

        return new SessionContext(timestamp, paths, options, overrideResolver);
    }
}

internal sealed record SessionContext(
    string SessionId,
    SessionPaths Paths,
    PipelineOptions Options,
    AreaOverrideLoader.AreaOverrideResolver? OverrideResolver)
{
    public string Root => Paths.Root;
    public string ManifestPath => Paths.ManifestPath;
}

internal sealed record SessionPaths(
    string Root,
    string DbcDir,
    string CrosswalkDir,
    string AdtDir,
    string AnalysisDir,
    string ViewerDir,
    string LogsDir,
    string ManifestPath);
