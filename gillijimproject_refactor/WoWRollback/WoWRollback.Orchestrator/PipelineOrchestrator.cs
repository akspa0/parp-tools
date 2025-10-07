using System;

namespace WoWRollback.Orchestrator;

internal sealed class PipelineOrchestrator
{
    public PipelineRunResult Run(PipelineOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        var session = SessionManager.CreateSession(options);

        var dbcRunner = new DbcStageRunner();
        var dbcResult = dbcRunner.Run(session);

        var adtRunner = new AdtStageRunner();
        var adtResults = adtRunner.Run(session);

        var viewerRunner = new ViewerStageRunner();
        var viewerResult = viewerRunner.Run(session, adtResults);

        var manifestWriter = new ManifestWriter();
        var pipelineResult = new PipelineRunResult(
            dbcResult.Success && viewerResult.Success && AllSucceeded(adtResults),
            session,
            dbcResult,
            adtResults,
            viewerResult);
        manifestWriter.Write(session, pipelineResult);

        return pipelineResult;
    }

    private static bool AllSucceeded(IReadOnlyList<AdtStageResult> results)
    {
        foreach (var result in results)
        {
            if (!result.Success)
            {
                return false;
            }
        }

        return true;
    }
}

internal sealed record PipelineRunResult(
    bool Success,
    SessionContext Session,
    DbcStageResult Dbc,
    IReadOnlyList<AdtStageResult> AdtResults,
    ViewerStageResult Viewer);
