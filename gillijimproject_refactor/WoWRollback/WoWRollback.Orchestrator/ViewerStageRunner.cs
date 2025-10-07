using System;
using System.IO;

namespace WoWRollback.Orchestrator;

internal sealed class ViewerStageRunner
{
    public ViewerStageResult Run(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }

        Directory.CreateDirectory(session.Paths.ViewerDir);

        // TODO: Copy viewer assets and generate overlays when available.

        return new ViewerStageResult(
            Success: true,
            ViewerDirectory: session.Paths.ViewerDir,
            OverlayCount: 0,
            Notes: "Viewer asset generation not yet implemented.");
    }
}

internal sealed record ViewerStageResult(
    bool Success,
    string ViewerDirectory,
    int OverlayCount,
    string? Notes);
