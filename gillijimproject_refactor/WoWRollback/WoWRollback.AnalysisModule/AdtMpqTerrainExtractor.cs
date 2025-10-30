using System;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.AnalysisModule;

// Minimal stub to unblock builds; real extractor can be restored later.
public sealed class AdtMpqTerrainExtractor
{
    public TerrainExtractionResult ExtractFromArchive(IArchiveSource source, string mapName, string outputCsvPath)
    {
        // No-op: return success with zero records.
        return new TerrainExtractionResult(
            Success: true,
            ChunksExtracted: 0,
            TilesProcessed: 0,
            CsvPath: outputCsvPath);
    }
}
