using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>Opt-in strict evidence wrapper above the unchanged placement reader.</summary>
public static class AdtPlacementAuditReader
{
    public static AdtPlacementCatalog Read(byte[] bytes, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        using var stream = new MemoryStream(bytes, writable: false);
        var summary = MapFileSummaryReader.Read(stream, sourcePath);
        if (summary.Chunks.Count == 0 || summary.Chunks.Max(c => c.EndOffset) != bytes.Length)
            throw new InvalidDataException("Incomplete ADT top-level chunk envelope.");
        foreach (var chunk in summary.Chunks)
        {
            if (chunk.EndOffset > bytes.Length) throw new InvalidDataException("ADT chunk exceeds captured bytes.");
            int stride = chunk.Id == MapChunkIds.Mddf ? 36 : chunk.Id == MapChunkIds.Modf ? 64
                : chunk.Id == MapChunkIds.Mmid || chunk.Id == MapChunkIds.Mwid ? 4 : 1;
            if (chunk.Size % stride != 0)
                throw new InvalidDataException($"Placement/index chunk {chunk.Id} has a partial {stride}-byte record.");
        }
        var catalog = AdtPlacementReader.Read(stream, summary);
        if (catalog.ModelPlacements.Any(p => string.IsNullOrWhiteSpace(p.ModelPath) || p.ModelPath.StartsWith("unknown_", StringComparison.Ordinal))
            || catalog.WorldModelPlacements.Any(p => string.IsNullOrWhiteSpace(p.ModelPath) || p.ModelPath.StartsWith("unknown_", StringComparison.Ordinal)))
            throw new InvalidDataException("Authored placement names are unresolved; retain input evidence, do not assert asset-family identities.");
        return catalog;
    }
}
