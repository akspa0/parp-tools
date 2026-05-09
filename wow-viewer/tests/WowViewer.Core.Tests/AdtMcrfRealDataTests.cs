using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMcrfRealDataTests
{
    [Fact]
    public void Build_StagedWrathRootAdt_PreservesRealDataMcrf_WhenAvailable()
    {
        if (!File.Exists(AdtRealDataTestCatalog.ListfilePath))
            return;

        foreach (StagedClientSurface client in AdtRealDataTestCatalog.GetStagedClients())
        {
            using MpqArchiveCatalog catalog = AdtRealDataTestCatalog.CreateCatalog(client);
            IReadOnlyList<string> candidates = AdtRealDataTestCatalog.GetCandidateRootAdts(catalog);
            foreach (string virtualPath in candidates)
            {
                byte[]? bytes = catalog.ReadFile(virtualPath);
                if (bytes is null)
                    continue;

                if (!TryFindMcrf(bytes, out bool hasTrailingSubchunk, out string trailingChunkId))
                    continue;

                string tempPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}_{Path.GetFileName(virtualPath)}");
                try
                {
                    File.WriteAllBytes(tempPath, bytes);
                    TerrainTileTensorPack pack;
                    try
                    {
                        pack = AdtTensorPackBuilder.Build(tempPath, buildVersion: client.BuildVersion);
                    }
                    catch (Exception ex) when (ex is InvalidDataException or OverflowException or EndOfStreamException or ArgumentOutOfRangeException)
                    {
                        continue;
                    }

                    if ((pack.McrfDoodadRefIndices?.Length ?? 0) + (pack.McrfWmoRefIndices?.Length ?? 0) == 0)
                        continue;

                    Assert.NotNull(pack.McrfDoodadRefCounts16);
                    Assert.NotNull(pack.McrfDoodadRefIndices);
                    Assert.NotNull(pack.McrfWmoRefCounts16);
                    Assert.NotNull(pack.McrfWmoRefIndices);
                    Assert.True(pack.McrfDoodadRefIndices!.Length + pack.McrfWmoRefIndices!.Length > 0, $"Expected real MCRF refs in {virtualPath} ({client.BuildVersion}).");
                    Assert.Contains("mcrf_doodad_ref_indices", pack.AvailableSignals);
                    Assert.DoesNotContain(pack.RawChunks, static chunk => chunk.ChunkId == "MCRF");
                    if (hasTrailingSubchunk)
                        Assert.False(string.Equals(trailingChunkId, "MCRF", StringComparison.OrdinalIgnoreCase));
                    return;
                }
                finally
                {
                    if (File.Exists(tempPath))
                        File.Delete(tempPath);
                }
            }
        }

        return;
    }

    private static bool TryFindMcrf(byte[] adtBytes, out bool hasTrailingSubchunk, out string trailingChunkId)
    {
        hasTrailingSubchunk = false;
        trailingChunkId = string.Empty;

        string tempPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}_mcrf_scan.adt");
        try
        {
            File.WriteAllBytes(tempPath, adtBytes);

            using FileStream stream = File.OpenRead(tempPath);
            MapFileSummary summary = MapFileSummaryReader.Read(stream, tempPath);
            foreach (MapChunkLocation chunk in summary.Chunks)
            {
                if (chunk.Id != MapChunkIds.Mcnk)
                    continue;

                byte[] payload = ReadPayload(stream, chunk);
                if (!AdtMcrfReader.TryLocateMcrfPayload(payload, out int mcrfOffset, out int mcrfSize))
                    continue;

                int trailingHeaderOffset = mcrfOffset + mcrfSize;
                if (trailingHeaderOffset <= payload.Length - ChunkHeader.SizeInBytes &&
                    ChunkHeaderReader.TryRead(payload.AsSpan(trailingHeaderOffset, ChunkHeader.SizeInBytes), out ChunkHeader trailingHeader))
                {
                    hasTrailingSubchunk = true;
                    trailingChunkId = trailingHeader.Id.ToString();
                }

                return true;
            }

            return false;
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    private static byte[] ReadPayload(Stream stream, MapChunkLocation chunk)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }
}
