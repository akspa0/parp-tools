using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMcseRealDataTests
{
    [Fact]
    public void Build_StagedRootAdt_PreservesRealDataMcse_WhenAvailable()
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

                    if (pack.McseEntryBytes is null || pack.McseEntryBytes.GetLength(0) == 0)
                        continue;

                    int entryCount = pack.McseEntryBytes.GetLength(0);
                    int entrySize = pack.McseEntryBytes.GetLength(1);
                    Assert.NotNull(pack.McseEmitterCounts16);
                    Assert.True(pack.McseEmitterCounts16!.Cast<ushort>().Sum(static count => count) >= entryCount, $"Expected MCSE counts to cover {entryCount} emitted entries in {virtualPath} ({client.BuildVersion}).");
                    Assert.Contains("mcse_entry_bytes", pack.AvailableSignals);
                    Assert.Contains(pack.RawChunks, static chunk => chunk.ChunkId == "MCSE");

                    if (entrySize == AdtMcseReader.StandardEntrySize)
                    {
                        Assert.NotNull(pack.McseEntryIds);
                        Assert.NotNull(pack.McsePositionXyz);
                        Assert.Equal(entryCount, pack.McseEntryIds!.Length);
                        Assert.Equal(entryCount, pack.McsePositionXyz!.GetLength(0));
                        Assert.Equal(3, pack.McsePositionXyz.GetLength(1));
                        Assert.Contains("mcse_entry_ids", pack.AvailableSignals);
                        Assert.Contains("mcse_position_xyz", pack.AvailableSignals);
                    }
                    else
                    {
                        Assert.Null(pack.McseEntryIds);
                        Assert.Null(pack.McsePositionXyz);
                    }

                    return;
                }
                finally
                {
                    if (File.Exists(tempPath))
                        File.Delete(tempPath);
                }
            }
        }
    }
}