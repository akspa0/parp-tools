using System.IO.Compression;
using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class NpzTileSerializerTests
{
    [Fact]
    public void Serialize_WritesRawChunkBlobEntriesAndMetadata()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_raw_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                MapName = "TestMap",
                BuildKey = "3.3.5.12340",
                SourceAdtPath = "tile_0_0.adt",
                AvailableSignals = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "raw_adt_chunks" },
                RawChunks =
                [
                    new TerrainRawChunkBlob
                    {
                        EntryName = "raw_chunks/root/top/MFBO_000",
                        SourceKind = "root",
                        SourcePath = "tile_0_0.adt",
                        Scope = "top-level",
                        ChunkId = "MFBO",
                        Data = [1, 2, 3, 4],
                    },
                ],
            };

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            ZipArchiveEntry rawEntry = Assert.Single(archive.Entries, static entry => entry.FullName == "raw_chunks/root/top/MFBO_000.npy");
            Assert.True(rawEntry.Length > 0);

            ZipArchiveEntry metadataEntry = Assert.Single(archive.Entries, static entry => entry.FullName == "metadata.json");
            using StreamReader reader = new(metadataEntry.Open(), Encoding.UTF8);
            string metadata = reader.ReadToEnd();

            Assert.Contains("\"raw_chunks\"", metadata, StringComparison.Ordinal);
            Assert.Contains("\"chunk_id\": \"MFBO\"", metadata, StringComparison.Ordinal);
            Assert.Contains("\"entry_name\": \"raw_chunks/root/top/MFBO_000\"", metadata, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_WritesDecodedPreservationSignals()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_decoded_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            byte[,,] mcmt = new byte[16, 16, 4];
            mcmt[0, 0, 0] = 7;

            byte[,,] mclv = new byte[257, 257, 4];
            mclv[0, 0, 0] = 1;
            mclv[0, 0, 1] = 2;
            mclv[0, 0, 2] = 3;
            mclv[0, 0, 3] = 4;

            int[,,] mfbo = new int[2, 3, 3];
            mfbo[0, 0, 0] = 123;
            mfbo[1, 2, 2] = -45;

            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                MapName = "TestMap",
                BuildKey = "4.0.0.11927",
                SourceAdtPath = "tile_0_0.adt",
                AvailableSignals = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "mcmt_material_ids",
                    "mamp_value",
                    "mclv_lighting_bytes",
                    "mfbo_flight_bounds",
                },
                McmtMaterialIds = mcmt,
                MampValue = [5],
                MclvLightingBytes = mclv,
                MfboFlightBounds = mfbo,
            };

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcmt_material_ids.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mamp_value.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mclv_lighting_bytes.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mfbo_flight_bounds.npy");
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_WritesSplitPlacementReferenceSignals()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_refs_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            int[,] mcrdCounts = new int[16, 16];
            int[,] mcrwCounts = new int[16, 16];
            mcrdCounts[0, 0] = 2;
            mcrwCounts[0, 0] = 1;

            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                MapName = "TestMap",
                BuildKey = "4.0.0.11927",
                SourceAdtPath = "tile_0_0.adt",
                AvailableSignals = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "mcrd_ref_indices",
                    "mcrw_ref_indices",
                },
                McrdRefCounts16 = mcrdCounts,
                McrdRefIndices = [11, 12],
                McrwRefCounts16 = mcrwCounts,
                McrwRefIndices = [21],
            };

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrd_ref_counts_16.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrd_ref_indices.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrw_ref_counts_16.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrw_ref_indices.npy");
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_WritesPreCataMcrfReferenceSignals()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_mcrf_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            int[,] doodadCounts = new int[16, 16];
            int[,] wmoCounts = new int[16, 16];
            doodadCounts[0, 0] = 2;
            wmoCounts[0, 0] = 1;

            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                MapName = "TestMap",
                BuildKey = "3.3.5.12340",
                SourceAdtPath = "tile_0_0.adt",
                AvailableSignals = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "mcrf_doodad_ref_indices",
                    "mcrf_wmo_ref_indices",
                },
                McrfDoodadRefCounts16 = doodadCounts,
                McrfDoodadRefIndices = [11, 12],
                McrfWmoRefCounts16 = wmoCounts,
                McrfWmoRefIndices = [21],
            };

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrf_doodad_ref_counts_16.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrf_doodad_ref_indices.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrf_wmo_ref_counts_16.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcrf_wmo_ref_indices.npy");
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_WritesMcseSignals()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_mcse_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            int[,] counts = new int[16, 16];
            counts[0, 0] = 2;
            byte[,] entryBytes = new byte[2, AdtMcseReader.StandardEntrySize];
            entryBytes[0, 0] = 1;
            entryBytes[1, 0] = 2;

            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                MapName = "TestMap",
                BuildKey = "3.3.5.12340",
                SourceAdtPath = "tile_0_0.adt",
                AvailableSignals = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "mcse_entry_bytes",
                    "mcse_entry_ids",
                    "mcse_position_xyz",
                },
                McseEmitterCounts16 = counts,
                McseEntryIds = [1001, 1002],
                McsePositionXyz = new float[,] { { 1.5f, 2.5f, 3.5f }, { 4.5f, 5.5f, 6.5f } },
                McseEntryBytes = entryBytes,
            };

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcse_emitter_counts_16.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcse_entry_ids.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcse_position_xyz.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "mcse_entry_bytes.npy");
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }
}