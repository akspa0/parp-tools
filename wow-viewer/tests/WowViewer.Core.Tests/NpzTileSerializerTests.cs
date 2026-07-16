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
                MinimapLightingProvenance = new MinimapLightingProvenance(
                    MinimapLightingProvenance.CurrentContractVersion,
                    "baked_tint_likely",
                    64,
                    0.8f,
                    1f,
                    0.6f,
                    0.24f,
                    0.9f,
                    null,
                    12f,
                    0.8f,
                    "inferred_global_lighting_chroma_match_not_capture_proof",
                    "fixture"),
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
                MinimapTextureFallbacks = new Dictionary<int, TerrainTextureFallbackResolution>
                {
                    [5] = new TerrainTextureFallbackResolution(
                        5,
                        "Tileset/Durotar/DurotarIGrass.blp",
                        "Tileset/Durotar/DurotarIGrass_s.blp",
                        TerrainTextureFallbackPolicy.SpecularCompanionRgbProxy),
                },
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
            Assert.Contains("\"minimap_lighting\"", metadata, StringComparison.Ordinal);
            Assert.Contains("\"estimated_time_of_day_hours\": 12", metadata, StringComparison.Ordinal);
            Assert.Contains("\"resolved_path\": \"Tileset/Durotar/DurotarIGrass_s.blp\"", metadata, StringComparison.Ordinal);
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
    public void Serialize_DoesNotShiftAnIncompleteTexturePayloadTable()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_texture_alignment_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                MclyTextureNames = ["Tileset/First.blp", "Tileset/Second.blp"],
                MclyTexturePixels = [new byte[2, 2, 3]],
            };

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            Assert.DoesNotContain(archive.Entries, static entry => entry.FullName == "mcly_texture_pixels_0.npy");
            ZipArchiveEntry metadataEntry = Assert.Single(archive.Entries, static entry => entry.FullName == "metadata.json");
            using StreamReader reader = new(metadataEntry.Open(), Encoding.UTF8);
            Assert.Contains("\"mcly_texture_payload_state\": \"incomplete_not_serialized\"", reader.ReadToEnd(), StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_WritesStrictGeometryTargetArraysAndCompletenessMetadata()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_strict_object_{Guid.NewGuid():N}");
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
                ObjectGeometryVisibleMask257 = new float[257, 257],
                ObjectGeometryVisibleTopElevation257 = new float[257, 257],
                ObjectGeometryVisibleTerrainElevation257 = new float[257, 257],
                ObjectGeometryVisibleSource257 = new byte[257, 257],
                ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                    ObjectGeometryTargetStatus.CompleteEmpty,
                    PlacementCount: 0,
                    GeometryResolvedPlacementCount: 0,
                    GeometryUnresolvedPlacementCount: 0,
                    FallbackRequiredPlacementCount: 0,
                    TriangleCount: 0,
                    VisiblePixelCount: 0,
                    OccludedPixelCount: 0,
                    TerrainUnknownPixelCount: 0,
                    LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
                ObjectGeometryFragmentTrace = ObjectGeometryFragmentTrace.Create(
                    Array.Empty<ObjectGeometryFragmentRecord>(),
                    Array.Empty<ObjectGeometryTargetAsset>()),
            };

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_visible_mask_257.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_visible_top_elevation_257.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_visible_terrain_elevation_257.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_visible_source_257.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_raster_xy.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_world_xyze.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_source_ids.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_source_classification.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_terrain_vertex_dense_xy.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_terrain_vertex_z.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_terrain_vertex_present.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_terrain_weights.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_terrain_liquid_elevation.npy");
            Assert.DoesNotContain(archive.Entries, static entry => entry.FullName == "object_precise_mask_257.npy");

            ZipArchiveEntry metadataEntry = Assert.Single(archive.Entries, static entry => entry.FullName == "metadata.json");
            using StreamReader reader = new(metadataEntry.Open(), Encoding.UTF8);
            string metadata = reader.ReadToEnd();
            Assert.Contains($"\"object_geometry_target_version\": \"{ObjectGeometryTargetProvenance.ContractVersion}\"", metadata, StringComparison.Ordinal);
            Assert.Contains("\"object_geometry_target_status\": \"CompleteEmpty\"", metadata, StringComparison.Ordinal);
            Assert.Contains("\"object_geometry_target_materialized\": true", metadata, StringComparison.Ordinal);
            Assert.Contains("\"object_geometry_fragment_count\": 0", metadata, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_RejectsIncompleteStrictProvenanceWithUnionArraysBeforeCreatingOutput()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_strict_reject_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                ObjectGeometryVisibleMask257 = new float[257, 257],
                ObjectGeometryVisibleTopElevation257 = new float[257, 257],
                ObjectGeometryVisibleTerrainElevation257 = new float[257, 257],
                ObjectGeometryVisibleSource257 = new byte[257, 257],
                ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                    ObjectGeometryTargetStatus.IncompleteGeometry,
                    PlacementCount: 1,
                    GeometryResolvedPlacementCount: 0,
                    GeometryUnresolvedPlacementCount: 1,
                    FallbackRequiredPlacementCount: 1,
                    TriangleCount: 0,
                    VisiblePixelCount: 0,
                    OccludedPixelCount: 0,
                    TerrainUnknownPixelCount: 0,
                    LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
            };

            InvalidDataException failure = Assert.Throws<InvalidDataException>(
                () => NpzTileSerializer.Serialize(pack, outputPath));

            Assert.Contains("nonmaterialized", failure.Message, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(outputPath));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_IncompleteStrictGeometryRetainsValidFragmentTraceWithoutUnionArrays()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_incomplete_trace_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            TerrainTileTensorPack pack = CreateIncompleteTracePack();

            NpzTileSerializer.Serialize(pack, outputPath);

            using ZipArchive archive = ZipFile.OpenRead(outputPath);
            Assert.DoesNotContain(archive.Entries, static entry => entry.FullName == "object_geometry_visible_mask_257.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_raster_xy.npy");
            Assert.Contains(archive.Entries, static entry => entry.FullName == "object_geometry_fragment_world_xyze.npy");

            ZipArchiveEntry metadataEntry = Assert.Single(archive.Entries, static entry => entry.FullName == "metadata.json");
            using StreamReader reader = new(metadataEntry.Open(), Encoding.UTF8);
            string metadata = reader.ReadToEnd();
            Assert.Contains("\"object_geometry_target_materialized\": false", metadata, StringComparison.Ordinal);
            Assert.Contains($"\"object_geometry_fragment_trace_schema\": \"{ObjectGeometryTargetProvenance.ContractVersion}\"", metadata, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_RejectsMutatedIncompleteFragmentTraceBeforeCreatingOutput()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_mutated_trace_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            TerrainTileTensorPack pack = CreateIncompleteTracePack();
            pack.ObjectGeometryFragmentTrace!.TerrainWeights[0, 0] += 0.1f;

            InvalidDataException failure = Assert.Throws<InvalidDataException>(
                () => NpzTileSerializer.Serialize(pack, outputPath));

            Assert.Contains("does not match", failure.Message, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(outputPath));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Serialize_RejectsMaterializedStrictUnionWithoutFragmentTrace()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_npz_strict_trace_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string outputPath = Path.Combine(tempDir, "tile.npz");
            TerrainTileTensorPack pack = new()
            {
                TileName = "tile_0_0",
                ObjectGeometryVisibleMask257 = new float[257, 257],
                ObjectGeometryVisibleTopElevation257 = new float[257, 257],
                ObjectGeometryVisibleTerrainElevation257 = new float[257, 257],
                ObjectGeometryVisibleSource257 = new byte[257, 257],
                ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                    ObjectGeometryTargetStatus.CompleteEmpty,
                    PlacementCount: 0,
                    GeometryResolvedPlacementCount: 0,
                    GeometryUnresolvedPlacementCount: 0,
                    FallbackRequiredPlacementCount: 0,
                    TriangleCount: 0,
                    VisiblePixelCount: 0,
                    OccludedPixelCount: 0,
                    TerrainUnknownPixelCount: 0,
                    LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
            };

            InvalidDataException failure = Assert.Throws<InvalidDataException>(
                () => NpzTileSerializer.Serialize(pack, outputPath));

            Assert.Contains("fragment trace", failure.Message, StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(outputPath));
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

    private static TerrainTileTensorPack CreateIncompleteTracePack()
    {
        ObjectGeometryTargetAsset[] assets =
        [
            new ObjectGeometryTargetAsset(
                AssetIndex: 0,
                Source: ObjectGeometryPixelSource.WmoTriangle,
                NormalizedAssetPath: "world/wmo/test.wmo"),
        ];
        ObjectGeometryFragmentTrace trace = ObjectGeometryFragmentTrace.Create(
        [
            new ObjectGeometryFragmentRecord(
                RasterX: 5,
                RasterY: 6,
                ObjectWorldX: 100f,
                ObjectWorldY: 200f,
                ObjectWorldZ: 25f,
                ObjectElevation: 25f,
                PlacementUniqueId: 72,
                AssetIndex: 0,
                SourceTriangleIndex: 4,
                Source: ObjectGeometryPixelSource.WmoTriangle,
                Classification: ObjectGeometryFragmentClassification.TerrainHidden,
                TerrainVertex0X: 4,
                TerrainVertex0Y: 6,
                TerrainVertex1X: 6,
                TerrainVertex1Y: 6,
                TerrainVertex2X: 5,
                TerrainVertex2Y: 7,
                TerrainVertex0Z: 26f,
                TerrainVertex1Z: 26f,
                TerrainVertex2Z: 26f,
                TerrainVertex0Present: true,
                TerrainVertex1Present: true,
                TerrainVertex2Present: true,
                TerrainWeight0: 0.25f,
                TerrainWeight1: 0.25f,
                TerrainWeight2: 0.5f,
                TerrainElevation: 26f,
                LiquidSurfaceElevation: float.NaN),
        ],
            assets);
        return new TerrainTileTensorPack
        {
            TileName = "tile_0_0",
            ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                ObjectGeometryTargetStatus.IncompleteGeometry,
                PlacementCount: 1,
                GeometryResolvedPlacementCount: 0,
                GeometryUnresolvedPlacementCount: 1,
                FallbackRequiredPlacementCount: 1,
                TriangleCount: 1,
                VisiblePixelCount: 0,
                OccludedPixelCount: 1,
                TerrainUnknownPixelCount: 0,
                LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
            ObjectGeometryTargetAssets = assets,
            ObjectGeometryFragmentTrace = trace,
        };
    }
}
