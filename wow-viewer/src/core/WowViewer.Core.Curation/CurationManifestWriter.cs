using Parquet;
using Parquet.Data;
using Parquet.Schema;

namespace WowViewer.Core.Curation;

/// <summary>
/// Writes the two-table Curation Manifest (data-model.md) alongside an existing v50 store:
/// <c>curation_manifest.parquet</c> (one row per tile), <c>curation_findings.parquet</c> (one row
/// per finding), and <c>curation_run.json</c> (the <see cref="CurationRunRecord"/> provenance).
/// This is the first C#-side Parquet <b>writer</b> in this codebase -- every existing Parquet
/// sidecar here was previously written Python-side; <see cref="Parquet.Net"/> was only used to
/// read (<c>V18StorePlacementsReader</c>). Strictly read-only with respect to the source store
/// (FR-014): the store's own <c>index.parquet</c> and Zarr arrays are never opened for writing.
/// </summary>
public static class CurationManifestWriter
{
    private static readonly ParquetSchema ManifestSchema = new(
        new DataField<string>("build"),
        new DataField<string>("map"),
        new DataField<int>("tile_x"),
        new DataField<int>("tile_y"),
        new DataField<long>("tile_id"),
        new DataField<string>("difficulty_bucket"),
        new DataField<string>("coverage_bucket"),
        new DataField<string>("lighting_bucket"),
        new DataField<string>("synthetic_fidelity_status"),
        new DataField<float?>("synthetic_fidelity_score"),
        new DataField<int>("finding_count"),
        new DataField<string>("curation_run_id"));

    private static readonly ParquetSchema FindingsSchema = new(
        new DataField<string>("build"),
        new DataField<string>("map"),
        new DataField<int>("tile_x"),
        new DataField<int>("tile_y"),
        new DataField<long>("tile_id"),
        new DataField<string>("category"),
        new DataField<string>("severity"),
        new DataField<string>("reason"),
        new DataField<string>("evaluability"),
        new DataField<string?>("signal"),
        new DataField<string>("curation_run_id"));

    /// <summary>
    /// Writes both Parquet tables plus the run record under
    /// <c>&lt;storeRoot&gt;/curation/&lt;runRecord.CurationRunId&gt;/</c> and updates the
    /// <c>&lt;storeRoot&gt;/curation/latest</c> pointer file. Verifies
    /// <paramref name="runRecord"/>'s tile count against <paramref name="records"/> before writing
    /// anything (SC-006) -- a partial write is a failure, not a partial success.
    /// </summary>
    public static string Write(
        string storeRoot,
        CurationRunRecord runRecord,
        IReadOnlyList<TileCurationRecord> records,
        IReadOnlyList<MismatchFinding> findings)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(storeRoot);
        ArgumentNullException.ThrowIfNull(runRecord);
        ArgumentNullException.ThrowIfNull(records);
        ArgumentNullException.ThrowIfNull(findings);

        runRecord.Verify(records.Count);

        string runDir = Path.Combine(storeRoot, "curation", runRecord.CurationRunId);
        Directory.CreateDirectory(runDir);

        WriteManifest(Path.Combine(runDir, "curation_manifest.parquet"), records);
        WriteFindings(Path.Combine(runDir, "curation_findings.parquet"), findings);
        File.WriteAllText(Path.Combine(runDir, "curation_run.json"), runRecord.ToJson());

        // The pointer is a plain-text file, not a symlink: symlink creation can require elevated
        // privileges on Windows (this project has hit that exact failure before with fixture
        // tests), and a one-line text file is trivially readable cross-language without any
        // platform-specific API.
        File.WriteAllText(Path.Combine(storeRoot, "curation", "latest"), runRecord.CurationRunId);

        return runDir;
    }

    private static void WriteManifest(string path, IReadOnlyList<TileCurationRecord> records)
    {
        using FileStream stream = File.Create(path);
        using ParquetWriter writer = ParquetWriter.CreateAsync(ManifestSchema, stream).GetAwaiter().GetResult();
        using ParquetRowGroupWriter group = writer.CreateRowGroup();

        WriteColumn(group, ManifestSchema, "build", records.Select(r => r.Build).ToArray());
        WriteColumn(group, ManifestSchema, "map", records.Select(r => r.Map).ToArray());
        WriteColumn(group, ManifestSchema, "tile_x", records.Select(r => r.TileX).ToArray());
        WriteColumn(group, ManifestSchema, "tile_y", records.Select(r => r.TileY).ToArray());
        WriteColumn(group, ManifestSchema, "tile_id", records.Select(r => r.TileId).ToArray());
        WriteColumn(group, ManifestSchema, "difficulty_bucket", records.Select(r => r.DifficultyBucket).ToArray());
        WriteColumn(group, ManifestSchema, "coverage_bucket", records.Select(r => r.CoverageBucket).ToArray());
        WriteColumn(group, ManifestSchema, "lighting_bucket", records.Select(r => r.LightingBucket).ToArray());
        WriteColumn(group, ManifestSchema, "synthetic_fidelity_status", records.Select(r => r.SyntheticFidelityStatus).ToArray());
        WriteColumn(group, ManifestSchema, "synthetic_fidelity_score", records.Select(r => r.SyntheticFidelityScore).ToArray());
        WriteColumn(group, ManifestSchema, "finding_count", records.Select(r => r.FindingCount).ToArray());
        WriteColumn(group, ManifestSchema, "curation_run_id", records.Select(r => r.CurationRunId).ToArray());
    }

    private static void WriteFindings(string path, IReadOnlyList<MismatchFinding> findings)
    {
        using FileStream stream = File.Create(path);
        using ParquetWriter writer = ParquetWriter.CreateAsync(FindingsSchema, stream).GetAwaiter().GetResult();
        using ParquetRowGroupWriter group = writer.CreateRowGroup();

        WriteColumn(group, FindingsSchema, "build", findings.Select(f => f.Build).ToArray());
        WriteColumn(group, FindingsSchema, "map", findings.Select(f => f.Map).ToArray());
        WriteColumn(group, FindingsSchema, "tile_x", findings.Select(f => f.TileX).ToArray());
        WriteColumn(group, FindingsSchema, "tile_y", findings.Select(f => f.TileY).ToArray());
        WriteColumn(group, FindingsSchema, "tile_id", findings.Select(f => f.TileId).ToArray());
        WriteColumn(group, FindingsSchema, "category", findings.Select(f => f.Category).ToArray());
        WriteColumn(group, FindingsSchema, "severity", findings.Select(f => f.Severity).ToArray());
        WriteColumn(group, FindingsSchema, "reason", findings.Select(f => f.Reason).ToArray());
        WriteColumn(group, FindingsSchema, "evaluability", findings.Select(f => f.Evaluability).ToArray());
        WriteColumn(group, FindingsSchema, "signal", findings.Select(f => f.Signal).ToArray());
        WriteColumn(group, FindingsSchema, "curation_run_id", findings.Select(f => f.CurationRunId).ToArray());
    }

    private static void WriteColumn(ParquetRowGroupWriter group, ParquetSchema schema, string fieldName, Array data)
    {
        DataField field = schema.GetDataFields().First(f => f.Name == fieldName);
        group.WriteColumnAsync(new DataColumn(field, data)).GetAwaiter().GetResult();
    }

    /// <summary>Reads back the row count of an already-written manifest, for verification (tests
    /// and orchestration code that wants to confirm a write actually landed the expected rows
    /// without re-deriving them).</summary>
    public static int ReadManifestRowCount(string manifestParquetPath)
    {
        using FileStream stream = File.OpenRead(manifestParquetPath);
        using ParquetReader reader = ParquetReader.CreateAsync(stream).GetAwaiter().GetResult();
        int total = 0;
        for (int rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using ParquetRowGroupReader group = reader.OpenRowGroupReader(rg);
            total += (int)group.RowCount;
        }
        return total;
    }
}
