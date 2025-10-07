using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WoWRollback.Orchestrator;

internal sealed class ManifestWriter
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public void Write(SessionContext session, PipelineRunResult result)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }
        if (result is null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        var manifest = new ManifestModel
        {
            SessionId = session.SessionId,
            Timestamp = DateTimeOffset.Now,
            Options = new ManifestOptionsModel
            {
                Maps = session.Options.Maps,
                Versions = session.Options.Versions,
                AlphaRoot = session.Options.AlphaRoot,
                OutputRoot = session.Options.OutputRoot,
                Serve = session.Options.Serve,
                Port = session.Options.Port,
                Verbose = session.Options.Verbose,
            },
            Dbc = new ManifestDbcModel
            {
                Success = result.Dbc.Success,
                Versions = MapDbcVersions(result.Dbc.Versions),
                SharedRoot = session.SharedDbcRoot,
                SharedCrosswalkRoot = session.SharedCrosswalkRoot,
            },
            Adt = new ManifestAdtModel
            {
                Results = MapAdtResults(result.AdtResults),
                SharedDbcRoot = session.SharedDbcRoot,
                SharedCrosswalkRoot = session.SharedCrosswalkRoot,
            },
            Viewer = new ManifestViewerModel
            {
                Success = true,
                ViewerDirectory = session.Paths.ViewerDir,
                OverlayCount = 0,
                Notes = "Viewer stage not implemented.",
            },
        };

        Directory.CreateDirectory(Path.GetDirectoryName(session.Paths.ManifestPath)!);
        using var stream = File.Create(session.Paths.ManifestPath);
        JsonSerializer.Serialize(stream, manifest, SerializerOptions);
    }

    private static IReadOnlyList<ManifestDbcVersionModel> MapDbcVersions(IReadOnlyList<DbcVersionResult> results)
    {
        var list = new List<ManifestDbcVersionModel>(results.Count);
        foreach (var result in results)
        {
            list.Add(new ManifestDbcVersionModel
            {
                SourceVersion = result.SourceVersion,
                SourceAlias = result.SourceAlias,
                SourceDirectory = result.SourceDirectory,
                DbcOutputDirectory = result.DbcOutputDirectory,
                CrosswalkOutputDirectory = result.CrosswalkOutputDirectory,
                DumpExitCode = result.DumpExitCode,
                CompareExitCode = result.CompareExitCode,
                Error = result.Error,
            });
        }

        return list;
    }

    private static IReadOnlyList<ManifestAdtResultModel> MapAdtResults(IReadOnlyList<AdtStageResult> results)
    {
        var list = new List<ManifestAdtResultModel>(results.Count);
        foreach (var result in results)
        {
            list.Add(new ManifestAdtResultModel
            {
                Map = result.Map,
                Version = result.Version,
                Success = result.Success,
                TilesProcessed = result.TilesProcessed,
                AreaIdsPatched = result.AreaIdsPatched,
                AdtOutputDirectory = result.AdtOutputDirectory,
                Error = result.Error,
            });
        }

        return list;
    }

    private sealed record ManifestModel
    {
        [JsonPropertyName("session_id")]
        public string SessionId { get; init; } = string.Empty;

        [JsonPropertyName("timestamp")]
        public DateTimeOffset Timestamp { get; init; }

        [JsonPropertyName("options")]
        public ManifestOptionsModel Options { get; init; } = new();

        [JsonPropertyName("dbc")]
        public ManifestDbcModel Dbc { get; init; } = new();

        [JsonPropertyName("adt")]
        public ManifestAdtModel Adt { get; init; } = new();

        [JsonPropertyName("viewer")]
        public ManifestViewerModel Viewer { get; init; } = new();
    }

    private sealed record ManifestOptionsModel
    {
        [JsonPropertyName("maps")]
        public IReadOnlyList<string> Maps { get; init; } = Array.Empty<string>();

        [JsonPropertyName("versions")]
        public IReadOnlyList<string> Versions { get; init; } = Array.Empty<string>();

        [JsonPropertyName("alpha_root")]
        public string AlphaRoot { get; init; } = string.Empty;

        [JsonPropertyName("output_root")]
        public string OutputRoot { get; init; } = string.Empty;

        [JsonPropertyName("serve")]
        public bool Serve { get; init; }

        [JsonPropertyName("port")]
        public int Port { get; init; }

        [JsonPropertyName("verbose")]
        public bool Verbose { get; init; }
    }

    private sealed record ManifestDbcModel
    {
        [JsonPropertyName("success")]
        public bool Success { get; init; }

        [JsonPropertyName("versions")]
        public IReadOnlyList<ManifestDbcVersionModel> Versions { get; init; } = Array.Empty<ManifestDbcVersionModel>();

        [JsonPropertyName("shared_root")]
        public string SharedRoot { get; init; } = string.Empty;

        [JsonPropertyName("shared_crosswalk_root")]
        public string SharedCrosswalkRoot { get; init; } = string.Empty;
    }

    private sealed record ManifestDbcVersionModel
    {
        [JsonPropertyName("source_version")]
        public string SourceVersion { get; init; } = string.Empty;

        [JsonPropertyName("source_alias")]
        public string SourceAlias { get; init; } = string.Empty;

        [JsonPropertyName("source_directory")]
        public string SourceDirectory { get; init; } = string.Empty;

        [JsonPropertyName("dbc_output_directory")]
        public string DbcOutputDirectory { get; init; } = string.Empty;

        [JsonPropertyName("crosswalk_output_directory")]
        public string CrosswalkOutputDirectory { get; init; } = string.Empty;

        [JsonPropertyName("dump_exit_code")]
        public int DumpExitCode { get; init; }

        [JsonPropertyName("compare_exit_code")]
        public int CompareExitCode { get; init; }

        [JsonPropertyName("error")]
        public string? Error { get; init; }
    }

    private sealed record ManifestAdtModel
    {
        [JsonPropertyName("results")]
        public IReadOnlyList<ManifestAdtResultModel> Results { get; init; } = Array.Empty<ManifestAdtResultModel>();

        [JsonPropertyName("shared_dbc_root")]
        public string SharedDbcRoot { get; init; } = string.Empty;

        [JsonPropertyName("shared_crosswalk_root")]
        public string SharedCrosswalkRoot { get; init; } = string.Empty;
    }

    private sealed record ManifestAdtResultModel
    {
        [JsonPropertyName("map")]
        public string Map { get; init; } = string.Empty;

        [JsonPropertyName("version")]
        public string Version { get; init; } = string.Empty;

        [JsonPropertyName("success")]
        public bool Success { get; init; }

        [JsonPropertyName("tiles_processed")]
        public int TilesProcessed { get; init; }

        [JsonPropertyName("area_ids_patched")]
        public int AreaIdsPatched { get; init; }

        [JsonPropertyName("adt_output_directory")]
        public string AdtOutputDirectory { get; init; } = string.Empty;

        [JsonPropertyName("error")]
        public string? Error { get; init; }
    }

    private sealed record ManifestViewerModel
    {
        [JsonPropertyName("success")]
        public bool Success { get; init; }

        [JsonPropertyName("viewer_directory")]
        public string ViewerDirectory { get; init; } = string.Empty;

        [JsonPropertyName("overlay_count")]
        public int OverlayCount { get; init; }

        [JsonPropertyName("notes")]
        public string? Notes { get; init; }
    }
}
