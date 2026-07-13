using WowViewer.Tools.ValidationCapture;
using System.Text.Json;

namespace WowViewer.Core.Tests;

[CollectionDefinition("Console", DisableParallelization = true)]
public sealed class ConsoleCollectionDefinition;

[Collection("Console")]
public sealed class ValidationCaptureCommandTests
{
    private static readonly object ConsoleLock = new();

    [Fact]
    public void Execute_CaptureMissingRequiredArguments_ReturnsOne()
    {
        lock (ConsoleLock)
        {
            using ConsoleCapture capture = new();

            int exitCode = ValidationCaptureCommand.Execute(["capture", "--client-root", "client-only"]);

            Assert.Equal(1, exitCode);
            Assert.Contains("capture requires --client-root", capture.Error, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Execute_CaptureDryRun_ReturnsZeroAndPrintsSummary()
    {
        lock (ConsoleLock)
        {
            using TemporaryDirectory temp = new();
            using ConsoleCapture capture = new();

            int exitCode = ValidationCaptureCommand.Execute(CreateCaptureArgs(temp.RootPath, "--dry-run"));

            Assert.Equal(0, exitCode);
            Assert.Contains("Validation capture shell dry-run succeeded.", capture.Output, StringComparison.Ordinal);
            Assert.Contains("Variant count: 5", capture.Output, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Execute_CaptureStubScene_WritesVariantPngs()
    {
        lock (ConsoleLock)
        {
            using TemporaryDirectory temp = new();
            using ConsoleCapture capture = new();

            int exitCode = ValidationCaptureCommand.Execute(CreateCaptureArgs(temp.RootPath, "--stub-scene"));

            Assert.Equal(0, exitCode);
            Assert.Contains("Validation capture stub run completed: 5/5 succeeded, 0 timed out.", capture.Output, StringComparison.Ordinal);

            Assert.True(File.Exists(Path.Combine(temp.RootPath, "primary", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "noliquids", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "noobjects", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "objectsonly", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "terrain-shade", "Azeroth_30_48_terrain_shade.png")));
            string terrainShadeManifestPath = Path.Combine(temp.RootPath, "terrain-shade", "Azeroth_30_48_terrain_shade.json");
            Assert.True(File.Exists(terrainShadeManifestPath));
            string terrainShadeManifest = File.ReadAllText(terrainShadeManifestPath);
            Assert.Contains("\"guidance_only\": true", terrainShadeManifest, StringComparison.Ordinal);
            Assert.Contains("\"deployment_input\": false", terrainShadeManifest, StringComparison.Ordinal);
            Assert.Contains("\"canonical_terrain_target\": \"mcvt_vertex_z\"", terrainShadeManifest, StringComparison.Ordinal);
            Assert.Contains("fixed_viewer_contract_not_client_light_tables", terrainShadeManifest, StringComparison.Ordinal);
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "images", "Azeroth_30_48_object_visibility_mask.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "images", "Azeroth_30_48_no_objects.png")));
        }
    }

    [Fact]
    public void Execute_CaptureStubScene_TerrainShadeOnly_WritesOnlyGuidanceCapture()
    {
        lock (ConsoleLock)
        {
            using TemporaryDirectory temp = new();
            using ConsoleCapture capture = new();
            string[] args = [.. CreateCaptureArgs(temp.RootPath, "--stub-scene"), "--variants", "terrain-shade"];

            int exitCode = ValidationCaptureCommand.Execute(args);

            Assert.Equal(0, exitCode);
            Assert.Contains("Validation capture stub run completed: 1/1 succeeded", capture.Output, StringComparison.Ordinal);
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "terrain-shade", "Azeroth_30_48_terrain_shade.png")));
            Assert.False(Directory.Exists(Path.Combine(temp.RootPath, "primary")));
        }
    }

    [Fact]
    public void Execute_CaptureBatchMissingLedger_ReturnsOne()
    {
        lock (ConsoleLock)
        {
            using TemporaryDirectory temp = new();
            using ConsoleCapture capture = new();

            int exitCode = ValidationCaptureCommand.Execute(
            [
                "capture-batch",
                "--client-root", temp.RootPath,
                "--map-input", "World\\Maps\\Azeroth\\Azeroth.wdt",
                "--dataset-root", temp.RootPath,
                "--output-root", temp.RootPath,
                "--ledger-path", Path.Combine(temp.RootPath, "missing-ledger.json"),
                "--dry-run",
            ]);

            Assert.Equal(1, exitCode);
            Assert.Contains("ledger file not found", capture.Error, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void Execute_CaptureBatchDryRun_ReturnsZeroAndPrintsSummary()
    {
        lock (ConsoleLock)
        {
            using TemporaryDirectory temp = new();
            string ledgerPath = Path.Combine(temp.RootPath, "manifest_capture_ledger.json");
            WriteLedger(ledgerPath);
            using ConsoleCapture capture = new();

            int exitCode = ValidationCaptureCommand.Execute(
            [
                "capture-batch",
                "--client-root", temp.RootPath,
                "--map-input", "World\\Maps\\Azeroth\\Azeroth.wdt",
                "--dataset-root", temp.RootPath,
                "--output-root", temp.RootPath,
                "--ledger-path", ledgerPath,
                "--dry-run",
            ]);

            Assert.Equal(0, exitCode);
            Assert.Contains("Validation capture batch dry-run succeeded.", capture.Output, StringComparison.Ordinal);
            Assert.Contains("Tile count: 2", capture.Output, StringComparison.Ordinal);
            Assert.Contains("Variant count: 10", capture.Output, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Execute_CaptureBatchStubScene_WritesPoseMetadataArtifacts()
    {
        lock (ConsoleLock)
        {
            using TemporaryDirectory temp = new();
            string ledgerPath = Path.Combine(temp.RootPath, "manifest_capture_ledger.json");
            WriteLedgerWithPose(ledgerPath);
            using ConsoleCapture capture = new();

            int exitCode = ValidationCaptureCommand.Execute(
            [
                "capture-batch",
                "--client-root", temp.RootPath,
                "--map-input", "World\\Maps\\Azeroth\\Azeroth.wdt",
                "--dataset-root", temp.RootPath,
                "--output-root", temp.RootPath,
                "--ledger-path", ledgerPath,
                "--stub-scene",
            ]);

            Assert.Equal(0, exitCode);
            Assert.Contains("Validation capture batch stub run completed: 5/5 succeeded, 0 timed out.", capture.Output, StringComparison.Ordinal);

            string posePath = Path.Combine(temp.RootPath, "pose-metadata", "Azeroth_30_48_pose.json");
            Assert.True(File.Exists(posePath));

            string poseJson = File.ReadAllText(posePath);
            Assert.Contains("\"asset_path\": \"world/wmo/azeroth/buildings/human_farm/farm.wmo\"", poseJson, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("\"unique_id\": 1337", poseJson, StringComparison.Ordinal);
            Assert.Contains("\"rot_y\": 90", poseJson, StringComparison.Ordinal);
        }
    }

    private static string[] CreateCaptureArgs(string rootPath, string modeFlag)
    {
        return
        [
            "capture",
            "--client-root", rootPath,
            "--map-input", "World\\Maps\\Azeroth\\Azeroth.wdt",
            "--dataset-root", rootPath,
            "--output-root", rootPath,
            "--tile-name", "Azeroth_30_48",
            "--tile-x", "30",
            "--tile-y", "48",
            "--build", "0.5.3.3368",
            modeFlag,
        ];
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            RootPath = Path.Combine(Path.GetTempPath(), "WowViewer.ValidationCaptureTests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(RootPath);
        }

        public string RootPath { get; }

        public void Dispose()
        {
            if (Directory.Exists(RootPath))
                Directory.Delete(RootPath, recursive: true);
        }
    }

    private sealed class ConsoleCapture : IDisposable
    {
        private readonly TextWriter _originalOut;
        private readonly TextWriter _originalError;
        private readonly StringWriter _outputWriter;
        private readonly StringWriter _errorWriter;

        public ConsoleCapture()
        {
            _originalOut = Console.Out;
            _originalError = Console.Error;
            _outputWriter = new StringWriter();
            _errorWriter = new StringWriter();
            Console.SetOut(_outputWriter);
            Console.SetError(_errorWriter);
        }

        public string Output => _outputWriter.ToString();

        public string Error => _errorWriter.ToString();

        public void Dispose()
        {
            Console.SetOut(_originalOut);
            Console.SetError(_originalError);
            _outputWriter.Dispose();
            _errorWriter.Dispose();
        }
    }

    private static void WriteLedger(string ledgerPath)
    {
        var payload = new
        {
            build = "3_3_5_12340",
            tiles = new[]
            {
                new { tile_name = "Azeroth_30_48", tile_x = 30, tile_y = 48, status = "pending_capture" },
                new { tile_name = "Azeroth_30_49", tile_x = 30, tile_y = 49, status = "captured_partial" },
                new { tile_name = "Azeroth_30_50", tile_x = 30, tile_y = 50, status = "captured_complete" },
            },
        };

        File.WriteAllText(ledgerPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void WriteLedgerWithPose(string ledgerPath)
    {
        var payload = new
        {
            build = "3_3_5_12340",
            tiles = new[]
            {
                new
                {
                    tile_name = "Azeroth_30_48",
                    tile_x = 30,
                    tile_y = 48,
                    status = "pending_capture",
                    asset_path = "world/wmo/azeroth/buildings/human_farm/farm.wmo",
                    instance_type = "modf",
                    unique_id = 1337,
                    rot_x = 0.0,
                    rot_y = 90.0,
                    rot_z = 0.0,
                    scale = 1.0,
                },
            },
        };

        File.WriteAllText(ledgerPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }
}
