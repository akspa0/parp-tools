using WowViewer.Tools.ValidationCapture;

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
            Assert.Contains("Variant count: 4", capture.Output, StringComparison.Ordinal);
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
            Assert.Contains("Validation capture stub run completed: 4/4 succeeded, 0 timed out.", capture.Output, StringComparison.Ordinal);

            Assert.True(File.Exists(Path.Combine(temp.RootPath, "primary", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "noliquids", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "noobjects", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "objectsonly", "Azeroth_30_48_viewer_validation.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "images", "Azeroth_30_48_object_visibility_mask.png")));
            Assert.True(File.Exists(Path.Combine(temp.RootPath, "images", "Azeroth_30_48_no_objects.png")));
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
}