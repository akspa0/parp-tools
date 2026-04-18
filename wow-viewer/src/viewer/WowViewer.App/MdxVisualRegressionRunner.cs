using System.Security.Cryptography;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;

namespace WowViewer.App;

internal sealed class MdxVisualRegressionManifest
{
    public List<MdxVisualRegressionCase> Cases { get; init; } = [];
}

internal sealed class MdxVisualRegressionCase
{
    public string Name { get; init; } = string.Empty;

    public string? InputPath { get; init; }

    public string? ArchiveRoot { get; init; }

    public string? VirtualPath { get; init; }

    public string? BuildLabel { get; init; }

    public int SequenceIndex { get; init; }

    public int TimeMs { get; init; }

    public int VisualSize { get; init; }

    public int VisualWidth { get; init; }

    public int VisualHeight { get; init; }

    public string CameraMode { get; init; } = "frame";

    public string? CameraPreset { get; init; }

    public float CameraAzimuthDegrees { get; init; } = 35.0f;

    public float CameraElevationDegrees { get; init; } = 25.0f;

    public float CameraFovDegrees { get; init; } = 60.0f;

    public float CameraZoomFactor { get; init; } = 0.72f;

    public string BaselineImagePath { get; init; } = string.Empty;

    public int MaxDifferentPixels { get; init; }

    public byte MaxChannelDelta { get; init; }
}

internal readonly record struct MdxVisualRegressionCaseResult(
    string Name,
    string ActualImagePath,
    string BaselineImagePath,
    string ActualHash,
    string BaselineHash,
    int Width,
    int Height,
    int DifferentPixels,
    byte MaxObservedChannelDelta,
    bool Passed,
    bool BaselineUpdated,
    string? DiffImagePath,
    string? FailureReason);

internal static class MdxVisualRegressionRunner
{
    private const int DefaultVisualSize = 512;

    public static int Run(string manifestPath, string? actualRoot, string? diffRoot, bool updateBaselines)
    {
        if (string.IsNullOrWhiteSpace(manifestPath))
            throw new ArgumentException("Provide --manifest <file.json> for mdx-visual-regression.", nameof(manifestPath));

        string resolvedManifestPath = Path.GetFullPath(manifestPath);
        string manifestDirectory = Path.GetDirectoryName(resolvedManifestPath)
            ?? throw new DirectoryNotFoundException($"Could not resolve a directory for manifest '{resolvedManifestPath}'.");
        if (!File.Exists(resolvedManifestPath))
            throw new FileNotFoundException($"Could not find MDX visual regression manifest '{resolvedManifestPath}'.", resolvedManifestPath);

        MdxVisualRegressionManifest manifest = JsonSerializer.Deserialize<MdxVisualRegressionManifest>(
            File.ReadAllText(resolvedManifestPath),
            new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                ReadCommentHandling = JsonCommentHandling.Skip,
                AllowTrailingCommas = true,
            })
            ?? throw new InvalidDataException($"Could not deserialize MDX visual regression manifest '{resolvedManifestPath}'.");

        if (manifest.Cases.Count == 0)
            throw new InvalidDataException($"MDX visual regression manifest '{resolvedManifestPath}' does not define any cases.");

        string resolvedActualRoot = string.IsNullOrWhiteSpace(actualRoot)
            ? Path.Combine(Path.GetTempPath(), "wowviewer-mdx-visual-regression")
            : Path.GetFullPath(actualRoot);
        Directory.CreateDirectory(resolvedActualRoot);

        string? resolvedDiffRoot = string.IsNullOrWhiteSpace(diffRoot) ? null : Path.GetFullPath(diffRoot);
        if (!string.IsNullOrWhiteSpace(resolvedDiffRoot))
            Directory.CreateDirectory(resolvedDiffRoot);

        List<MdxVisualRegressionCaseResult> results = new(manifest.Cases.Count);
        foreach (MdxVisualRegressionCase regressionCase in manifest.Cases)
            results.Add(RunCase(regressionCase, manifestDirectory, resolvedActualRoot, resolvedDiffRoot, updateBaselines));

        int passedCount = results.Count(static result => result.Passed);
        int failedCount = results.Count - passedCount;
        foreach (MdxVisualRegressionCaseResult result in results)
        {
            string status = result.Passed ? (result.BaselineUpdated ? "updated" : "pass") : "FAIL";
            Console.WriteLine(
                $"MDX visual regression {status}: case={result.Name} size={result.Width}x{result.Height} diffPixels={result.DifferentPixels} maxDelta={result.MaxObservedChannelDelta} actualHash={result.ActualHash} baselineHash={result.BaselineHash}");
            Console.WriteLine($"  actual={result.ActualImagePath}");
            Console.WriteLine($"  baseline={result.BaselineImagePath}");
            if (!string.IsNullOrWhiteSpace(result.DiffImagePath))
                Console.WriteLine($"  diff={result.DiffImagePath}");
            if (!string.IsNullOrWhiteSpace(result.FailureReason))
                Console.WriteLine($"  reason={result.FailureReason}");
        }

        Console.WriteLine($"MDX visual regression summary: passed={passedCount} failed={failedCount} updated={results.Count(static result => result.BaselineUpdated)} actualRoot={resolvedActualRoot}");
        return failedCount == 0 ? 0 : 1;
    }

    private static MdxVisualRegressionCaseResult RunCase(MdxVisualRegressionCase regressionCase, string manifestDirectory, string actualRoot, string? diffRoot, bool updateBaselines)
    {
        if (string.IsNullOrWhiteSpace(regressionCase.Name))
            throw new InvalidDataException("Every MDX visual regression case must define a non-empty 'name'.");
        if (string.IsNullOrWhiteSpace(regressionCase.BaselineImagePath))
            throw new InvalidDataException($"MDX visual regression case '{regressionCase.Name}' must define 'baselineImagePath'.");

        string safeName = SanitizeFileName(regressionCase.Name);
        string actualImagePath = Path.Combine(actualRoot, $"{safeName}.png");
        string baselineImagePath = ResolveOptionalPath(manifestDirectory, regressionCase.BaselineImagePath)
            ?? throw new InvalidDataException($"MDX visual regression case '{regressionCase.Name}' has an invalid baseline image path.");

        MdxPreviewLoadRequest request = BuildRequest(regressionCase, manifestDirectory);
        MdxGpuPreviewCaptureRunner.Capture(request, actualImagePath);

        if (updateBaselines)
        {
            string? baselineDirectory = Path.GetDirectoryName(baselineImagePath);
            if (!string.IsNullOrWhiteSpace(baselineDirectory))
                Directory.CreateDirectory(baselineDirectory);

            File.Copy(actualImagePath, baselineImagePath, overwrite: true);
            (int width, int height, string actualHash) = ReadImageFingerprint(actualImagePath);
            return new MdxVisualRegressionCaseResult(
                regressionCase.Name,
                actualImagePath,
                baselineImagePath,
                actualHash,
                actualHash,
                width,
                height,
                DifferentPixels: 0,
                MaxObservedChannelDelta: 0,
                Passed: true,
                BaselineUpdated: true,
                DiffImagePath: null,
                FailureReason: null);
        }

        if (!File.Exists(baselineImagePath))
        {
            (int width, int height, string actualHash) = ReadImageFingerprint(actualImagePath);
            return new MdxVisualRegressionCaseResult(
                regressionCase.Name,
                actualImagePath,
                baselineImagePath,
                actualHash,
                BaselineHash: string.Empty,
                width,
                height,
                DifferentPixels: -1,
                MaxObservedChannelDelta: 255,
                Passed: false,
                BaselineUpdated: false,
                DiffImagePath: null,
                FailureReason: "Baseline image is missing.");
        }

        return CompareAgainstBaseline(regressionCase, actualImagePath, baselineImagePath, diffRoot, safeName);
    }

    private static MdxVisualRegressionCaseResult CompareAgainstBaseline(MdxVisualRegressionCase regressionCase, string actualImagePath, string baselineImagePath, string? diffRoot, string safeName)
    {
        using Image<Rgba32> actual = Image.Load<Rgba32>(actualImagePath);
        using Image<Rgba32> baseline = Image.Load<Rgba32>(baselineImagePath);

        string actualHash = ComputePixelHash(actual);
        string baselineHash = ComputePixelHash(baseline);
        if (actual.Width != baseline.Width || actual.Height != baseline.Height)
        {
            return new MdxVisualRegressionCaseResult(
                regressionCase.Name,
                actualImagePath,
                baselineImagePath,
                actualHash,
                baselineHash,
                actual.Width,
                actual.Height,
                DifferentPixels: -1,
                MaxObservedChannelDelta: 255,
                Passed: false,
                BaselineUpdated: false,
                DiffImagePath: null,
                FailureReason: $"Image dimensions differ. actual={actual.Width}x{actual.Height} baseline={baseline.Width}x{baseline.Height}");
        }

        int differentPixels = 0;
        byte maxObservedChannelDelta = 0;
        Image<Rgba32>? diffImage = null;
        if (!string.IsNullOrWhiteSpace(diffRoot))
            diffImage = new Image<Rgba32>(actual.Width, actual.Height);

        for (int y = 0; y < actual.Height; y++)
        {
            for (int x = 0; x < actual.Width; x++)
            {
                Rgba32 actualPixel = actual[x, y];
                Rgba32 baselinePixel = baseline[x, y];
                byte observedDelta = GetMaxChannelDelta(actualPixel, baselinePixel);
                if (observedDelta > maxObservedChannelDelta)
                    maxObservedChannelDelta = observedDelta;

                bool differs = observedDelta > regressionCase.MaxChannelDelta;
                if (differs)
                    differentPixels++;

                if (diffImage is not null)
                {
                    diffImage[x, y] = differs
                        ? new Rgba32(255, 0, 255, 255)
                        : new Rgba32((byte)(actualPixel.R / 3), (byte)(actualPixel.G / 3), (byte)(actualPixel.B / 3), 255);
                }
            }
        }

        string? diffImagePath = null;
        if (diffImage is not null && differentPixels > 0)
        {
            diffImagePath = Path.Combine(diffRoot!, $"{safeName}.diff.png");
            diffImage.Save(diffImagePath);
        }

        diffImage?.Dispose();

        bool passed = differentPixels <= regressionCase.MaxDifferentPixels;
        return new MdxVisualRegressionCaseResult(
            regressionCase.Name,
            actualImagePath,
            baselineImagePath,
            actualHash,
            baselineHash,
            actual.Width,
            actual.Height,
            differentPixels,
            maxObservedChannelDelta,
            passed,
            BaselineUpdated: false,
            diffImagePath,
            passed ? null : $"Expected <= {regressionCase.MaxDifferentPixels} differing pixels with per-channel delta <= {regressionCase.MaxChannelDelta}, found {differentPixels} differing pixels.");
    }

    private static MdxPreviewLoadRequest BuildRequest(MdxVisualRegressionCase regressionCase, string manifestDirectory)
    {
        string? inputPath = ResolveOptionalPath(manifestDirectory, regressionCase.InputPath);
        string? archiveRoot = ResolveOptionalPath(manifestDirectory, regressionCase.ArchiveRoot);
        string? virtualPath = regressionCase.VirtualPath;
        if (string.IsNullOrWhiteSpace(inputPath) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
            throw new InvalidDataException($"MDX visual regression case '{regressionCase.Name}' must define either 'inputPath' or both 'archiveRoot' and 'virtualPath'.");

        int visualWidth = regressionCase.VisualSize > 0 ? regressionCase.VisualSize : regressionCase.VisualWidth > 0 ? regressionCase.VisualWidth : DefaultVisualSize;
        int visualHeight = regressionCase.VisualSize > 0 ? regressionCase.VisualSize : regressionCase.VisualHeight > 0 ? regressionCase.VisualHeight : DefaultVisualSize;
        if (visualWidth < 16 || visualHeight < 16)
            throw new InvalidDataException($"MDX visual regression case '{regressionCase.Name}' must use a visual size of at least 16x16.");

        if (!Enum.TryParse(regressionCase.CameraMode, ignoreCase: true, out PreviewCameraMode cameraMode))
            throw new InvalidDataException($"MDX visual regression case '{regressionCase.Name}' uses unsupported camera mode '{regressionCase.CameraMode}'.");

        return new MdxPreviewLoadRequest
        {
            InputPath = string.IsNullOrWhiteSpace(archiveRoot) ? inputPath : null,
            ArchiveRoot = archiveRoot,
            VirtualPath = string.IsNullOrWhiteSpace(archiveRoot) ? null : virtualPath,
            BuildLabel = regressionCase.BuildLabel,
            SequenceIndex = regressionCase.SequenceIndex,
            TimeMs = regressionCase.TimeMs,
            VisualWidth = visualWidth,
            VisualHeight = visualHeight,
            Camera = new PreviewCameraSettings
            {
                Mode = cameraMode,
                PresetName = regressionCase.CameraPreset,
                AzimuthDegrees = regressionCase.CameraAzimuthDegrees,
                ElevationDegrees = regressionCase.CameraElevationDegrees,
                FieldOfViewDegrees = regressionCase.CameraFovDegrees,
                ZoomFactor = regressionCase.CameraZoomFactor,
            },
        };
    }

    private static (int Width, int Height, string Hash) ReadImageFingerprint(string path)
    {
        using Image<Rgba32> image = Image.Load<Rgba32>(path);
        return (image.Width, image.Height, ComputePixelHash(image));
    }

    private static string ComputePixelHash(Image<Rgba32> image)
    {
        using SHA256 sha256 = SHA256.Create();
        byte[] bytes = new byte[image.Width * image.Height * 4];
        int offset = 0;
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                Rgba32 pixel = image[x, y];
                bytes[offset + 0] = pixel.R;
                bytes[offset + 1] = pixel.G;
                bytes[offset + 2] = pixel.B;
                bytes[offset + 3] = pixel.A;
                offset += 4;
            }
        }

        return Convert.ToHexStringLower(sha256.ComputeHash(bytes));
    }

    private static byte GetMaxChannelDelta(Rgba32 actualPixel, Rgba32 baselinePixel)
    {
        byte red = (byte)Math.Abs(actualPixel.R - baselinePixel.R);
        byte green = (byte)Math.Abs(actualPixel.G - baselinePixel.G);
        byte blue = (byte)Math.Abs(actualPixel.B - baselinePixel.B);
        byte alpha = (byte)Math.Abs(actualPixel.A - baselinePixel.A);
        return Math.Max(Math.Max(red, green), Math.Max(blue, alpha));
    }

    private static string SanitizeFileName(string name)
    {
        char[] invalidChars = Path.GetInvalidFileNameChars();
        char[] sanitized = name.Select(static c => c).ToArray();
        for (int index = 0; index < sanitized.Length; index++)
        {
            if (invalidChars.Contains(sanitized[index]) || char.IsWhiteSpace(sanitized[index]))
                sanitized[index] = '_';
        }

        return sanitized.Length == 0 ? "case" : new string(sanitized);
    }

    private static string? ResolveOptionalPath(string baseDirectory, string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return null;

        return Path.IsPathRooted(path)
            ? Path.GetFullPath(path)
            : Path.GetFullPath(Path.Combine(baseDirectory, path));
    }
}
