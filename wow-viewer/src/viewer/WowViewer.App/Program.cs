using System.Text.Json;
using WowViewer.Core;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.App;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Contains("--help", StringComparer.OrdinalIgnoreCase) || args.Contains("-h", StringComparer.OrdinalIgnoreCase))
        {
            ShowUsage();
            return 0;
        }

        if (args.Length == 0)
        {
            using WowViewerDesktopApp app = new();
            app.Run();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();

        try
        {
            switch (command)
            {
                case "viewer":
                case "app":
                    using (WowViewerDesktopApp app = new(ParseViewerSession(tail)))
                        app.Run();
                    return 0;

                case "m2-frame":
                    return RunM2Frame(tail);

                case "m2-gpu-frame":
                    return RunM2GpuFrame(tail);

                default:
                    using (WowViewerDesktopApp app = new(ParseViewerSession(args)))
                        app.Run();
                    return 0;
            }
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunM2Frame(string[] args)
    {
        M2PreviewLoadRequest request = ParseRequiredM2Request(args, defaultVisualSize: 256);
        M2PreviewLoadResult result = M2PreviewLoader.Load(request);
        M2RuntimeGoldenFrame frame = result.FrameResult.GoldenFrame;

        Console.WriteLine($"WowViewer.App m2-frame: model={frame.CanonicalModelPath} version=0x{frame.ModelVersion:X} sequence={frame.RequestedSequenceIndex}->{frame.ResolvedSequenceIndex} timeMs={frame.TimeMs} bones={frame.BoneCount} skinnedVertices={frame.SkinnedVertexCount} visiblePasses={frame.VisiblePassCount} batches={frame.Batches.Count} hash={frame.RuntimeHash}");
        foreach (M2RuntimeGoldenBatch batch in frame.Batches)
            Console.WriteLine($"WowViewer.App m2-frame batch[{batch.BatchIndex}]: family={batch.Family} handler={batch.Handler} direct={batch.Direct} entries={batch.EntryCount} effect={batch.EffectKey} vertices={batch.VertexCount} indices={batch.IndexCount}");

        M2RenderFrame renderFrame = result.FrameResult.RenderFrame;
        Console.WriteLine($"WowViewer.App m2-frame render-frame: commands={renderFrame.CommandCount} backendVertices={renderFrame.BackendVertexCount} backendIndices={renderFrame.BackendIndexCount} submittedVertices={renderFrame.SubmittedVertexCount} submittedIndices={renderFrame.SubmittedIndexCount} hash={renderFrame.FrameHash}");
        Console.WriteLine($"WowViewer.App m2-frame effects: particles={result.FrameResult.EffectRuntimeState.Particles.Count} visibleParticles={result.FrameResult.EffectRuntimeState.VisibleParticleEmitterCount} ribbons={result.FrameResult.EffectRuntimeState.Ribbons.Count} visibleRibbons={result.FrameResult.EffectRuntimeState.VisibleRibbonEmitterCount}");
        Console.WriteLine($"WowViewer.App m2-frame visual: size={result.FrameResult.VisualSnapshot.Width}x{result.FrameResult.VisualSnapshot.Height} litPixels={result.FrameResult.VisualSnapshot.LitPixelCount} hash={result.FrameResult.VisualSnapshot.VisualHash}");

        string? goldenOutput = GetOption(args, "--golden-output");
        string? renderFrameOutput = GetOption(args, "--render-frame-output");
        string? visualOutput = GetOption(args, "--visual-output");
        if (!string.IsNullOrWhiteSpace(goldenOutput))
            WriteJson(goldenOutput, frame);
        if (!string.IsNullOrWhiteSpace(renderFrameOutput))
            WriteJson(renderFrameOutput, renderFrame);
        if (!string.IsNullOrWhiteSpace(visualOutput))
            WriteBmp(visualOutput, result.FrameResult.VisualSnapshot);

        return 0;
    }

    private static int RunM2GpuFrame(string[] args)
    {
        M2PreviewLoadRequest request = ParseRequiredM2Request(args, defaultVisualSize: 512);
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(output))
            throw new ArgumentException("Provide --output <file.bmp> for m2-gpu-frame.");

        string outputPath = Path.GetFullPath(output);
        M2GpuPreviewCaptureRunner.Capture(request, outputPath);
        Console.WriteLine($"Wrote {outputPath}");
        return 0;
    }

    private static WowViewerSession? ParseViewerSession(string[] args)
    {
        string? workspaceText = GetOption(args, "--workspace", "-w");
        string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? virtualPath = GetOption(args, "--virtual-path", "-v");
        string? buildLabel = GetOption(args, "--build-label", "-b");
        string? profileIndexText = GetOption(args, "--profile-index", "-p");
        string? sequenceIndexText = GetOption(args, "--sequence-index", "-s");
        string? timeMsText = GetOption(args, "--time-ms", "-t");
        string? visualSizeText = GetOption(args, "--visual-size");

        if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
            return null;

        int.TryParse(profileIndexText, out int profileIndex);
        int.TryParse(sequenceIndexText, out int sequenceIndex);
        int.TryParse(timeMsText, out int timeMs);
        int.TryParse(visualSizeText, out int visualSize);
        if (visualSize < 16)
            visualSize = 384;

        WowViewerSession session = WowViewerSession.CreateDefault();
        session.WorkspaceMode = ParseWorkspaceMode(workspaceText);
        session.Source.Kind = string.IsNullOrWhiteSpace(archiveRoot)
            ? WowViewerAssetSourceKind.LocalFile
            : WowViewerAssetSourceKind.ArchiveVirtualPath;
        session.Source.InputPath = string.IsNullOrWhiteSpace(archiveRoot) ? input ?? string.Empty : string.Empty;
        session.Source.ArchiveRoot = archiveRoot ?? string.Empty;
        session.Source.VirtualPath = string.IsNullOrWhiteSpace(archiveRoot) ? string.Empty : (virtualPath ?? input ?? string.Empty);
        session.Source.BuildLabel = buildLabel ?? string.Empty;
        session.ProfileIndex = profileIndex;
        session.SequenceIndex = Math.Max(0, sequenceIndex);
        session.TimeMs = Math.Max(0, timeMs);
        session.VisualSize = visualSize;
        session.Normalize();
        return session;
    }

    private static M2PreviewLoadRequest ParseRequiredM2Request(string[] args, int defaultVisualSize)
    {
        string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? virtualPath = GetOption(args, "--virtual-path", "-v");
        string? buildLabel = GetOption(args, "--build-label", "-b");
        string? profileIndexText = GetOption(args, "--profile-index", "-p");
        string? sequenceIndexText = GetOption(args, "--sequence-index", "-s");
        string? timeMsText = GetOption(args, "--time-ms", "-t");
        string? visualSizeText = GetOption(args, "--visual-size");

        if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
            virtualPath = input;

        if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
            throw new ArgumentException("Provide --input <file.m2|file.mdx|file.mdl> or --archive-root <dir> with --virtual-path <path/to/file.m2|file.mdx|file.mdl>.");

        int profileIndex = 0;
        if (!string.IsNullOrWhiteSpace(profileIndexText)
            && (!int.TryParse(profileIndexText, out profileIndex) || profileIndex < 0 || profileIndex > 99))
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "--profile-index must be an integer in the range 0..99.");
        }

        if (string.IsNullOrWhiteSpace(sequenceIndexText) || !int.TryParse(sequenceIndexText, out int sequenceIndex) || sequenceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceIndexText), "--sequence-index must be a non-negative integer.");

        int timeMs = 0;
        if (!string.IsNullOrWhiteSpace(timeMsText) && !int.TryParse(timeMsText, out timeMs))
            throw new ArgumentOutOfRangeException(nameof(timeMsText), "--time-ms must be an integer.");

        int visualSize = defaultVisualSize;
        if (!string.IsNullOrWhiteSpace(visualSizeText))
        {
            if (!int.TryParse(visualSizeText, out visualSize) || visualSize < 16)
                throw new ArgumentOutOfRangeException(nameof(visualSizeText), "--visual-size must be an integer greater than or equal to 16.");
        }

        return new M2PreviewLoadRequest
        {
            InputPath = string.IsNullOrWhiteSpace(archiveRoot) ? input : null,
            ArchiveRoot = archiveRoot,
            VirtualPath = string.IsNullOrWhiteSpace(archiveRoot) ? null : (virtualPath ?? input),
            BuildLabel = buildLabel,
            ProfileIndex = profileIndex,
            SequenceIndex = sequenceIndex,
            TimeMs = timeMs,
            VisualWidth = visualSize,
            VisualHeight = visualSize,
        };
    }

    private static void WriteJson<T>(string output, T payload)
    {
        string outputPath = Path.GetFullPath(output);
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));
        Console.WriteLine($"Wrote {outputPath}");
    }

    private static void WriteBmp(string output, M2SoftwareVisualSnapshot snapshot)
    {
        string outputPath = Path.GetFullPath(output);
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using FileStream stream = File.Create(outputPath);
        M2SoftwareVisualSnapshotBuilder.WriteBmp(stream, snapshot);
        Console.WriteLine($"Wrote {outputPath}");
    }

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int index = 0; index < args.Length; index++)
        {
            if (!names.Contains(args[index], StringComparer.OrdinalIgnoreCase))
                continue;

            if (index + 1 >= args.Length)
                return null;

            return args[index + 1];
        }

        return null;
    }

    private static string? GetFirstPositionalArgument(string[] args)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string arg = args[index];
            if (!arg.StartsWith("-", StringComparison.Ordinal))
                return arg;

            index++;
        }

        return null;
    }

    private static void ShowUsage()
    {
        Console.WriteLine("WowViewer.App");
        Console.WriteLine($"Solution: {ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
        Console.WriteLine($"PM4 canonical owner: {Pm4Boundary.CanonicalOwner}");
        Console.WriteLine($"PM4 legacy reference: {Pm4Boundary.LegacyReference}");
        Console.WriteLine($"Runtime service areas: {RuntimeBoundaries.All.Length}");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  wowviewer-app");
        Console.WriteLine("  wowviewer-app viewer [--workspace m2|wmo|mdx] [--archive-root <game|data dir> --virtual-path <path/to/file> | --input <file>] [--build-label <label>] [--profile-index <n>] [--sequence-index <n>] [--time-ms <ms>] [--visual-size <px>]");
        Console.WriteLine("  wowviewer-app m2-frame --archive-root <game|data dir> --virtual-path <path/to/file.m2> --sequence-index <n> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--golden-output <json>] [--render-frame-output <json>] [--visual-output <bmp>]");
        Console.WriteLine("  wowviewer-app m2-frame --input <file.m2> --sequence-index <n> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--golden-output <json>] [--render-frame-output <json>] [--visual-output <bmp>]");
        Console.WriteLine("  wowviewer-app m2-gpu-frame --archive-root <game|data dir> --virtual-path <path/to/file.m2> --sequence-index <n> --output <file.bmp> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--visual-size <px>]");
        Console.WriteLine("  wowviewer-app m2-gpu-frame --input <file.m2> --sequence-index <n> --output <file.bmp> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--visual-size <px>]");
    }

    private static WowViewerWorkspaceMode ParseWorkspaceMode(string? workspaceText)
    {
        return workspaceText?.Trim().ToLowerInvariant() switch
        {
            null or "" or "m2" => WowViewerWorkspaceMode.StandaloneM2,
            "wmo" => WowViewerWorkspaceMode.StandaloneWmo,
            "mdx" => WowViewerWorkspaceMode.StandaloneMdx,
            _ => throw new ArgumentException("--workspace must be one of: m2, wmo, mdx."),
        };
    }
}
