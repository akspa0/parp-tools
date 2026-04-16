using System.Text.Json;
using WowViewer.Core;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime;
using WowViewer.Core.Runtime.M2;

if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
{
    ShowUsage();
    return;
}

string command = args[0].ToLowerInvariant();
string[] tail = args.Skip(1).ToArray();

switch (command)
{
    case "m2-frame":
        RunM2Frame(tail);
        break;
    default:
        Console.Error.WriteLine($"Unknown WowViewer.App command '{command}'.");
        ShowUsage();
        Environment.ExitCode = 1;
        break;
}

static void RunM2Frame(string[] args)
{
    string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
    string? archiveRoot = GetOption(args, "--archive-root", "-r");
    string? virtualPath = GetOption(args, "--virtual-path", "-v");
    string? profileIndexText = GetOption(args, "--profile-index", "-p");
    string? sequenceIndexText = GetOption(args, "--sequence-index", "-s");
    string? timeMsText = GetOption(args, "--time-ms", "-t");
    string? goldenOutput = GetOption(args, "--golden-output");
    string? renderFrameOutput = GetOption(args, "--render-frame-output");
    string? visualOutput = GetOption(args, "--visual-output");

    if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
        virtualPath = input;

    if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
    {
        Console.Error.WriteLine("Error: provide --input <file.m2|file.mdx|file.mdl> or --archive-root <dir> with --virtual-path <path/to/file.m2|file.mdx|file.mdl>.");
        Environment.ExitCode = 1;
        return;
    }

    int profileIndex = 0;
    if (!string.IsNullOrWhiteSpace(profileIndexText)
        && (!int.TryParse(profileIndexText, out profileIndex) || profileIndex < 0 || profileIndex > 99))
    {
        Console.Error.WriteLine("Error: --profile-index must be an integer in the range 0..99.");
        Environment.ExitCode = 1;
        return;
    }

    if (string.IsNullOrWhiteSpace(sequenceIndexText) || !int.TryParse(sequenceIndexText, out int sequenceIndex) || sequenceIndex < 0)
    {
        Console.Error.WriteLine("Error: --sequence-index must be a non-negative integer.");
        Environment.ExitCode = 1;
        return;
    }

    int timeMs = 0;
    if (!string.IsNullOrWhiteSpace(timeMsText) && !int.TryParse(timeMsText, out timeMs))
    {
        Console.Error.WriteLine("Error: --time-ms must be an integer.");
        Environment.ExitCode = 1;
        return;
    }

    string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
        ? virtualPath
        : input!;

    try
    {
        M2RuntimeFrameResult result = BuildM2Frame(sourceLabel, archiveRoot, virtualPath, input, profileIndex, sequenceIndex, timeMs);
        M2RuntimeGoldenFrame frame = result.GoldenFrame;
        Console.WriteLine($"WowViewer.App m2-frame: model={frame.CanonicalModelPath} version=0x{frame.ModelVersion:X} sequence={frame.RequestedSequenceIndex}->{frame.ResolvedSequenceIndex} timeMs={frame.TimeMs} bones={frame.BoneCount} skinnedVertices={frame.SkinnedVertexCount} visiblePasses={frame.VisiblePassCount} batches={frame.Batches.Count} hash={frame.RuntimeHash}");
        foreach (M2RuntimeGoldenBatch batch in frame.Batches)
            Console.WriteLine($"WowViewer.App m2-frame batch[{batch.BatchIndex}]: family={batch.Family} handler={batch.Handler} direct={batch.Direct} entries={batch.EntryCount} effect={batch.EffectKey} vertices={batch.VertexCount} indices={batch.IndexCount}");
        Console.WriteLine($"WowViewer.App m2-frame render-frame: commands={result.RenderFrame.CommandCount} backendVertices={result.RenderFrame.BackendVertexCount} backendIndices={result.RenderFrame.BackendIndexCount} submittedVertices={result.RenderFrame.SubmittedVertexCount} submittedIndices={result.RenderFrame.SubmittedIndexCount} hash={result.RenderFrame.FrameHash}");
        Console.WriteLine($"WowViewer.App m2-frame effects: particles={result.EffectRuntimeState.Particles.Count} visibleParticles={result.EffectRuntimeState.VisibleParticleEmitterCount} ribbons={result.EffectRuntimeState.Ribbons.Count} visibleRibbons={result.EffectRuntimeState.VisibleRibbonEmitterCount}");
        Console.WriteLine($"WowViewer.App m2-frame visual: size={result.VisualSnapshot.Width}x{result.VisualSnapshot.Height} litPixels={result.VisualSnapshot.LitPixelCount} hash={result.VisualSnapshot.VisualHash}");

        if (!string.IsNullOrWhiteSpace(goldenOutput))
            WriteJson(goldenOutput, frame);
        if (!string.IsNullOrWhiteSpace(renderFrameOutput))
            WriteJson(renderFrameOutput, result.RenderFrame);
        if (!string.IsNullOrWhiteSpace(visualOutput))
            WriteBmp(visualOutput, result.VisualSnapshot);
    }
    catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        Environment.ExitCode = 1;
    }
}

static M2RuntimeFrameResult BuildM2Frame(
    string sourceLabel,
    string? archiveRoot,
    string? virtualPath,
    string? input,
    int profileIndex,
    int sequenceIndex,
    int timeMs)
{
    byte[] modelBytes = ReadBytes(sourceLabel, archiveRoot, virtualPath, input);
    using MemoryStream modelStream = new(modelBytes, writable: false);
    M2ModelDocument model = M2ModelReader.Read(modelStream, sourceLabel);

    using MemoryStream geometryStream = new(modelBytes, writable: false);
    M2GeometryDocument geometry = M2GeometryReader.Read(geometryStream, sourceLabel);

    M2SkinProfileRuntimeState skinState = M2SkinProfileRuntime.Choose(model, profileIndex);
    byte[] skinBytes = ReadCompanionBytes(skinState.Selection.CompanionPath, archiveRoot, input);
    using MemoryStream skinStream = new(skinBytes, writable: false);
    M2SkinDocument skin = M2SkinReader.Read(skinStream, skinState.Selection.CompanionPath);
    skinState = M2SkinProfileRuntime.Initialize(M2SkinProfileRuntime.Load(skinState, skin));

    M2StaticRenderModel renderModel = M2StaticRenderModelBuilder.Build(geometry, skinState);
    M2ExternalAnimationRuntimeState externalAnimationState = M2ExternalAnimationRuntime.Choose(model, sequenceIndex);
    if (externalAnimationState.UsesExternalFile && !string.IsNullOrWhiteSpace(externalAnimationState.CompanionPath))
    {
        byte[] animBytes = ReadCompanionBytes(externalAnimationState.CompanionPath, archiveRoot, input);
        using MemoryStream animStream = new(animBytes, writable: false);
        M2ExternalAnimationDocument animation = M2AnimationReader.Read(animStream, externalAnimationState.CompanionPath);
        externalAnimationState = M2ExternalAnimationRuntime.Load(externalAnimationState, animation);
    }

    return M2RuntimeFramePipeline.Build(model, renderModel, sequenceIndex, timeMs, externalAnimationState);
}

static byte[] ReadBytes(string sourceLabel, string? archiveRoot, string? virtualPath, string? input)
{
    if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
        return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], new ArchiveCatalogBootstrapOptions());

    if (!string.IsNullOrWhiteSpace(input) && File.Exists(input))
        return File.ReadAllBytes(input);

    throw new FileNotFoundException($"Could not read M2 input '{sourceLabel}'.", sourceLabel);
}

static byte[] ReadCompanionBytes(string companionPath, string? archiveRoot, string? input)
{
    if (!string.IsNullOrWhiteSpace(archiveRoot))
        return ArchiveVirtualFileReader.ReadVirtualFile(companionPath, [archiveRoot], new ArchiveCatalogBootstrapOptions());

    string localPath = companionPath;
    if (!Path.IsPathRooted(localPath) && !string.IsNullOrWhiteSpace(input))
    {
        string? inputDirectory = Path.GetDirectoryName(Path.GetFullPath(input));
        if (!string.IsNullOrWhiteSpace(inputDirectory))
            localPath = Path.Combine(inputDirectory, Path.GetFileName(companionPath));
    }

    return File.ReadAllBytes(localPath);
}

static void WriteJson<T>(string output, T payload)
{
    string outputPath = Path.GetFullPath(output);
    string? directory = Path.GetDirectoryName(outputPath);
    if (!string.IsNullOrWhiteSpace(directory))
        Directory.CreateDirectory(directory);

    File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));
    Console.WriteLine($"Wrote {outputPath}");
}

static void WriteBmp(string output, M2SoftwareVisualSnapshot snapshot)
{
    string outputPath = Path.GetFullPath(output);
    string? directory = Path.GetDirectoryName(outputPath);
    if (!string.IsNullOrWhiteSpace(directory))
        Directory.CreateDirectory(directory);

    using FileStream stream = File.Create(outputPath);
    M2SoftwareVisualSnapshotBuilder.WriteBmp(stream, snapshot);
    Console.WriteLine($"Wrote {outputPath}");
}

static string? GetOption(string[] args, params string[] names)
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

static string? GetFirstPositionalArgument(string[] args)
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

static void ShowUsage()
{
    Console.WriteLine("WowViewer.App");
    Console.WriteLine($"Solution: {ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
    Console.WriteLine($"PM4 canonical owner: {Pm4Boundary.CanonicalOwner}");
    Console.WriteLine($"PM4 legacy reference: {Pm4Boundary.LegacyReference}");
    Console.WriteLine($"Runtime service areas: {RuntimeBoundaries.All.Length}");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  wowviewer-app m2-frame --archive-root <game|data dir> --virtual-path <path/to/file.m2> --sequence-index <n> [--time-ms <ms>] [--profile-index <n>] [--golden-output <json>] [--render-frame-output <json>] [--visual-output <bmp>]");
    Console.WriteLine("  wowviewer-app m2-frame --input <file.m2> --sequence-index <n> [--time-ms <ms>] [--profile-index <n>] [--golden-output <json>] [--render-frame-output <json>] [--visual-output <bmp>]");
}
