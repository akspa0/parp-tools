using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Renderer.Headless;
using WowViewer.Core.Renderer.Scene;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Tools.Capture;

internal static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "--help" or "-h")
        {
            Console.WriteLine("""
                WowViewer.Tool.Capture — headless terrain tile renderer

                Usage:
                  render    --client-root <dir> --tile-name <name> --output <path> [--resolution <int>]

                Examples:
                  render --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368" --tile-name Azeroth_30_48 --output output.png
                """);
            return 0;
        }

        return args[0] switch
        {
            "render" => RunRender(args[1..]),
            var x => throw new InvalidOperationException($"Unknown command: {x}")
        };
    }

    private static int RunRender(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? tileName = GetOption(args, "--tile-name", "-t");
        string? outputPath = GetOption(args, "--output", "-o");
        int resolution = GetIntOption(args, "--resolution", "-r") ?? 512;

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(tileName) || string.IsNullOrWhiteSpace(outputPath))
        {
            Console.Error.WriteLine("Error: --client-root, --tile-name, and --output are required.");
            return 1;
        }

        Console.WriteLine($"Loading tile: {tileName}");
        Console.WriteLine($"Client root: {clientRoot}");
        Console.WriteLine($"Output: {outputPath}");
        Console.WriteLine($"Resolution: {resolution}");

        string mapName = tileName[..tileName.LastIndexOf('_', tileName.LastIndexOf('_') - 1)];
        string adtPath = $"World\\Maps\\{mapName}\\{tileName}.adt";

        IArchiveCatalogFactory factory = new NativeMpqServiceFactory();
        using IArchiveCatalog catalog = factory.Create();
        catalog.LoadArchives([clientRoot]);

        if (!catalog.FileExists(adtPath))
        {
            Console.Error.WriteLine($"ADT file not found: {adtPath}");
            return 2;
        }

        byte[]? adtData = catalog.ReadFile(adtPath);
        if (adtData == null)
        {
            Console.Error.WriteLine($"Failed to read ADT: {adtPath}");
            return 2;
        }

        var fileSummary = MapFileSummaryReader.Read(new MemoryStream(adtData), adtPath);
        var tileData = WorldTerrainTileBuilder.Read(new MemoryStream(adtData), fileSummary);

        if (tileData.ChunkCount == 0)
        {
            Console.Error.WriteLine("Tile has no chunks");
            return 2;
        }

        Console.WriteLine($"Chunks: {tileData.ChunkCount}, Layers: {tileData.TotalTextureLayerCount}");

        using var context = new HeadlessContext(resolution, resolution);
        using var surface = new RenderSurface(context.GL, resolution, resolution);
        var camera = new SceneCamera();
        using var renderer = new SceneRenderer(context.GL);

        float tileSize = 16f * 533.333f;
        float half = tileSize / 2f;

        camera.LookAtPosition(new Vector3(half, half, 50f), 800f, 180f, -60f);

        surface.Clear(0.34f, 0.38f, 0.42f, 1f);

        renderer.RenderTile(camera, tileData, RenderVariant.Primary);

        var capture = new FrameCapture(context, resolution, resolution);
        byte[] rgba = capture.CaptureRgba();
        PngWriter.WritePng(outputPath, rgba, resolution, resolution);

        Console.WriteLine($"Saved: {outputPath}");
        return 0;
    }

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (names.Contains(args[i], StringComparer.OrdinalIgnoreCase) && i + 1 < args.Length)
                return args[i + 1];
        }
        return null;
    }

    private static int? GetIntOption(string[] args, params string[] names)
    {
        string? value = GetOption(args, names);
        return int.TryParse(value, out int parsed) ? parsed : null;
    }
}
