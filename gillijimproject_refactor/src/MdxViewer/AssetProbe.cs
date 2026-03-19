using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using SereniaBLPLib;

namespace MdxViewer;

internal static class AssetProbe
{
    public static bool TryRun(string[] args)
    {
        int probeIndex = Array.FindIndex(args, arg => arg.Equals("--probe-mdx", StringComparison.OrdinalIgnoreCase));
        if (probeIndex < 0)
            return false;

        if (args.Length <= probeIndex + 2)
        {
            Console.Error.WriteLine("Usage: MdxViewer --probe-mdx <gamePath> <modelVirtualPath> [--listfile <path>]");
            Environment.ExitCode = 1;
            return true;
        }

        string gamePath = args[probeIndex + 1];
        string modelVirtualPath = args[probeIndex + 2];
        string? listfilePath = TryGetOptionValue(args, "--listfile");

        ViewerLog.Verbose = true;
        MdxFile.Verbose = true;

        try
        {
            Run(gamePath, modelVirtualPath, listfilePath);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[AssetProbe] Failed: {ex}");
            Environment.ExitCode = 1;
        }

        return true;
    }

    private static void Run(string gamePath, string modelVirtualPath, string? listfilePath)
    {
        Console.WriteLine($"[AssetProbe] Game path: {gamePath}");
        Console.WriteLine($"[AssetProbe] Model path: {modelVirtualPath}");
        if (!string.IsNullOrWhiteSpace(listfilePath))
            Console.WriteLine($"[AssetProbe] Listfile: {listfilePath}");

        using var dataSource = new MpqDataSource(gamePath, listfilePath);
        byte[]? modelBytes = dataSource.ReadFile(modelVirtualPath) ?? dataSource.ReadFile(modelVirtualPath.Replace('/', '\\'));
        if (modelBytes == null)
            throw new FileNotFoundException($"Model not found in data source: {modelVirtualPath}");

        using var ms = new MemoryStream(modelBytes);
        var mdx = MdxFile.Load(ms);

        Console.WriteLine($"[AssetProbe] Version={mdx.Version} Name={mdx.Model.Name}");
        Console.WriteLine($"[AssetProbe] Textures={mdx.Textures.Count} Materials={mdx.Materials.Count} Geosets={mdx.Geosets.Count}");
        Console.WriteLine();

        for (int i = 0; i < mdx.Textures.Count; i++)
        {
            var texture = mdx.Textures[i];
            Console.WriteLine($"Texture[{i}] ReplaceableId={texture.ReplaceableId} Flags=0x{texture.Flags:X8} Path='{texture.Path}'");

            var probe = ProbeTexture(dataSource, modelVirtualPath, texture);
            if (probe == null)
            {
                Console.WriteLine("  Decode: not found");
            }
            else
            {
                Console.WriteLine($"  ResolvedPath: {probe.Value.ResolvedPath}");
                Console.WriteLine($"  Size: {probe.Value.Width}x{probe.Value.Height}");
                Console.WriteLine($"  Alpha: kind={probe.Value.AlphaKind} zero={probe.Value.ZeroAlphaPixels} full={probe.Value.FullAlphaPixels} translucent={probe.Value.TranslucentAlphaPixels}");
            }
        }

        Console.WriteLine();
        for (int materialIndex = 0; materialIndex < mdx.Materials.Count; materialIndex++)
        {
            var material = mdx.Materials[materialIndex];
            Console.WriteLine($"Material[{materialIndex}] PriorityPlane={material.PriorityPlane} Layers={material.Layers.Count}");
            for (int layerIndex = 0; layerIndex < material.Layers.Count; layerIndex++)
            {
                var layer = material.Layers[layerIndex];
                Console.WriteLine(
                    $"  Layer[{layerIndex}] TextureId={layer.TextureId} Blend={layer.BlendMode} Flags=0x{(uint)layer.Flags:X8} CoordId={layer.CoordId} StaticAlpha={layer.StaticAlpha:F3}");
            }
        }

        Console.WriteLine();
        for (int geosetIndex = 0; geosetIndex < mdx.Geosets.Count; geosetIndex++)
        {
            var geoset = mdx.Geosets[geosetIndex];
            Console.WriteLine(
                $"Geoset[{geosetIndex}] MaterialId={geoset.MaterialId} Vertices={geoset.Vertices.Count} Indices={geoset.Indices.Count} TexCoords={geoset.TexCoords.Count}");
        }
    }

    private static string? TryGetOptionValue(string[] args, string optionName)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i].Equals(optionName, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }

        return null;
    }

    private static TextureProbeResult? ProbeTexture(IDataSource dataSource, string modelVirtualPath, MdlTexture texture)
    {
        foreach (string candidate in EnumerateTextureCandidates(modelVirtualPath, texture))
        {
            byte[]? data = dataSource.ReadFile(candidate);
            if (data == null)
                continue;

            try
            {
                return DecodeTexture(candidate, data);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  DecodeError: {candidate} -> {ex.Message}");
                return null;
            }
        }

        return null;
    }

    private static IEnumerable<string> EnumerateTextureCandidates(string modelVirtualPath, MdlTexture texture)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(texture.Path))
        {
            string direct = texture.Path;
            if (seen.Add(direct.Replace('/', '\\').TrimStart('\\')))
                yield return direct.Replace('/', '\\').TrimStart('\\');

            string fileName = Path.GetFileName(direct);
            if (!string.IsNullOrWhiteSpace(fileName))
            {
                string? modelDir = Path.GetDirectoryName(modelVirtualPath.Replace('/', '\\'));
                if (!string.IsNullOrWhiteSpace(modelDir))
                {
                    string combined = Path.Combine(modelDir, fileName).Replace('/', '\\');
                    if (seen.Add(combined))
                        yield return combined;
                }
            }
        }
    }

    private static TextureProbeResult DecodeTexture(string resolvedPath, byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var blp = new BlpFile(stream);
        using Bitmap bitmap = blp.GetBitmap(0);

        var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
        BitmapData bitmapData = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            byte[] pixels = new byte[bitmapData.Stride * bitmapData.Height];
            Marshal.Copy(bitmapData.Scan0, pixels, 0, pixels.Length);

            int zeroAlpha = 0;
            int fullAlpha = 0;
            int translucentAlpha = 0;

            for (int i = 3; i < pixels.Length; i += 4)
            {
                byte alpha = pixels[i];
                if (alpha == 0)
                    zeroAlpha++;
                else if (alpha == 255)
                    fullAlpha++;
                else
                    translucentAlpha++;
            }

            return new TextureProbeResult(
                resolvedPath,
                bitmap.Width,
                bitmap.Height,
                ClassifyTextureAlpha(zeroAlpha, translucentAlpha),
                zeroAlpha,
                fullAlpha,
                translucentAlpha);
        }
        finally
        {
            bitmap.UnlockBits(bitmapData);
        }
    }

    private static string ClassifyTextureAlpha(int zeroAlphaPixels, int translucentAlphaPixels)
    {
        if (translucentAlphaPixels > 0)
            return "Translucent";

        if (zeroAlphaPixels > 0)
            return "Binary";

        return "Opaque";
    }

    private readonly record struct TextureProbeResult(
        string ResolvedPath,
        int Width,
        int Height,
        string AlphaKind,
        int ZeroAlphaPixels,
        int FullAlphaPixels,
        int TranslucentAlphaPixels);
}