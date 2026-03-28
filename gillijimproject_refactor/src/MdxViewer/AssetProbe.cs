using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using WowViewer.Core.Blp;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Mdx;
using CoreBlpCompressionType = WowViewer.Core.Blp.BlpCompressionType;
using CoreBlpPixelFormat = WowViewer.Core.Blp.BlpPixelFormat;
using CoreMdxChunkSummary = WowViewer.Core.Mdx.MdxChunkSummary;
using SereniaBlpFile = SereniaBLPLib.BlpFile;

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
        WowFileDetection modelDetection = WowFileDetector.Detect(ms, modelVirtualPath);
        ms.Position = 0;
        MdxSharedProbeResult? sharedMdxSummary = null;
        string? sharedMdxError = null;
        if (modelDetection.Kind == WowFileKind.Mdx)
            sharedMdxSummary = TryReadSharedMdxSummary(modelVirtualPath, modelBytes, out sharedMdxError);

        var mdx = MdxFile.Load(ms);

        Console.WriteLine($"[AssetProbe] SharedDetect kind={modelDetection.Kind} version={FormatVersion(modelDetection.Version)}");
        if (sharedMdxSummary is MdxSharedProbeResult sharedMdx)
        {
            Console.WriteLine($"[AssetProbe] SharedMDX: version={FormatVersion(sharedMdx.Version)} model={sharedMdx.ModelName ?? "n/a"} blendTime={FormatVersion(sharedMdx.BlendTime)} chunks={sharedMdx.ChunkCount} knownChunks={sharedMdx.KnownChunkCount} unknownChunks={sharedMdx.UnknownChunkCount} textures={sharedMdx.TextureCount} replaceableTextures={sharedMdx.ReplaceableTextureCount} materials={sharedMdx.MaterialCount} materialLayers={sharedMdx.MaterialLayerCount} firstChunks={FormatChunkList(sharedMdx.Chunks)} firstTextures={FormatTextureList(sharedMdx.TexturePaths)} firstMaterials={FormatMaterialList(sharedMdx.MaterialLayers)}");
        }
        else if (!string.IsNullOrWhiteSpace(sharedMdxError))
        {
            Console.WriteLine($"[AssetProbe] SharedMDXError: {sharedMdxError}");
        }

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
                Console.WriteLine($"  SharedDetect: kind={probe.Value.DetectedKind} version={FormatVersion(probe.Value.DetectedVersion)}");
                if (probe.Value.SharedBlpSummary is BlpSharedProbeResult sharedBlp)
                {
                    Console.WriteLine(
                        $"  SharedBLP: format={sharedBlp.Signature} version={FormatVersion(sharedBlp.Version)} compression={sharedBlp.Compression} alphaBits={sharedBlp.AlphaDepthBits} pixelFormat={sharedBlp.PixelFormat} size={sharedBlp.Width}x{sharedBlp.Height} mips={sharedBlp.MipCount} inBoundsMips={sharedBlp.InBoundsMipLevelCount} outOfBoundsMips={sharedBlp.OutOfBoundsMipLevelCount}");
                }
                else if (!string.IsNullOrWhiteSpace(probe.Value.SharedBlpError))
                {
                    Console.WriteLine($"  SharedBLPError: {probe.Value.SharedBlpError}");
                }

                if (probe.Value.DecodeSucceeded)
                {
                    Console.WriteLine($"  Size: {probe.Value.Width}x{probe.Value.Height}");
                    Console.WriteLine($"  Alpha: kind={probe.Value.AlphaKind} zero={probe.Value.ZeroAlphaPixels} full={probe.Value.FullAlphaPixels} translucent={probe.Value.TranslucentAlphaPixels}");
                }
                else
                {
                    Console.WriteLine($"  DecodeError: {probe.Value.DecodeError}");
                }
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

            WowFileDetection detection = DetectFile(candidate, data);
            BlpSharedProbeResult? sharedBlpSummary = null;
            string? sharedBlpError = null;
            if (detection.Kind == WowFileKind.Blp)
                sharedBlpSummary = TryReadSharedBlpSummary(candidate, data, out sharedBlpError);

            try
            {
                return DecodeTexture(candidate, data, detection.Kind, detection.Version, sharedBlpSummary, sharedBlpError);
            }
            catch (Exception ex)
            {
                return new TextureProbeResult(
                    candidate,
                    detection.Kind,
                    detection.Version,
                    sharedBlpSummary,
                    sharedBlpError,
                    0,
                    0,
                    "n/a",
                    0,
                    0,
                    0,
                    false,
                    ex.Message);
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

    private static TextureProbeResult DecodeTexture(
        string resolvedPath,
        byte[] data,
        WowFileKind detectedKind,
        uint? detectedVersion,
        BlpSharedProbeResult? sharedBlpSummary,
        string? sharedBlpError)
    {
        using var stream = new MemoryStream(data);
        using var blp = new SereniaBlpFile(stream);
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
                detectedKind,
                detectedVersion,
                sharedBlpSummary,
                sharedBlpError,
                bitmap.Width,
                bitmap.Height,
                ClassifyTextureAlpha(zeroAlpha, translucentAlpha),
                zeroAlpha,
                fullAlpha,
                translucentAlpha,
                true,
                null);
        }
        finally
        {
            bitmap.UnlockBits(bitmapData);
        }
    }

    private static WowFileDetection DetectFile(string path, byte[] data)
    {
        using var stream = new MemoryStream(data, writable: false);
        return WowFileDetector.Detect(stream, path);
    }

    private static BlpSharedProbeResult? TryReadSharedBlpSummary(string resolvedPath, byte[] data, out string? error)
    {
        try
        {
            using var stream = new MemoryStream(data, writable: false);
            BlpSummary summary = BlpSummaryReader.Read(stream, resolvedPath);
            error = null;
            return new BlpSharedProbeResult(
                summary.Signature,
                summary.Version,
                summary.Compression,
                summary.AlphaDepthBits,
                summary.PixelFormat,
                summary.Width,
                summary.Height,
                summary.MipMaps.Count,
                summary.InBoundsMipLevelCount,
                summary.OutOfBoundsMipLevelCount);
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return null;
        }
    }

    private static MdxSharedProbeResult? TryReadSharedMdxSummary(string resolvedPath, byte[] data, out string? error)
    {
        try
        {
            using var stream = new MemoryStream(data, writable: false);
            MdxSummary summary = MdxSummaryReader.Read(stream, resolvedPath);
            error = null;
            return new MdxSharedProbeResult(
                summary.Version,
                summary.ModelName,
                summary.BlendTime,
                summary.ChunkCount,
                summary.KnownChunkCount,
                summary.UnknownChunkCount,
                summary.TextureCount,
                summary.ReplaceableTextureCount,
                summary.Textures.Take(2).Select(static texture => string.IsNullOrWhiteSpace(texture.Path) ? $"Replaceable#{texture.ReplaceableId}" : texture.Path!).ToArray(),
                summary.MaterialCount,
                summary.MaterialLayerCount,
                summary.Materials.Take(2)
                    .SelectMany(static material => material.Layers.Take(1).Select(layer => $"tex{layer.TextureId}/blend{layer.BlendMode}/alpha{layer.StaticAlpha:F3}"))
                    .ToArray(),
                summary.Chunks.Take(4).Select(static chunk => chunk.Id.ToString()).ToArray());
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return null;
        }
    }

    private static string FormatVersion(uint? version)
    {
        return version?.ToString() ?? "n/a";
    }

    private static string FormatChunkList(IReadOnlyList<string> chunkIds)
    {
        return chunkIds.Count == 0 ? "n/a" : string.Join(",", chunkIds);
    }

    private static string FormatTextureList(IReadOnlyList<string> texturePaths)
    {
        return texturePaths.Count == 0 ? "n/a" : string.Join(",", texturePaths);
    }

    private static string FormatMaterialList(IReadOnlyList<string> materialLayers)
    {
        return materialLayers.Count == 0 ? "n/a" : string.Join(",", materialLayers);
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
        WowFileKind DetectedKind,
        uint? DetectedVersion,
        BlpSharedProbeResult? SharedBlpSummary,
        string? SharedBlpError,
        int Width,
        int Height,
        string AlphaKind,
        int ZeroAlphaPixels,
        int FullAlphaPixels,
        int TranslucentAlphaPixels,
        bool DecodeSucceeded,
        string? DecodeError);

    private readonly record struct BlpSharedProbeResult(
        string Signature,
        uint? Version,
        CoreBlpCompressionType Compression,
        byte AlphaDepthBits,
        CoreBlpPixelFormat PixelFormat,
        int Width,
        int Height,
        int MipCount,
        int InBoundsMipLevelCount,
        int OutOfBoundsMipLevelCount);

    private readonly record struct MdxSharedProbeResult(
        uint? Version,
        string? ModelName,
        uint? BlendTime,
        int ChunkCount,
        int KnownChunkCount,
        int UnknownChunkCount,
        int TextureCount,
        int ReplaceableTextureCount,
        IReadOnlyList<string> TexturePaths,
        int MaterialCount,
        int MaterialLayerCount,
        IReadOnlyList<string> MaterialLayers,
        IReadOnlyList<string> Chunks);
}