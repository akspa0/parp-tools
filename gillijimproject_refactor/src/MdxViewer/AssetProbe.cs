using System.Drawing;
using System.Drawing.Imaging;
using System.Numerics;
using System.Runtime.InteropServices;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using MdxViewer.Terrain;
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
        if (probeIndex >= 0)
        {
            if (args.Length <= probeIndex + 2)
            {
                Console.Error.WriteLine("Usage: MdxViewer --probe-mdx <gamePath> <modelVirtualPath> [--listfile <path>]");
                Environment.ExitCode = 1;
                return true;
            }

            string mdxGamePath = args[probeIndex + 1];
            string mdxModelVirtualPath = args[probeIndex + 2];
            string? mdxListfilePath = TryGetOptionValue(args, "--listfile");

            ViewerLog.Verbose = true;
            MdxFile.Verbose = true;

            try
            {
                Run(mdxGamePath, mdxModelVirtualPath, mdxListfilePath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AssetProbe] Failed: {ex}");
                Environment.ExitCode = 1;
            }

            return true;
        }

        int m2ProbeIndex = Array.FindIndex(args,
            arg => arg.Equals("--probe-m2-adapter", StringComparison.OrdinalIgnoreCase)
                || arg.Equals("--probe-m2", StringComparison.OrdinalIgnoreCase));
        int m2RuntimeProbeIndex = Array.FindIndex(args,
            arg => arg.Equals("--probe-m2-runtime", StringComparison.OrdinalIgnoreCase));

        if (m2RuntimeProbeIndex >= 0)
        {
            if (args.Length <= m2RuntimeProbeIndex + 2)
            {
                Console.Error.WriteLine("Usage: MdxViewer --probe-m2-runtime <gamePath> <modelVirtualPath> [--build <version>] [--skin <virtualPath>] [--listfile <path>]");
                Environment.ExitCode = 1;
                return true;
            }

            string runtimeGamePath = args[m2RuntimeProbeIndex + 1];
            string runtimeModelVirtualPath = args[m2RuntimeProbeIndex + 2];
            string? runtimeBuildVersion = TryGetOptionValue(args, "--build") ?? "3.3.5.12340";
            string? runtimeSkinOverride = TryGetOptionValue(args, "--skin");
            string? runtimeListfilePath = TryGetOptionValue(args, "--listfile");

            ViewerLog.Verbose = true;

            try
            {
                RunM2RuntimeProbe(runtimeGamePath, runtimeModelVirtualPath, runtimeListfilePath, runtimeBuildVersion, runtimeSkinOverride);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AssetProbe:M2Runtime] Failed: {ex}");
                Environment.ExitCode = 1;
            }

            return true;
        }

        if (m2ProbeIndex < 0)
            return false;

        if (args.Length <= m2ProbeIndex + 2)
        {
            Console.Error.WriteLine("Usage: MdxViewer --probe-m2-adapter <gamePath> <modelVirtualPath> [--build <version>] [--skin <virtualPath>] [--listfile <path>]");
            Environment.ExitCode = 1;
            return true;
        }

        string gamePath = args[m2ProbeIndex + 1];
        string modelVirtualPath = args[m2ProbeIndex + 2];
        string? buildVersion = TryGetOptionValue(args, "--build") ?? "3.3.5.12340";
        string? skinOverride = TryGetOptionValue(args, "--skin");
        string? listfilePath = TryGetOptionValue(args, "--listfile");

        ViewerLog.Verbose = true;
        MdxFile.Verbose = true;

        try
        {
            RunM2AdapterProbe(gamePath, modelVirtualPath, listfilePath, buildVersion, skinOverride);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[AssetProbe:M2] Failed: {ex}");
            Environment.ExitCode = 1;
        }

        return true;
    }

    private static void RunM2AdapterProbe(string gamePath, string modelVirtualPath, string? listfilePath, string buildVersion, string? skinOverride)
    {
        string normalizedModelPath = modelVirtualPath.Replace('/', '\\').TrimStart('\\');

        Console.WriteLine($"[M2-ADAPT-PROBE] Game path: {gamePath}");
        Console.WriteLine($"[M2-ADAPT-PROBE] Model path: {normalizedModelPath}");
        Console.WriteLine($"[M2-ADAPT-PROBE] Build: {buildVersion}");
        if (!string.IsNullOrWhiteSpace(skinOverride))
            Console.WriteLine($"[M2-ADAPT-PROBE] Skin override: {skinOverride}");
        if (!string.IsNullOrWhiteSpace(listfilePath))
            Console.WriteLine($"[M2-ADAPT-PROBE] Listfile: {listfilePath}");

        using var dataSource = new MpqDataSource(gamePath, listfilePath);
        byte[]? modelBytes = dataSource.ReadFile(normalizedModelPath)
            ?? dataSource.ReadFile(normalizedModelPath.Replace('\\', '/'));
        if (modelBytes == null)
            throw new FileNotFoundException($"Model not found in data source: {normalizedModelPath}");

        if (!WarcraftNetM2Adapter.IsMd20(modelBytes) && !WarcraftNetM2Adapter.IsMd21(modelBytes))
            throw new InvalidDataException($"Model '{Path.GetFileName(normalizedModelPath)}' is not an M2-family container (MD20/MD21).");

        M2Profile? profile = FormatProfileRegistry.ResolveModelProfile(buildVersion);
        if (profile == null)
            throw new InvalidOperationException($"No M2 profile is registered for build '{buildVersion}'.");

        WarcraftNetM2Adapter.ValidateModelProfile(modelBytes, normalizedModelPath, profile, buildVersion);

        List<string> skinCandidates = string.IsNullOrWhiteSpace(skinOverride)
            ? WarcraftNetM2Adapter.BuildSkinCandidates(normalizedModelPath).Distinct(StringComparer.OrdinalIgnoreCase).ToList()
            : new List<string> { skinOverride.Replace('/', '\\').TrimStart('\\') };

        int missingSkinCount = 0;
        int failedSkinCount = 0;
        Exception? lastSkinFailure = null;

        foreach (string skinPath in skinCandidates)
        {
            byte[]? skinBytes = dataSource.ReadFile(skinPath)
                ?? dataSource.ReadFile(skinPath.Replace('\\', '/'));
            if (skinBytes == null)
            {
                missingSkinCount++;
                continue;
            }

            Console.WriteLine($"[M2-ADAPT-PROBE] Trying skin: {skinPath} ({skinBytes.Length} bytes)");
            try
            {
                var runtimeModel = WarcraftNetM2Adapter.BuildRuntimeModel(modelBytes, skinBytes, normalizedModelPath, buildVersion);
                Console.WriteLine($"[M2-ADAPT-PROBE] Selected skin: {skinPath}");
                PrintRendererEquivalentDiagnostics(runtimeModel, normalizedModelPath, skinPath);
                return;
            }
            catch (Exception ex)
            {
                failedSkinCount++;
                lastSkinFailure = ex;
                Console.WriteLine($"[M2-ADAPT-PROBE] Skin candidate failed: {skinPath} ({ex.Message})");
            }
        }

        if (string.Equals(profile.ProfileId, FormatProfileRegistry.M2Profile3018303.ProfileId, StringComparison.Ordinal))
        {
            Console.WriteLine("[M2-ADAPT-PROBE] No external .skin resolved; trying embedded root-profile fallback.");
            var runtimeModel = WarcraftNetM2Adapter.BuildRuntimeModel(modelBytes, null, normalizedModelPath, buildVersion);
            PrintRendererEquivalentDiagnostics(runtimeModel, normalizedModelPath, "<embedded-root-profile>");
            return;
        }

        throw new InvalidDataException(
            $"No usable .skin for {Path.GetFileName(normalizedModelPath)}. candidates={skinCandidates.Count}, missing={missingSkinCount}, failed={failedSkinCount}.",
            lastSkinFailure);
    }

    private static void RunM2RuntimeProbe(string gamePath, string modelVirtualPath, string? listfilePath, string buildVersion, string? skinOverride)
    {
        string normalizedModelPath = modelVirtualPath.Replace('/', '\\').TrimStart('\\');

        Console.WriteLine($"[M2-RUNTIME-PROBE] Game path: {gamePath}");
        Console.WriteLine($"[M2-RUNTIME-PROBE] Model path: {normalizedModelPath}");
        Console.WriteLine($"[M2-RUNTIME-PROBE] Build: {buildVersion}");
        if (!string.IsNullOrWhiteSpace(skinOverride))
            Console.WriteLine($"[M2-RUNTIME-PROBE] Skin override: {skinOverride}");
        if (!string.IsNullOrWhiteSpace(listfilePath))
            Console.WriteLine($"[M2-RUNTIME-PROBE] Listfile: {listfilePath}");

        using var dataSource = new MpqDataSource(gamePath, listfilePath);
        byte[]? modelBytes = dataSource.ReadFile(normalizedModelPath)
            ?? dataSource.ReadFile(normalizedModelPath.Replace('\\', '/'));
        if (modelBytes == null)
            throw new FileNotFoundException($"Model not found in data source: {normalizedModelPath}");

        M2Profile? profile = FormatProfileRegistry.ResolveModelProfile(buildVersion);
        if (profile == null)
            throw new InvalidOperationException($"No M2 profile is registered for build '{buildVersion}'.");

        WarcraftNetM2Adapter.ValidateModelProfile(modelBytes, normalizedModelPath, profile, buildVersion);

        List<string> skinCandidates = string.IsNullOrWhiteSpace(skinOverride)
            ? WarcraftNetM2Adapter.BuildSkinCandidates(normalizedModelPath).Distinct(StringComparer.OrdinalIgnoreCase).ToList()
            : new List<string> { skinOverride.Replace('/', '\\').TrimStart('\\') };

        int missingSkinCount = 0;
        int failedSkinCount = 0;
        Exception? lastSkinFailure = null;

        foreach (string skinPath in skinCandidates)
        {
            byte[]? skinBytes = dataSource.ReadFile(skinPath)
                ?? dataSource.ReadFile(skinPath.Replace('\\', '/'));
            if (skinBytes == null)
            {
                missingSkinCount++;
                continue;
            }

            Console.WriteLine($"[M2-RUNTIME-PROBE] Trying skin: {skinPath} ({skinBytes.Length} bytes)");
            try
            {
                var runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(modelBytes, skinBytes, normalizedModelPath, skinPath);
                int vertexCount = runtimeModel.Sections.Sum(static section => section.Vertices.Count);
                int triangleCount = runtimeModel.Sections.Sum(static section => section.Indices.Count / 3);
                int transparentSectionCount = runtimeModel.Sections.Count(static section => section.Material.IsTransparent);
                Console.WriteLine($"[M2-RUNTIME-PROBE] Selected skin: {skinPath}");
                Console.WriteLine($"[M2-RUNTIME-PROBE] sections={runtimeModel.Sections.Count} transparentSections={transparentSectionCount} vertices={vertexCount} triangles={triangleCount} boundsMin={runtimeModel.BoundsMin} boundsMax={runtimeModel.BoundsMax}");
                return;
            }
            catch (Exception ex)
            {
                failedSkinCount++;
                lastSkinFailure = ex;
                Console.WriteLine($"[M2-RUNTIME-PROBE] Skin candidate failed: {skinPath} ({ex.Message})");
            }
        }

        throw new InvalidDataException(
            $"No usable .skin for runtime probe {Path.GetFileName(normalizedModelPath)}. candidates={skinCandidates.Count}, missing={missingSkinCount}, failed={failedSkinCount}.",
            lastSkinFailure);
    }

    private static void PrintRendererEquivalentDiagnostics(MdxFile runtimeModel, string modelPath, string selectedSkinPath)
    {
        int totalGeosets = runtimeModel.Geosets.Count;
        int validGeosets = 0;
        int indexRejected = 0;
        int emptySkipped = 0;

        for (int i = 0; i < runtimeModel.Geosets.Count; i++)
        {
            var geoset = runtimeModel.Geosets[i];
            int vertCount = geoset.Vertices.Count;
            int indexCount = geoset.Indices.Count;

            if (vertCount == 0 || indexCount == 0)
            {
                emptySkipped++;
                continue;
            }

            int maxIndex = geoset.Indices.Max(static idx => (int)idx);
            if (maxIndex >= vertCount)
            {
                indexRejected++;
                Console.WriteLine(
                    $"[M2-DIAG-CPU] Geoset {i} would be rejected by renderer index validation (maxIndex={maxIndex}, vertCount={vertCount}, indexCount={indexCount})");
                continue;
            }

            validGeosets++;
        }

        Console.WriteLine(
            $"[M2-DIAG-CPU] {modelPath}: {totalGeosets} geosets, {validGeosets} valid, {indexRejected} index-rejected, {emptySkipped} empty-skipped (skin={selectedSkinPath})");
        Console.WriteLine($"[M2-DIAG-CPU] {WarcraftNetM2Adapter.SummarizeGeometry(runtimeModel)}");

        int maxGeosetSummaries = Math.Min(runtimeModel.Geosets.Count, 12);
        for (int geosetIndex = 0; geosetIndex < maxGeosetSummaries; geosetIndex++)
        {
            var geoset = runtimeModel.Geosets[geosetIndex];
            string materialSummary = DescribeProbeMaterial(runtimeModel, geoset.MaterialId);
            Console.WriteLine(
                $"[M2-DIAG-MAT] geoset={geosetIndex} material={geoset.MaterialId} verts={geoset.Vertices.Count} tris={geoset.Indices.Count / 3} layers={materialSummary}");
        }
    }

    private static string DescribeProbeMaterial(MdxFile runtimeModel, int materialId)
    {
        if (materialId < 0 || materialId >= runtimeModel.Materials.Count)
            return "<invalid-material>";

        var material = runtimeModel.Materials[materialId];
        if (material.Layers.Count == 0)
            return "<no-layers>";

        return string.Join(", ",
            material.Layers.Select((layer, index) =>
            {
                string texturePath = layer.TextureId >= 0 && layer.TextureId < runtimeModel.Textures.Count
                    ? runtimeModel.Textures[layer.TextureId].Path
                    : "<missing>";
                return $"[{index}]tex={layer.TextureId}:{texturePath} blend={layer.BlendMode} coord={layer.CoordId} flags={layer.Flags}";
            }));
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
        MdxGeometryProbeResult? sharedMdxGeometry = null;
        string? sharedMdxError = null;
        string? sharedMdxGeometryError = null;
        if (modelDetection.Kind == WowFileKind.Mdx)
        {
            sharedMdxSummary = TryReadSharedMdxSummary(modelVirtualPath, modelBytes, out sharedMdxError);
            sharedMdxGeometry = TryReadSharedMdxGeometry(modelVirtualPath, modelBytes, out sharedMdxGeometryError);
        }

        var mdx = MdxFile.Load(ms);

        Console.WriteLine($"[AssetProbe] SharedDetect kind={modelDetection.Kind} version={FormatVersion(modelDetection.Version)}");
        if (sharedMdxSummary is MdxSharedProbeResult sharedMdx)
        {
            Console.WriteLine($"[AssetProbe] SharedMDX: version={FormatVersion(sharedMdx.Version)} model={sharedMdx.ModelName ?? "n/a"} blendTime={FormatVersion(sharedMdx.BlendTime)} chunks={sharedMdx.ChunkCount} knownChunks={sharedMdx.KnownChunkCount} unknownChunks={sharedMdx.UnknownChunkCount} textures={sharedMdx.TextureCount} replaceableTextures={sharedMdx.ReplaceableTextureCount} materials={sharedMdx.MaterialCount} materialLayers={sharedMdx.MaterialLayerCount} firstChunks={FormatChunkList(sharedMdx.Chunks)} firstTextures={FormatTextureList(sharedMdx.TexturePaths)} firstMaterials={FormatMaterialList(sharedMdx.MaterialLayers)}");

            if (sharedMdx.PivotPointCount > 0)
                Console.WriteLine($"[AssetProbe] SharedPIVT: count={sharedMdx.PivotPointCount} first={FormatVector(sharedMdx.FirstPivotPoint)}");

            if (sharedMdx.CollisionVertexCount.HasValue && sharedMdx.CollisionTriangleCount.HasValue)
            {
                Console.WriteLine(
                    $"[AssetProbe] SharedCLID: vertices={sharedMdx.CollisionVertexCount.Value} triangles={sharedMdx.CollisionTriangleCount.Value} bounds={FormatBounds(sharedMdx.CollisionBoundsMin, sharedMdx.CollisionBoundsMax)}");
            }
        }
        else if (!string.IsNullOrWhiteSpace(sharedMdxError))
        {
            Console.WriteLine($"[AssetProbe] SharedMDXError: {sharedMdxError}");
        }

        if (!string.IsNullOrWhiteSpace(sharedMdxGeometryError))
            Console.WriteLine($"[AssetProbe] SharedMDXGeometryError: {sharedMdxGeometryError}");

        Console.WriteLine($"[AssetProbe] Version={mdx.Version} Name={mdx.Model.Name}");
        Console.WriteLine($"[AssetProbe] Textures={mdx.Textures.Count} Materials={mdx.Materials.Count} Geosets={(sharedMdxGeometry?.GeosetCount ?? mdx.Geosets.Count)}");
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
        if (sharedMdxGeometry is MdxGeometryProbeResult geometryProbe)
        {
            for (int geosetIndex = 0; geosetIndex < geometryProbe.Geosets.Count; geosetIndex++)
            {
                MdxGeosetGeometry geoset = geometryProbe.Geosets[geosetIndex];
                Console.WriteLine(
                    $"Geoset[{geosetIndex}] MaterialId={geoset.MaterialId} Vertices={geoset.VertexCount} Indices={geoset.IndexCount} Triangles={geoset.TriangleCount} UvSets={geoset.UvSetCount} PrimaryTexCoords={geoset.PrimaryUvCount} MatrixGroups={geoset.MatrixGroupCount} MatrixIndices={geoset.MatrixIndexCount}");
            }
        }
        else
        {
            for (int geosetIndex = 0; geosetIndex < mdx.Geosets.Count; geosetIndex++)
            {
                var geoset = mdx.Geosets[geosetIndex];
                Console.WriteLine(
                    $"Geoset[{geosetIndex}] MaterialId={geoset.MaterialId} Vertices={geoset.Vertices.Count} Indices={geoset.Indices.Count} TexCoords={geoset.TexCoords.Count}");
            }
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
                summary.PivotPointCount,
                summary.PivotPoints.FirstOrDefault()?.Position,
                summary.Collision?.VertexCount,
                summary.Collision?.TriangleCount,
                summary.Collision?.BoundsMin,
                summary.Collision?.BoundsMax,
                summary.Chunks.Take(4).Select(static chunk => chunk.Id.ToString()).ToArray());
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return null;
        }
    }

    private static MdxGeometryProbeResult? TryReadSharedMdxGeometry(string resolvedPath, byte[] data, out string? error)
    {
        try
        {
            using var stream = new MemoryStream(data, writable: false);
            MdxGeometryFile geometry = MdxGeometryReader.Read(stream, resolvedPath);
            error = null;
            return new MdxGeometryProbeResult(
                geometry.GeosetCount,
                geometry.Geosets.ToArray());
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

    private static string FormatVector(Vector3? vector)
    {
        if (!vector.HasValue)
            return "n/a";

        Vector3 value = vector.Value;
        return $"({value.X:F3},{value.Y:F3},{value.Z:F3})";
    }

    private static string FormatBounds(Vector3? min, Vector3? max)
    {
        return $"{FormatVector(min)}..{FormatVector(max)}";
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
        int PivotPointCount,
        Vector3? FirstPivotPoint,
        int? CollisionVertexCount,
        int? CollisionTriangleCount,
        Vector3? CollisionBoundsMin,
        Vector3? CollisionBoundsMax,
        IReadOnlyList<string> Chunks);

    private readonly record struct MdxGeometryProbeResult(
        int GeosetCount,
        IReadOnlyList<MdxGeosetGeometry> Geosets);
}