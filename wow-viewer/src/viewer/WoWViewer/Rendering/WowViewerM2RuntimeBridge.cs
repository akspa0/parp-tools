using MdxLTool.Formats.Mdx;
using WoWViewer.DataSources;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

internal static class WowViewerM2RuntimeBridge
{
    private const string NativeRendererSettingName = "PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER";

    public static M2StaticRenderModel BuildStaticRenderModel(byte[] modelBytes, byte[] skinBytes, string modelPath, string skinPath)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        ArgumentNullException.ThrowIfNull(skinBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(skinPath);

        using MemoryStream modelStream = new(modelBytes, writable: false);
        M2GeometryDocument geometry = M2GeometryReader.Read(modelStream, modelPath);

        using MemoryStream skinStream = new(skinBytes, writable: false);
        M2SkinDocument skin = M2SkinReader.Read(skinStream, skinPath.Replace('/', '\\'));

        int profileIndex = GuessProfileIndex(geometry.Model.Identity.CanonicalModelPath, skin.SourcePath);
        M2SkinProfileSelection selection = new(profileIndex, skin.SourcePath);
        M2SkinProfileRuntimeState chosen = new(geometry.Model, selection, M2SkinProfileStage.Chosen, loadedSkin: null, activeSkinProfile: null);
        M2SkinProfileRuntimeState loaded = M2SkinProfileRuntime.Load(chosen, skin);
        M2SkinProfileRuntimeState initialized = M2SkinProfileRuntime.Initialize(loaded);

        return M2StaticRenderModelBuilder.Build(geometry, initialized);
    }

    public static bool PreferNativeStaticRenderer
    {
        get
        {
            string? value = Environment.GetEnvironmentVariable(NativeRendererSettingName);
            if (string.IsNullOrWhiteSpace(value))
                return true;

            if (string.Equals(value, "0", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "false", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "no", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "off", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            return string.Equals(value, "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "yes", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "on", StringComparison.OrdinalIgnoreCase);
        }
    }

    public static bool ShouldUseNativeStaticRenderer(MdxFile? adaptedMdx)
        => adaptedMdx == null || PreferNativeStaticRenderer;

    public static M2Renderer CreateRenderer(
        GL gl,
        M2StaticRenderModel runtimeModel,
        MdxFile? adaptedMdx,
        string? modelDir,
        IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver,
        string? buildVersion,
        string sourceModelPath,
        bool deferInitialTextureLoads = false)
    {
        ArgumentNullException.ThrowIfNull(gl);
        ArgumentNullException.ThrowIfNull(runtimeModel);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        if (ShouldUseNativeStaticRenderer(adaptedMdx))
            return new M2Renderer(gl, runtimeModel, sourceModelPath, dataSource, texResolver);

        string resolvedModelDir = modelDir ?? Path.GetDirectoryName(sourceModelPath) ?? string.Empty;
        return new M2Renderer(
            new MdxRenderer(gl, adaptedMdx!, resolvedModelDir, dataSource, texResolver, sourceModelPath, true, buildVersion, deferInitialTextureLoads: deferInitialTextureLoads),
            runtimeModel,
            sourceModelPath);
    }

    private static int GuessProfileIndex(string modelPath, string skinPath)
    {
        string modelBaseName = Path.GetFileNameWithoutExtension(modelPath);
        string skinBaseName = Path.GetFileNameWithoutExtension(skinPath);
        if (skinBaseName.StartsWith(modelBaseName, StringComparison.OrdinalIgnoreCase))
        {
            string suffix = skinBaseName[modelBaseName.Length..];
            if (suffix.Length == 2
                && char.IsDigit(suffix[0])
                && char.IsDigit(suffix[1])
                && int.TryParse(suffix, out int parsedIndex))
            {
                return parsedIndex;
            }
        }

        return 0;
    }
}
