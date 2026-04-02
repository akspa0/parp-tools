using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace MdxViewer.Rendering;

internal static class WowViewerM2RuntimeBridge
{
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