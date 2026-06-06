using System.Reflection;
using System.Runtime.ExceptionServices;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class M2EmbeddedProfileRealDataTests
{
    private static Assembly? s_viewerAssembly;

    [Fact]
    public void IsM2FamilyContainer_UsesRootMagicInsteadOfFilenameConvention()
    {
        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'. Build WoWViewer before running this test.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type adapterType = viewerAssembly.GetType("WoWViewer.Rendering.WarcraftNetM2Adapter", throwOnError: true)!;
        MethodInfo isM2FamilyContainer = adapterType.GetMethod(
            "IsM2FamilyContainer",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;

        byte[] mdlxBytes = [(byte)'M', (byte)'D', (byte)'L', (byte)'X', 0, 0, 0, 0];
        byte[] md20Bytes = [(byte)'M', (byte)'D', (byte)'2', (byte)'0', 0, 0, 0, 0];
        byte[] md21Bytes = [(byte)'M', (byte)'D', (byte)'2', (byte)'1', 0, 0, 0, 0];

        Assert.False((bool)isM2FamilyContainer.Invoke(null, [mdlxBytes])!);
        Assert.True((bool)isM2FamilyContainer.Invoke(null, [md20Bytes])!);
        Assert.True((bool)isM2FamilyContainer.Invoke(null, [md21Bytes])!);
    }

    [Fact]
    public void WorldAssetManager_ClassicPreferredPaths_DoNotJumpToM2First()
    {
        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'. Build WoWViewer before running this test.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type worldAssetManagerType = viewerAssembly.GetType("WoWViewer.Terrain.WorldAssetManager", throwOnError: true)!;
        MethodInfo helper = worldAssetManagerType.GetMethod(
            "EnumeratePreferredClassicModelPaths",
            BindingFlags.Static | BindingFlags.NonPublic)!;

        string[] mdxCandidates = ((IEnumerable<string>)helper.Invoke(null, ["World\\Generic\\Crate.mdx"])!).ToArray();
        string[] mdlCandidates = ((IEnumerable<string>)helper.Invoke(null, ["World\\Generic\\Crate.mdl"])!).ToArray();

        Assert.Equal(["World\\Generic\\Crate.mdx", "World\\Generic\\Crate.mdl"], mdxCandidates);
        Assert.Equal(["World\\Generic\\Crate.mdl", "World\\Generic\\Crate.mdx"], mdlCandidates);
        Assert.DoesNotContain(mdxCandidates, candidate => candidate.EndsWith(".m2", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(mdlCandidates, candidate => candidate.EndsWith(".m2", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void WmoRenderer_ClassicPreferredPaths_DoNotJumpToM2First()
    {
        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'. Build WoWViewer before running this test.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type wmoRendererType = viewerAssembly.GetType("WoWViewer.Rendering.WmoRenderer", throwOnError: true)!;
        MethodInfo helper = wmoRendererType.GetMethod(
            "EnumeratePreferredClassicDoodadPaths",
            BindingFlags.Static | BindingFlags.NonPublic)!;

        string[] mdxCandidates = ((IEnumerable<string>)helper.Invoke(null, ["World\\Generic\\Crate.mdx"])!).ToArray();
        string[] mdlCandidates = ((IEnumerable<string>)helper.Invoke(null, ["World\\Generic\\Crate.mdl"])!).ToArray();

        Assert.Equal(["World\\Generic\\Crate.mdx", "World\\Generic\\Crate.mdl"], mdxCandidates);
        Assert.Equal(["World\\Generic\\Crate.mdl", "World\\Generic\\Crate.mdx"], mdlCandidates);
        Assert.DoesNotContain(mdxCandidates, candidate => candidate.EndsWith(".m2", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(mdlCandidates, candidate => candidate.EndsWith(".m2", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void BuildRuntimeModel_PreRelease301_EmbeddedProfileSample_HasRenderableGeometry()
    {
        string dataPath = Path.Combine(
            GetWowViewerRoot(),
            "..",
            "output",
            "tmp",
            "wowarchive-clients",
            "3_0_1_8303",
            "World of Warcraft",
            "Data");
        if (!Directory.Exists(dataPath))
            return;

        using MpqArchiveCatalog catalog = new();
        catalog.LoadArchives([dataPath]);
        catalog.LoadListfile(AdtRealDataTestCatalog.ListfilePath);
        catalog.LoadListfileEntries(ReadAdditionalListfileEntries());

        (string virtualPath, byte[] modelBytes) = FindPreRelease301ModelSample(catalog);

        Assert.True(modelBytes.Length > 0, "Expected at least one staged 3.0.1.8303 M2 sample to be readable from MPQ archives.");

        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'. Build WoWViewer before running this test.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type adapterType = viewerAssembly.GetType("WoWViewer.Rendering.WarcraftNetM2Adapter", throwOnError: true)!;
        MethodInfo buildRuntimeModel = adapterType.GetMethod(
            "BuildRuntimeModel",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;
        MethodInfo hasRenderableGeometry = adapterType.GetMethod(
            "HasRenderableGeometry",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;
        MethodInfo summarizeGeometry = adapterType.GetMethod(
            "SummarizeGeometry",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;

        object runtimeModel;
        try
        {
            runtimeModel = buildRuntimeModel.Invoke(null, [modelBytes, null, virtualPath, "3.0.1.8303"])!;
        }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }

        bool hasGeometry = (bool)hasRenderableGeometry.Invoke(null, [runtimeModel])!;
        string summary = (string)summarizeGeometry.Invoke(null, [runtimeModel])!;
        Assert.True(hasGeometry, $"{virtualPath} produced no renderable geometry ({summary}).");
    }

    [Fact]
    public void BuildRuntimeModel_PreRelease301_FirstStagedSamples_MatchViewerRouteAndHaveGeometry()
    {
        string dataPath = Path.Combine(
            GetWowViewerRoot(),
            "..",
            "output",
            "tmp",
            "wowarchive-clients",
            "3_0_1_8303",
            "World of Warcraft",
            "Data");
        if (!Directory.Exists(dataPath))
            return;

        using MpqArchiveCatalog catalog = new();
        catalog.LoadArchives([dataPath]);
        catalog.LoadListfile(AdtRealDataTestCatalog.ListfilePath);
        catalog.LoadListfileEntries(ReadAdditionalListfileEntries());

        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'. Build WoWViewer before running this test.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type adapterType = viewerAssembly.GetType("WoWViewer.Rendering.WarcraftNetM2Adapter", throwOnError: true)!;
        MethodInfo buildRuntimeModel = adapterType.GetMethod(
            "BuildRuntimeModel",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;
        MethodInfo hasRenderableGeometry = adapterType.GetMethod(
            "HasRenderableGeometry",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;
        MethodInfo summarizeGeometry = adapterType.GetMethod(
            "SummarizeGeometry",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;
        MethodInfo buildSkinCandidates = adapterType.GetMethod(
            "BuildSkinCandidates",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;

        List<string> failures = [];
        foreach ((string virtualPath, byte[] modelBytes) in EnumeratePreRelease301Samples(catalog, maxSamples: 16))
        {
            object candidateList = buildSkinCandidates.Invoke(null, [virtualPath])!;
            IReadOnlyList<string> skinCandidates = (candidateList as IEnumerable<string>)?.ToArray()
                ?? throw new InvalidOperationException("BuildSkinCandidates did not return an enumerable path list.");

            byte[]? skinBytes = null;
            string routeLabel = "embedded";
            foreach (string skinPath in skinCandidates.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                skinBytes = catalog.ReadFile(skinPath);
                if (skinBytes is { Length: > 0 })
                {
                    routeLabel = skinPath;
                    break;
                }
            }

            try
            {
                object runtimeModel = buildRuntimeModel.Invoke(null, [modelBytes, skinBytes, virtualPath, "3.0.1.8303"])!;
                bool hasGeometry = (bool)hasRenderableGeometry.Invoke(null, [runtimeModel])!;
                string summary = (string)summarizeGeometry.Invoke(null, [runtimeModel])!;
                if (!hasGeometry)
                    failures.Add($"{virtualPath} via {routeLabel}: {summary}");
            }
            catch (TargetInvocationException ex) when (ex.InnerException != null)
            {
                failures.Add($"{virtualPath} via {routeLabel}: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
            }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void ViewerRoute_MissingPrimaryTexture_KeepsPrimaryLayerRenderable()
    {
        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'. Build WoWViewer before running this test.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type rendererType = viewerAssembly.GetType("WoWViewer.Rendering.MdxRenderer", throwOnError: true)!;
        MethodInfo shouldSkipLayerWhenTextureIsMissing = rendererType.GetMethod(
            "ShouldSkipLayerWhenTextureIsMissing",
            BindingFlags.NonPublic | BindingFlags.Static)!;

        bool shouldSkipPrimaryLayer = (bool)shouldSkipLayerWhenTextureIsMissing.Invoke(null, [0])!;
        bool shouldSkipSecondaryLayer = (bool)shouldSkipLayerWhenTextureIsMissing.Invoke(null, [1])!;

        Assert.False(shouldSkipPrimaryLayer);
        Assert.True(shouldSkipSecondaryLayer);
    }

    private static string GetWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the test output directory.");
    }

    private static Assembly LoadViewerAssembly(string viewerAssemblyPath)
    {
        if (s_viewerAssembly != null)
            return s_viewerAssembly;

        string normalizedTargetPath = Path.GetFullPath(viewerAssemblyPath);
        s_viewerAssembly = Assembly.LoadFrom(normalizedTargetPath);
        return s_viewerAssembly;
    }

    private static IEnumerable<string> ReadAdditionalListfileEntries()
    {
        string partsDirectory = Path.Combine(GetWowViewerRoot(), "libs", "wowdev", "wow-listfile", "parts");
        if (!Directory.Exists(partsDirectory))
            yield break;

        foreach (string csvPath in Directory.EnumerateFiles(partsDirectory, "*.csv", SearchOption.TopDirectoryOnly))
        {
            foreach (string line in File.ReadLines(csvPath))
            {
                int separatorIndex = line.IndexOf(';');
                if (separatorIndex < 0 || separatorIndex + 1 >= line.Length)
                    continue;

                string entry = line[(separatorIndex + 1)..].Trim();
                if (!string.IsNullOrWhiteSpace(entry))
                    yield return entry.Replace('/', '\\');
            }
        }
    }

    private static (string VirtualPath, byte[] ModelBytes) FindPreRelease301ModelSample(MpqArchiveCatalog catalog)
    {
        foreach ((string virtualPath, byte[] modelBytes) in EnumeratePreRelease301Samples(catalog, maxSamples: 1))
            return (virtualPath, modelBytes);

        throw new Xunit.Sdk.XunitException("Expected at least one staged 3.0.1-compatible MD20 model to be discoverable from the archive catalog.");
    }

    private static IEnumerable<(string VirtualPath, byte[] ModelBytes)> EnumeratePreRelease301Samples(MpqArchiveCatalog catalog, int maxSamples)
    {
        int yielded = 0;
        foreach (string virtualPath in ReadAdditionalListfileEntries()
                     .Where(static path =>
                         path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)
                         || path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)
                         || path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase)))
        {
            if (!catalog.FileExists(virtualPath))
                continue;

            byte[]? modelBytes = catalog.ReadFile(virtualPath);
            if (modelBytes is not { Length: >= 8 })
                continue;

            uint magic = BitConverter.ToUInt32(modelBytes, 0);
            uint version = BitConverter.ToUInt32(modelBytes, 4);
            if (magic != 0x3032444D)
                continue;

            if (version is >= 0x104 and <= 0x108)
            {
                yield return (virtualPath, modelBytes);
                yielded++;
                if (yielded >= maxSamples)
                    yield break;
            }
        }
    }
}
