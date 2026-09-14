using System.Reflection;
using System.Runtime.ExceptionServices;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class M2EmbeddedProfileRealDataTests
{
    private static Assembly? s_viewerAssembly;

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
    public void BuildEmbeddedStaticRenderModel_SyntheticEra100_TransfersAlphaKeyBlendMode()
    {
        byte[] modelBytes = M2Era100ModelReaderTests.CreateSyntheticEra100M2WithMaterial(blendMode: 1);

        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        if (!File.Exists(viewerAssemblyPath))
        {
            viewerAssemblyPath = Path.Combine(
                GetWowViewerRoot(),
                "src",
                "viewer",
                "WoWViewer",
                "bin",
                "Debug",
                "net10.0",
                "ParpToolsWoWViewer.dll");
        }
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type adapterType = viewerAssembly.GetType("WoWViewer.Rendering.WarcraftNetM2Adapter", throwOnError: true)!;
        MethodInfo buildEmbedded = adapterType.GetMethod(
            "BuildEmbeddedStaticRenderModel",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;

        object renderModel = buildEmbedded.Invoke(null, [modelBytes, "SyntheticAlphaKey.m2", "2.0.0.5610"])!;
        Assert.NotNull(renderModel);

        PropertyInfo sectionsProp = renderModel.GetType().GetProperty("Sections")!;
        System.Collections.IList sections = (System.Collections.IList)sectionsProp.GetValue(renderModel)!;
        Assert.NotEmpty(sections);

        object section0 = sections[0]!;
        PropertyInfo materialProp = section0.GetType().GetProperty("Material")!;
        object material0 = materialProp.GetValue(section0)!;

        PropertyInfo blendModeProp = material0.GetType().GetProperty("BlendMode")!;
        object blendModeVal = blendModeProp.GetValue(material0)!;
        Assert.Equal("AlphaKey", blendModeVal.ToString());

        PropertyInfo isTransparentProp = material0.GetType().GetProperty("IsTransparent")!;
        bool isTransparent = (bool)isTransparentProp.GetValue(material0)!;
        Assert.False(isTransparent, "AlphaKey is alpha cutout (depth-tested), not back-to-front blended transparent pass.");
    }

    [Fact]
    public void BuildEra100StaticRenderModel_TransfersTextureBindingsAndFallbackSlots()
    {
        byte[] modelBytes = M2Era100ModelReaderTests.CreateSyntheticEra100M2WithMaterial(blendMode: 0);

        string viewerAssemblyPath = Path.Combine(
            GetWowViewerRoot(),
            "src",
            "viewer",
            "WoWViewer",
            "bin",
            "Debug",
            "net10.0-windows",
            "ParpToolsWoWViewer.dll");
        if (!File.Exists(viewerAssemblyPath))
        {
            viewerAssemblyPath = Path.Combine(
                GetWowViewerRoot(),
                "src",
                "viewer",
                "WoWViewer",
                "bin",
                "Debug",
                "net10.0",
                "ParpToolsWoWViewer.dll");
        }
        Assert.True(File.Exists(viewerAssemblyPath), $"Expected built viewer assembly at '{viewerAssemblyPath}'.");

        Assembly viewerAssembly = LoadViewerAssembly(viewerAssemblyPath);
        Type bridgeType = viewerAssembly.GetType("WoWViewer.Rendering.WowViewerM2RuntimeBridge", throwOnError: true)!;
        MethodInfo buildEra100 = bridgeType.GetMethod(
            "BuildEra100StaticRenderModel",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)!;

        object renderModel = buildEra100.Invoke(null, [modelBytes, "Creature\\DragonSpawn\\DragonSpawnArmored.mdx"])!;
        Assert.NotNull(renderModel);

        PropertyInfo sectionsProp = renderModel.GetType().GetProperty("Sections")!;
        System.Collections.IList sections = (System.Collections.IList)sectionsProp.GetValue(renderModel)!;
        Assert.NotEmpty(sections);

        object section0 = sections[0]!;
        PropertyInfo materialProp = section0.GetType().GetProperty("Material")!;
        object material0 = materialProp.GetValue(section0)!;

        PropertyInfo textureBindingsProp = material0.GetType().GetProperty("TextureBindings")!;
        System.Collections.IList bindings = (System.Collections.IList)textureBindingsProp.GetValue(material0)!;
        Assert.NotEmpty(bindings);

        PropertyInfo texturePathProp = material0.GetType().GetProperty("TexturePath")!;
        string? texturePath = (string?)texturePathProp.GetValue(material0);
        Assert.False(string.IsNullOrWhiteSpace(texturePath));
    }

    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public M2EmbeddedProfileRealDataTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Inspect200OrcFemaleSequencesAndBones()
    {
        string[] candidates = [
            @"H:\CLIENTS\TBC\2.X_Pre-Release_Windows_enUS_2.0.0.5610\World of Warcraft\Data",
            @"H:\CLIENTS\TBC\2.X_Retail_Windows_enUS_2.4.3.8606\World of Warcraft\Data",
        ];
        string? dataDir = candidates.FirstOrDefault(Directory.Exists);
        if (dataDir == null)
            return;

        using MpqArchiveCatalog catalog = new();
        catalog.LoadArchives([dataDir]);
        byte[]? modelBytes = catalog.ReadFile(@"Character\Orc\Female\OrcFemale.m2");
        Assert.NotNull(modelBytes);

        using MemoryStream stream = new(modelBytes, writable: false);
        var dispatch = WowViewer.Core.IO.M2Chunked.M2ModelReaderDispatcher.ReadDetailed(stream, @"Character\Orc\Female\OrcFemale.m2");
        var doc = dispatch.Document;

        _output.WriteLine($"TOTAL SEQUENCES: {doc.Sequences.Count}");
        for (int i = 0; i < doc.Sequences.Count; i++)
        {
            var seq = doc.Sequences[i];
            var name = WowViewer.Core.M2.M2AnimationNameResolver.GetSequenceDisplayName(seq.AnimationId, seq.VariationIndex);
            _output.WriteLine($"SEQ[{i:D3}]: id={seq.AnimationId} var={seq.VariationIndex} name={name}");
        }

        // Check an animated bone with multiple keyframes in sequence 0
        Assert.True(doc.Sequences.Count > 0);

        var standSeq = doc.Sequences[0];
        Assert.Equal(0, standSeq.AnimationId);
        Assert.Equal(3333u, standSeq.StartTimestamp);
        Assert.Equal(5800u, standSeq.EndTimestamp);
        Assert.Equal(2467u, standSeq.Duration);

        // Verify animation name resolution
        Assert.Equal("Stand", WowViewer.Core.M2.M2AnimationNameResolver.GetSequenceDisplayName(standSeq.AnimationId, standSeq.VariationIndex));
        var lootSeq = doc.Sequences[27];
        Assert.Equal(50, lootSeq.AnimationId);
        Assert.Equal("Loot", WowViewer.Core.M2.M2AnimationNameResolver.GetSequenceDisplayName(lootSeq.AnimationId, lootSeq.VariationIndex));
        var thrownSeq = doc.Sequences[24];
        Assert.Equal(107, thrownSeq.AnimationId);
        Assert.Equal("AttackThrown", WowViewer.Core.M2.M2AnimationNameResolver.GetSequenceDisplayName(thrownSeq.AnimationId, thrownSeq.VariationIndex));

        // Verify animated bone sampling actually advances through timeline and changes values
        var animatedBone = doc.Bones[22];
        var rot0 = WowViewer.Core.Runtime.M2.M2TrackSampler.SampleCompressedQuaternion(doc.RawBytes, doc, 0, 0, animatedBone.RotationTrack, System.Numerics.Quaternion.Identity);
        var rot1 = WowViewer.Core.Runtime.M2.M2TrackSampler.SampleCompressedQuaternion(doc.RawBytes, doc, 0, 1200, animatedBone.RotationTrack, System.Numerics.Quaternion.Identity);
        Assert.NotEqual(rot0, rot1);
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
