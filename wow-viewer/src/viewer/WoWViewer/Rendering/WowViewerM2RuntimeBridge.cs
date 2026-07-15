using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
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

    /// <summary>
    /// Builds the native runtime representation for WoW 1.0.0's embedded M2 division.
    /// This deliberately does not materialize an MdxFile: the source and draw contract remain M2.
    /// </summary>
    public static M2StaticRenderModel BuildEra100StaticRenderModel(byte[] modelBytes, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        using MemoryStream stream = new(modelBytes, writable: false);
        M2DispatchResult dispatch = M2ModelReaderDispatcher.ReadDetailed(stream, modelPath);
        if (dispatch.Era != M2Era1121EraTag.Md20_1X_V100_Era100)
            throw new InvalidDataException($"M2 '{modelPath}' did not classify as the 1.0.0-era 0x100 layout.");

        M2Era100Geometry geometry = dispatch.Document.InlineEra100Geometry
            ?? throw new InvalidDataException($"1.0.0 M2 '{modelPath}' did not contain an embedded render division.");

        List<M2StaticRenderSection> sections = [];
        List<M2StructuredRenderSection> structuredSections = [];
        for (int sectionIndex = 0; sectionIndex < geometry.Sections.Count; sectionIndex++)
        {
            M2Era100Section sourceSection = geometry.Sections[sectionIndex];
            if (!TryBuildEra100Section(geometry, sourceSection, out List<M2StaticRenderVertex> vertices, out List<uint> indices))
                continue;

            M2Era100Batch[] batches = geometry.Batches
                .Where(batch => batch.SkinSectionIndex == sectionIndex)
                .OrderBy(batch => batch.MaterialLayer)
                .ToArray();
            if (batches.Length == 0)
                batches = [new M2Era100Batch(0, 0, 0, (ushort)sectionIndex, sourceSection.SubmeshId, 0, 0, 0, 0, 0, 0, 0, 0)];

            List<M2StructuredRenderPass> passes = [];
            foreach (M2Era100Batch batch in batches)
            {
                M2StaticRenderMaterial material = BuildEra100Material(geometry, batch);
                passes.Add(new M2StructuredRenderPass(passes.Count, material));
                sections.Add(new M2StaticRenderSection(
                    sectionIndex,
                    sourceSection.SubmeshId,
                    boneComboIndex: 0,
                    boneCount: 0,
                    boneInfluences: 0,
                    centerBoneIndex: 0,
                    vertices,
                    indices,
                    material));
            }

            structuredSections.Add(new M2StructuredRenderSection(
                sectionIndex,
                sourceSection.SubmeshId,
                boneComboIndex: 0,
                boneCount: 0,
                boneInfluences: 0,
                centerBoneIndex: 0,
                vertices,
                indices,
                passes));
        }

        if (sections.Count == 0)
            throw new InvalidDataException($"1.0.0 M2 '{modelPath}' contained no drawable embedded sections.");

        return new M2StaticRenderModel(dispatch.Document, sections, structuredSections, [], usesCompatibilityFallback: false);
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

    private static bool TryBuildEra100Section(
        M2Era100Geometry geometry,
        M2Era100Section sourceSection,
        out List<M2StaticRenderVertex> vertices,
        out List<uint> indices)
    {
        vertices = [];
        indices = [];

        if (sourceSection.IndexStart > int.MaxValue || sourceSection.IndexCount > int.MaxValue)
            return false;

        int start = (int)sourceSection.IndexStart;
        int count = (int)sourceSection.IndexCount;
        if (start < 0 || count < 3 || start > geometry.Triangles.Count || count > geometry.Triangles.Count - start)
            return false;

        Dictionary<ushort, uint> remap = [];
        int endExclusive = start + count - (count % 3);
        for (int indexPosition = start; indexPosition < endExclusive; indexPosition++)
        {
            ushort sourceVertexIndex = geometry.Triangles[indexPosition];
            if (sourceVertexIndex >= geometry.RenderVertices.Count)
                return false;

            if (!remap.TryGetValue(sourceVertexIndex, out uint mappedIndex))
            {
                M2Era100Vertex vertex = geometry.RenderVertices[sourceVertexIndex];
                mappedIndex = (uint)vertices.Count;
                remap.Add(sourceVertexIndex, mappedIndex);
                vertices.Add(new M2StaticRenderVertex(
                    vertex.Position,
                    vertex.Normal,
                    vertex.TexCoord0,
                    vertex.TexCoord1,
                    new System.Numerics.Vector4(vertex.BoneIndex0, vertex.BoneIndex1, vertex.BoneIndex2, vertex.BoneIndex3),
                    new System.Numerics.Vector4(vertex.BoneWeight0 / 255f, vertex.BoneWeight1 / 255f, vertex.BoneWeight2 / 255f, vertex.BoneWeight3 / 255f)));
            }

            indices.Add(mappedIndex);
        }

        return vertices.Count > 0 && indices.Count >= 3;
    }

    private static M2StaticRenderMaterial BuildEra100Material(M2Era100Geometry geometry, M2Era100Batch batch)
    {
        List<M2StaticRenderTextureBinding> bindings = [];
        for (int stage = 0; stage < batch.TextureCount; stage++)
        {
            int lookupIndex = batch.TextureComboIndex + stage;
            short lookupValue = lookupIndex >= 0 && lookupIndex < geometry.TextureLookup.Count
                ? geometry.TextureLookup[lookupIndex]
                : (short)-1;

            // A negative textureCombos entry is a replaceable texture supplied at runtime
            // (character/creature skin), not an index into the model's texture array — the client
            // resolves slot ~value through its own table (FUN_0071a540). We have no runtime skin to
            // bind, so carry the slot as the replaceable id and leave the path unresolved rather
            // than letting the out-of-range index collapse every batch onto one fallback texture.
            M2Era100Texture? texture = null;
            ushort? textureId = null;
            uint replaceableId;
            if (lookupValue >= 0)
            {
                textureId = (ushort)lookupValue;
                texture = lookupValue < geometry.Textures.Count ? geometry.Textures[lookupValue] : null;
                replaceableId = texture?.Type ?? 0;
            }
            else
            {
                replaceableId = (uint)~lookupValue + 1;
            }

            bindings.Add(new M2StaticRenderTextureBinding(
                stage, lookupIndex, textureId, texture?.Filename, replaceableId, texture?.Flags ?? 0,
                null, null, null, null, null, null));
        }

        M2Era100Material material = batch.MaterialIndex < geometry.Materials.Count
            ? geometry.Materials[batch.MaterialIndex]
            : default;
        M2BlendMode blendMode = MapEra100BlendMode(material.BlendMode);

        M2StaticRenderTextureBinding? primary = bindings.FirstOrDefault();
        M2EffectRecipe recipe = new(
            bindings.Count > 0 ? M2DiffuseEffectFamily.T1 : M2DiffuseEffectFamily.None,
            blendMode == M2BlendMode.Opaque ? M2CombinerEffectFamily.Opaque : M2CombinerEffectFamily.Mod,
            isProjected: false,
            usesColorAnimation: false,
            usesTransparencyAnimation: false,
            usesTextureTransformAnimation: false,
            suppressCombinedTransparency: false,
            isHeuristic: true);
        return new M2StaticRenderMaterial(
            batchIndex: batch.SkinSectionIndex,
            batch.Flags,
            batch.PriorityPlane,
            batch.ShaderId,
            batch.GeosetIndex,
            (short)batch.ColorIndex,
            batch.MaterialIndex,
            batch.MaterialLayer,
            batch.TextureCount,
            batch.TextureComboIndex,
            batch.TextureCoordComboIndex,
            batch.TextureWeightComboIndex,
            batch.TextureTransformComboIndex,
            material.Flags,
            material.BlendMode,
            blendMode,
            primary?.TexturePath,
            primary?.ReplaceableId ?? 0,
            primary?.TextureFlags ?? 0,
            bindings,
            recipe);
    }

    /// <summary>
    /// M2Material.blendMode (header 0x84) → renderer blend mode. Values per FUN_0071a910 /
    /// FUN_0071a150: 0 opaque, 1 alpha-key, 2 alpha, 3 and 4 additive, 5 mod, 6 mod2x.
    /// </summary>
    private static M2BlendMode MapEra100BlendMode(ushort rawBlendMode) => rawBlendMode switch
    {
        0 => M2BlendMode.Opaque,
        1 => M2BlendMode.AlphaKey,
        2 => M2BlendMode.AlphaBlend,
        3 => M2BlendMode.NoAlphaAdd,
        4 => M2BlendMode.Add,
        5 => M2BlendMode.Mod,
        6 => M2BlendMode.Mod2X,
        _ => M2BlendMode.Opaque,
    };
}
