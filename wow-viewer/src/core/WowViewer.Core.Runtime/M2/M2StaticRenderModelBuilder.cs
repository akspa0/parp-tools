using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public static class M2StaticRenderModelBuilder
{
    public static M2StaticRenderModel Build(M2GeometryDocument geometry, M2SkinProfileRuntimeState state)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(state);

        if (state.Stage != M2SkinProfileStage.Initialized || state.ActiveSkinProfile is null)
            throw new InvalidOperationException("Cannot build a static M2 render model before the active skin profile is initialized.");

        return Build(geometry, state.ActiveSkinProfile);
    }

    public static M2StaticRenderModel Build(M2GeometryDocument geometry, M2ActiveSkinProfile activeSkinProfile)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(activeSkinProfile);

        M2SkinDocument skin = activeSkinProfile.Skin;
        List<M2StaticRenderSection> compatibilitySections = new(activeSkinProfile.ActiveSections.Count);
        List<M2StructuredRenderSection> structuredSections = new(activeSkinProfile.ActiveSections.Count);

        foreach (M2ActiveSkinSection activeSection in activeSkinProfile.ActiveSections)
        {
            if (activeSection.IndexCount < 3)
                continue;

            List<M2StaticRenderVertex> vertices = new();
            List<uint> indices = new();
            Dictionary<ushort, uint> remap = new();

            int start = activeSection.IndexStart;
            int endExclusive = Math.Min(skin.TriangleIndices.Count, start + activeSection.IndexCount);
            if (start < 0 || start >= endExclusive)
                continue;

            for (int indexPosition = start; indexPosition < endExclusive; indexPosition++)
            {
                ushort localSkinVertexIndex = skin.TriangleIndices[indexPosition];
                if (!TryGetVertex(geometry, skin, localSkinVertexIndex, out M2GeometryVertex vertex))
                    continue;

                if (!remap.TryGetValue(localSkinVertexIndex, out uint mappedIndex))
                {
                    mappedIndex = (uint)vertices.Count;
                    remap.Add(localSkinVertexIndex, mappedIndex);
                    vertices.Add(new M2StaticRenderVertex(
                        vertex.Position,
                        vertex.Normal,
                        vertex.TextureCoords0,
                        vertex.BoneIndices,
                        vertex.BoneWeights));
                }

                indices.Add(mappedIndex);
            }

            int trimmedCount = indices.Count - (indices.Count % 3);
            if (trimmedCount != indices.Count)
                indices.RemoveRange(trimmedCount, indices.Count - trimmedCount);

            if (vertices.Count == 0 || indices.Count < 3)
                continue;

            IReadOnlyList<M2ActiveSkinBatch> orderedBatches = activeSection.Batches
                .OrderBy(static value => value.MaterialLayer)
                .ThenBy(static value => value.BatchIndex)
                .ToArray();

            List<M2StructuredRenderPass> passes = BuildStructuredPasses(geometry, orderedBatches);
            structuredSections.Add(new M2StructuredRenderSection(
                activeSection.SectionIndex,
                activeSection.SkinSectionId,
                vertices,
                indices,
                passes));

            foreach (M2StructuredRenderPass pass in passes)
            {
                compatibilitySections.Add(new M2StaticRenderSection(
                    activeSection.SectionIndex,
                    activeSection.SkinSectionId,
                    vertices,
                    indices,
                    pass.Material));
            }
        }

        return new M2StaticRenderModel(geometry.Model, compatibilitySections, structuredSections, activeSkinProfile.UsesCompatibilityFallback);
    }

    private static List<M2StructuredRenderPass> BuildStructuredPasses(M2GeometryDocument geometry, IReadOnlyList<M2ActiveSkinBatch> orderedBatches)
    {
        List<M2StructuredRenderPass> passes = new(Math.Max(orderedBatches.Count, 1));

        if (orderedBatches.Count == 0)
        {
            passes.Add(new M2StructuredRenderPass(0, BuildMaterial(geometry, batch: null)));
            return passes;
        }

        for (int passIndex = 0; passIndex < orderedBatches.Count; passIndex++)
            passes.Add(new M2StructuredRenderPass(passIndex, BuildMaterial(geometry, orderedBatches[passIndex])));

        return passes;
    }

    private static bool TryGetVertex(M2GeometryDocument geometry, M2SkinDocument skin, ushort localSkinVertexIndex, out M2GeometryVertex vertex)
    {
        vertex = default;

        if (localSkinVertexIndex >= skin.VertexLookup.Count)
            return false;

        int globalIndex = skin.VertexLookup[localSkinVertexIndex] + (int)skin.GlobalVertexOffset;
        if (globalIndex < 0 || globalIndex >= geometry.Vertices.Count)
        {
            globalIndex = skin.VertexLookup[localSkinVertexIndex];
            if (globalIndex < 0 || globalIndex >= geometry.Vertices.Count)
                return false;
        }

        vertex = geometry.Vertices[globalIndex];
        return true;
    }

    private static M2StaticRenderMaterial BuildMaterial(M2GeometryDocument geometry, M2ActiveSkinBatch? batch)
    {
        int batchIndex = batch?.BatchIndex ?? -1;
        byte batchFlags = batch?.Flags ?? 0;
        byte priorityPlane = batch?.PriorityPlane ?? 0;
        ushort shaderId = batch?.ShaderId ?? 0;
        ushort geosetIndex = batch?.GeosetIndex ?? 0;
        short colorIndex = batch?.ColorIndex ?? (short)-1;
        ushort renderFlagsIndex = batch?.RenderFlagsIndex ?? 0;
        ushort materialLayer = batch?.MaterialLayer ?? 0;
        ushort textureCount = batch?.TextureCount ?? 0;
        ushort textureComboIndex = batch?.TextureComboIndex ?? 0;
        ushort textureCoordComboIndex = batch?.TextureCoordComboIndex ?? 0;
        ushort transparencyComboIndex = batch?.TransparencyComboIndex ?? 0;
        ushort textureAnimationLookupIndex = batch?.TextureAnimationLookupIndex ?? 0;

        ushort renderFlags = 0;
        ushort rawBlendMode = 0;
        M2BlendMode blendMode = M2BlendMode.Opaque;
        if (renderFlagsIndex != ushort.MaxValue && renderFlagsIndex < geometry.RenderFlags.Count)
        {
            M2GeometryRenderFlag renderFlag = geometry.RenderFlags[renderFlagsIndex];
            renderFlags = renderFlag.Flags;
            rawBlendMode = renderFlag.RawBlendMode;
            blendMode = renderFlag.BlendMode;
        }

        List<M2StaticRenderTextureBinding> textureBindings = BuildTextureBindings(geometry, batch);
        string? texturePath = null;
        uint replaceableId = 0;
        uint textureFlags = 0;
        M2StaticRenderTextureBinding? primaryBinding = textureBindings.FirstOrDefault(static value => value.TextureId.HasValue || !string.IsNullOrWhiteSpace(value.TexturePath) || value.ReplaceableId != 0);
        if (primaryBinding is not null)
        {
            texturePath = primaryBinding.TexturePath;
            replaceableId = primaryBinding.ReplaceableId;
            textureFlags = primaryBinding.TextureFlags;
        }

        M2EffectRecipe effectRecipe = BuildEffectRecipe(batch, textureBindings, blendMode, renderFlags, colorIndex);

        return new M2StaticRenderMaterial(
            batchIndex,
            batchFlags,
            priorityPlane,
            shaderId,
            geosetIndex,
            colorIndex,
            renderFlagsIndex,
            materialLayer,
            textureCount,
            textureComboIndex,
            textureCoordComboIndex,
            transparencyComboIndex,
            textureAnimationLookupIndex,
            renderFlags,
            rawBlendMode,
            blendMode,
            texturePath,
            replaceableId,
            textureFlags,
                textureBindings,
                effectRecipe);
    }

    private static List<M2StaticRenderTextureBinding> BuildTextureBindings(M2GeometryDocument geometry, M2ActiveSkinBatch? batch)
    {
        if (batch is null || batch.TextureCount == 0)
            return [];

        List<M2StaticRenderTextureBinding> bindings = new(batch.TextureCount);
        for (int stageIndex = 0; stageIndex < batch.TextureCount; stageIndex++)
        {
            int? textureLookupIndex = TryResolveLookupIndex(batch.TextureComboIndex, stageIndex, geometry.TextureLookup.Count);
            ushort? textureId = null;
            string? texturePath = null;
            uint replaceableId = 0;
            uint textureFlags = 0;
            if (textureLookupIndex is int textureLookupOffset)
            {
                ushort candidateTextureId = geometry.TextureLookup[textureLookupOffset].TextureId;
                textureId = candidateTextureId;
                if (candidateTextureId < geometry.Textures.Count)
                {
                    M2GeometryTexture texture = geometry.Textures[candidateTextureId];
                    texturePath = texture.Filename;
                    replaceableId = texture.ReplaceableId;
                    textureFlags = texture.Flags;
                }
            }

            int? textureCoordLookupIndex = TryResolveLookupIndex(batch.TextureCoordComboIndex, stageIndex, geometry.TextureUnitLookup.Count);
            ushort? textureCoordLookupValue = textureCoordLookupIndex is int textureCoordLookupOffset
                ? geometry.TextureUnitLookup[textureCoordLookupOffset].BatchIndex
                : null;

            int? transparencyLookupIndex = TryResolveLookupIndex(batch.TransparencyComboIndex, stageIndex, geometry.TransparencyLookup.Count);
            ushort? transparencyLookupValue = transparencyLookupIndex is int transparencyLookupOffset
                ? geometry.TransparencyLookup[transparencyLookupOffset].TransparencyIndex
                : null;

            int? textureAnimationLookupIndex = TryResolveLookupIndex(batch.TextureAnimationLookupIndex, stageIndex, geometry.TextureAnimationLookup.Count);
            ushort? textureAnimationLookupValue = textureAnimationLookupIndex is int textureAnimationLookupOffset
                ? geometry.TextureAnimationLookup[textureAnimationLookupOffset].TextureAnimationIndex
                : null;

            bindings.Add(new M2StaticRenderTextureBinding(
                stageIndex,
                textureLookupIndex,
                textureId,
                texturePath,
                replaceableId,
                textureFlags,
                textureCoordLookupIndex,
                textureCoordLookupValue,
                transparencyLookupIndex,
                transparencyLookupValue,
                textureAnimationLookupIndex,
                textureAnimationLookupValue));
        }

        return bindings;
    }

    private static int? TryResolveLookupIndex(ushort comboIndex, int stageIndex, int lookupCount)
    {
        if (comboIndex == ushort.MaxValue)
            return null;

        int resolvedIndex = comboIndex + stageIndex;
        if (resolvedIndex < 0 || resolvedIndex >= lookupCount)
            return null;

        return resolvedIndex;
    }

    private static M2EffectRecipe BuildEffectRecipe(
        M2ActiveSkinBatch? batch,
        IReadOnlyList<M2StaticRenderTextureBinding> textureBindings,
        M2BlendMode blendMode,
        ushort renderFlags,
        short colorIndex)
    {
        int textureStageCount = Math.Max(textureBindings.Count, batch?.TextureCount ?? 0);
        bool isProjected = batch is not null
            && (((batch.Flags & 0x4) != 0) || ((batch.GeosetIndex & 0x2) != 0));
        M2DiffuseEffectFamily diffuseFamily = isProjected
            ? M2DiffuseEffectFamily.Projected
            : textureStageCount switch
            {
                <= 0 => M2DiffuseEffectFamily.None,
                1 => M2DiffuseEffectFamily.T1,
                2 => M2DiffuseEffectFamily.T1T2,
                3 => M2DiffuseEffectFamily.T1T2T3,
                _ => M2DiffuseEffectFamily.T1T2T3T4,
            };

        M2CombinerEffectFamily combinerFamily = blendMode switch
        {
            M2BlendMode.Opaque => M2CombinerEffectFamily.Opaque,
            M2BlendMode.AlphaKey => M2CombinerEffectFamily.AlphaKey,
            M2BlendMode.AlphaBlend => M2CombinerEffectFamily.Decal,
            M2BlendMode.NoAlphaAdd => M2CombinerEffectFamily.Add,
            M2BlendMode.Add => M2CombinerEffectFamily.Add,
            M2BlendMode.Mod => M2CombinerEffectFamily.Mod,
            M2BlendMode.Mod2X => M2CombinerEffectFamily.Mod2X,
            M2BlendMode.BlendAdd => M2CombinerEffectFamily.Fade,
            _ => M2CombinerEffectFamily.Unknown,
        };

        bool usesTransparencyAnimation = textureBindings.Any(static binding => binding.TransparencyLookupValue is ushort value && value != ushort.MaxValue);
        bool usesTextureTransformAnimation = (batch is not null && (batch.Flags & 0x2) != 0)
            || textureBindings.Any(static binding => binding.TextureAnimationLookupValue is ushort value && value != ushort.MaxValue);
        bool usesColorAnimation = colorIndex >= 0;
        bool suppressCombinedTransparency = (batch is not null && (batch.Flags & 0x40) != 0)
            || (renderFlags & 0x40) != 0;

        return new M2EffectRecipe(
            diffuseFamily,
            combinerFamily,
            isProjected,
            usesColorAnimation,
            usesTransparencyAnimation,
            usesTextureTransformAnimation,
            suppressCombinedTransparency,
            isHeuristic: true);
    }
}