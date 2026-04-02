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
        List<M2StaticRenderSection> sections = new(activeSkinProfile.ActiveSections.Count);

        foreach (M2ActiveSkinSection activeSection in activeSkinProfile.ActiveSections)
        {
            if (activeSection.IndexCount < 3)
                continue;

            M2ActiveSkinBatch? batch = activeSection.Batches.Count > 0 ? activeSection.Batches[0] : null;
            M2StaticRenderMaterial material = BuildMaterial(geometry, batch);
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

            sections.Add(new M2StaticRenderSection(
                activeSection.SectionIndex,
                activeSection.SkinSectionId,
                vertices,
                indices,
                material));
        }

        return new M2StaticRenderModel(geometry.Model, sections, activeSkinProfile.UsesCompatibilityFallback);
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
        short colorIndex = batch?.ColorIndex ?? (short)-1;
        ushort materialIndex = batch?.MaterialIndex ?? 0;
        ushort textureComboIndex = batch?.TextureComboIndex ?? 0;
        ushort textureCoordComboIndex = batch?.TextureCoordComboIndex ?? 0;
        ushort transparencyComboIndex = batch?.TransparencyComboIndex ?? 0;
        ushort textureAnimationLookupIndex = batch?.TextureAnimationLookupIndex ?? 0;

        ushort renderFlags = 0;
        ushort rawBlendMode = 0;
        M2BlendMode blendMode = M2BlendMode.Opaque;
        if (materialIndex < geometry.RenderFlags.Count)
        {
            M2GeometryRenderFlag renderFlag = geometry.RenderFlags[materialIndex];
            renderFlags = renderFlag.Flags;
            rawBlendMode = renderFlag.RawBlendMode;
            blendMode = renderFlag.BlendMode;
        }

        string? texturePath = null;
        uint replaceableId = 0;
        uint textureFlags = 0;
        if (textureComboIndex < geometry.TextureLookup.Count)
        {
            ushort textureId = geometry.TextureLookup[textureComboIndex].TextureId;
            if (textureId < geometry.Textures.Count)
            {
                M2GeometryTexture texture = geometry.Textures[textureId];
                texturePath = texture.Filename;
                replaceableId = texture.ReplaceableId;
                textureFlags = texture.Flags;
            }
        }

        return new M2StaticRenderMaterial(
            batchIndex,
            batchFlags,
            priorityPlane,
            colorIndex,
            materialIndex,
            textureComboIndex,
            textureCoordComboIndex,
            transparencyComboIndex,
            textureAnimationLookupIndex,
            renderFlags,
            rawBlendMode,
            blendMode,
            texturePath,
            replaceableId,
            textureFlags);
    }
}