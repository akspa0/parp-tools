using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2RenderFrame
{
    public M2RenderFrame(
        string canonicalModelPath,
        int timeMs,
        IReadOnlyList<M2RenderDrawCommand> drawCommands,
        string frameHash)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(canonicalModelPath);
        ArgumentNullException.ThrowIfNull(drawCommands);
        ArgumentException.ThrowIfNullOrWhiteSpace(frameHash);

        CanonicalModelPath = canonicalModelPath;
        TimeMs = timeMs;
        DrawCommands = drawCommands;
        FrameHash = frameHash;
    }

    public string CanonicalModelPath { get; }

    public int TimeMs { get; }

    public IReadOnlyList<M2RenderDrawCommand> DrawCommands { get; }

    public string FrameHash { get; }

    public int CommandCount => DrawCommands.Count;

    public int SubmittedVertexCount => DrawCommands.Sum(static command => command.SubmittedVertexCount);

    public int SubmittedIndexCount => DrawCommands.Sum(static command => command.SubmittedIndexCount);

    public int BackendVertexCount => DrawCommands.Sum(static command => command.Vertices.Count);

    public int BackendIndexCount => DrawCommands.Sum(static command => command.Indices.Count);
}

public sealed class M2RenderDrawCommand
{
    public M2RenderDrawCommand(
        int batchIndex,
        M2RenderEntryFamily family,
        string handlerName,
        bool isDirect,
        bool usesDedicatedStateScope,
        string modelKey,
        string effectKey,
        int textureSortKey,
        int stateBucket,
        IReadOnlyList<string> entryKeys,
        int submittedVertexCount,
        int submittedIndexCount,
        IReadOnlyList<M2RenderBackendVertex> vertices,
        IReadOnlyList<uint> indices,
        IReadOnlyList<M2RenderConsumerTextureState> textures,
        Vector3 diffuseColor,
        Vector3 emissiveColor,
        float alpha,
        bool receivesLighting,
        M2BlendMode blendMode,
        bool depthWrite,
        bool alphaTest,
        bool isTransparent,
        bool isAdditive,
        bool isTwoSided,
        string? effectObjectKey,
        string? nativeEffectFamilyKey)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(handlerName);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);
        ArgumentNullException.ThrowIfNull(entryKeys);
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(textures);
        ArgumentOutOfRangeException.ThrowIfNegative(submittedVertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(submittedIndexCount);

        BatchIndex = batchIndex;
        Family = family;
        HandlerName = handlerName;
        IsDirect = isDirect;
        UsesDedicatedStateScope = usesDedicatedStateScope;
        ModelKey = modelKey;
        EffectKey = effectKey;
        TextureSortKey = textureSortKey;
        StateBucket = stateBucket;
        EntryKeys = entryKeys;
        SubmittedVertexCount = submittedVertexCount;
        SubmittedIndexCount = submittedIndexCount;
        Vertices = vertices;
        Indices = indices;
        Textures = textures;
        DiffuseColor = diffuseColor;
        EmissiveColor = emissiveColor;
        Alpha = alpha;
        ReceivesLighting = receivesLighting;
        BlendMode = blendMode;
        DepthWrite = depthWrite;
        AlphaTest = alphaTest;
        IsTransparent = isTransparent;
        IsAdditive = isAdditive;
        IsTwoSided = isTwoSided;
        EffectObjectKey = effectObjectKey;
        NativeEffectFamilyKey = nativeEffectFamilyKey;
    }

    public int BatchIndex { get; }

    public M2RenderEntryFamily Family { get; }

    public string HandlerName { get; }

    public bool IsDirect { get; }

    public bool UsesDedicatedStateScope { get; }

    public string ModelKey { get; }

    public string EffectKey { get; }

    public int TextureSortKey { get; }

    public int StateBucket { get; }

    public IReadOnlyList<string> EntryKeys { get; }

    public int SubmittedVertexCount { get; }

    public int SubmittedIndexCount { get; }

    public IReadOnlyList<M2RenderBackendVertex> Vertices { get; }

    public IReadOnlyList<uint> Indices { get; }

    public IReadOnlyList<M2RenderConsumerTextureState> Textures { get; }

    public Vector3 DiffuseColor { get; }

    public Vector3 EmissiveColor { get; }

    public float Alpha { get; }

    public bool ReceivesLighting { get; }

    public M2BlendMode BlendMode { get; }

    public bool DepthWrite { get; }

    public bool AlphaTest { get; }

    public bool IsTransparent { get; }

    public bool IsAdditive { get; }

    public bool IsTwoSided { get; }

    public string? EffectObjectKey { get; }

    public string? NativeEffectFamilyKey { get; }
}

public readonly record struct M2RenderBackendVertex(
    Vector3 Position,
    Vector3 Normal,
    Vector2 TextureCoords);

public static class M2RenderFrameBuilder
{
    public static M2RenderFrame Build(
        M2StaticRenderModel renderModel,
        M2SkinnedRenderModel? skinnedRenderModel,
        M2RenderConsumerFrameState consumerState,
        M2SceneSubmissionPlan submissionPlan,
        int timeMs)
    {
        ArgumentNullException.ThrowIfNull(renderModel);
        ArgumentNullException.ThrowIfNull(consumerState);
        ArgumentNullException.ThrowIfNull(submissionPlan);

        Dictionary<int, M2StructuredRenderSection> staticSections = renderModel.StructuredSections.ToDictionary(static section => section.SectionIndex);
        Dictionary<int, M2SkinnedRenderSection> skinnedSections = skinnedRenderModel?.Sections.ToDictionary(static section => section.Source.SectionIndex) ?? [];
        Dictionary<(int SectionIndex, int PassIndex, int BatchIndex), M2RenderConsumerPassState> passesByFullKey = consumerState.Passes.ToDictionary(static pass => (pass.AnimatedPass.SectionIndex, pass.AnimatedPass.PassIndex, pass.AnimatedPass.BatchIndex));
        Dictionary<(int SectionIndex, int PassIndex), M2RenderConsumerPassState> passesBySectionPass = consumerState.Passes
            .GroupBy(static pass => (pass.AnimatedPass.SectionIndex, pass.AnimatedPass.PassIndex))
            .ToDictionary(static group => group.Key, static group => group.First());

        List<M2RenderDrawCommand> commands = new(submissionPlan.Batches.Count);
        foreach (M2SceneSubmissionBatch batch in submissionPlan.Batches)
        {
            List<M2RenderBackendVertex> vertices = new(batch.VertexCount);
            List<uint> indices = new(batch.IndexCount);
            M2RenderConsumerPassState? firstPass = null;

            foreach (M2SceneSubmissionEntry entry in batch.Entries)
            {
                if (entry.SectionIndex is not int sectionIndex)
                    continue;

                if (!TryGetSectionVertices(staticSections, skinnedSections, sectionIndex, out IReadOnlyList<M2RenderBackendVertex>? sectionVertices, out IReadOnlyList<uint>? sectionIndices))
                    continue;

                uint baseVertex = checked((uint)vertices.Count);
                vertices.AddRange(sectionVertices);
                foreach (uint index in sectionIndices)
                    indices.Add(baseVertex + index);

                if (entry.PassIndex is int passIndex && entry.BatchIndex is int batchEntryIndex)
                {
                    firstPass ??= passesByFullKey.GetValueOrDefault((sectionIndex, passIndex, batchEntryIndex))
                        ?? passesBySectionPass.GetValueOrDefault((sectionIndex, passIndex));
                }
            }

            M2ResolvedEffect? resolvedEffect = firstPass?.ResolvedEffect;
            commands.Add(new M2RenderDrawCommand(
                batch.BatchIndex,
                batch.Family,
                batch.HandlerName,
                batch.IsDirect,
                batch.UsesDedicatedStateScope,
                batch.ModelKey,
                batch.EffectKey,
                batch.TextureSortKey,
                batch.StateBucket,
                batch.Entries.Select(static entry => entry.EntryKey).ToArray(),
                batch.VertexCount,
                batch.IndexCount,
                vertices,
                indices,
                firstPass?.Textures ?? [],
                firstPass?.DiffuseColor ?? Vector3.One,
                firstPass?.EmissiveColor ?? Vector3.Zero,
                firstPass?.Alpha ?? 1.0f,
                firstPass?.ReceivesLighting ?? true,
                firstPass?.SourcePass.Material.BlendMode ?? M2BlendMode.Opaque,
                resolvedEffect?.DepthWrite ?? true,
                resolvedEffect?.AlphaTest ?? false,
                resolvedEffect?.IsTransparent ?? false,
                resolvedEffect?.IsAdditive ?? false,
                resolvedEffect?.IsTwoSided ?? false,
                resolvedEffect?.EffectObjectKey,
                resolvedEffect?.NativeEffectFamilyKey));
        }

        string hash = ComputeHash(renderModel.Model.Identity.CanonicalModelPath, timeMs, commands);
        return new M2RenderFrame(renderModel.Model.Identity.CanonicalModelPath, timeMs, commands, hash);
    }

    private static bool TryGetSectionVertices(
        IReadOnlyDictionary<int, M2StructuredRenderSection> staticSections,
        IReadOnlyDictionary<int, M2SkinnedRenderSection> skinnedSections,
        int sectionIndex,
        out IReadOnlyList<M2RenderBackendVertex> vertices,
        out IReadOnlyList<uint> indices)
    {
        vertices = [];
        indices = [];

        if (skinnedSections.TryGetValue(sectionIndex, out M2SkinnedRenderSection? skinned))
        {
            vertices = skinned.Vertices
                .Select(static vertex => new M2RenderBackendVertex(vertex.Position, vertex.Normal, vertex.TextureCoords))
                .ToArray();
            indices = skinned.Source.Indices;
            return true;
        }

        if (staticSections.TryGetValue(sectionIndex, out M2StructuredRenderSection? section))
        {
            vertices = section.Vertices
                .Select(static vertex => new M2RenderBackendVertex(vertex.Position, vertex.Normal, vertex.TextureCoords0))
                .ToArray();
            indices = section.Indices;
            return true;
        }

        return false;
    }

    private static string ComputeHash(string canonicalModelPath, int timeMs, IReadOnlyList<M2RenderDrawCommand> commands)
    {
        StringBuilder builder = new();
        builder.Append(canonicalModelPath).Append('|').Append(timeMs.ToString(CultureInfo.InvariantCulture));
        foreach (M2RenderDrawCommand command in commands)
        {
            builder.Append("|command:")
                .Append(command.BatchIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(command.Family).Append(',')
                .Append(command.HandlerName).Append(',')
                .Append(command.IsDirect).Append(',')
                .Append(command.EffectKey).Append(',')
                .Append(command.TextureSortKey.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(command.StateBucket.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(command.SubmittedVertexCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(command.SubmittedIndexCount.ToString(CultureInfo.InvariantCulture));

            foreach (M2RenderBackendVertex vertex in command.Vertices)
            {
                builder.Append("|v:")
                    .Append(FormatVector(vertex.Position)).Append(',')
                    .Append(FormatVector(vertex.Normal)).Append(',')
                    .Append(FormatVector2(vertex.TextureCoords));
            }

            foreach (uint index in command.Indices)
                builder.Append("|i:").Append(index.ToString(CultureInfo.InvariantCulture));
        }

        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string FormatVector(Vector3 value)
    {
        return string.Create(CultureInfo.InvariantCulture, $"{Round(value.X):F4},{Round(value.Y):F4},{Round(value.Z):F4}");
    }

    private static string FormatVector2(Vector2 value)
    {
        return string.Create(CultureInfo.InvariantCulture, $"{Round(value.X):F4},{Round(value.Y):F4}");
    }

    private static float Round(float value)
    {
        return MathF.Round(value, 4, MidpointRounding.AwayFromZero);
    }
}
