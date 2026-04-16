namespace WowViewer.Core.Runtime.M2;

[Flags]
public enum M2RuntimeOptions
{
    None = 0,
    UseZFill = 0x1,
    UseClipPlanes = 0x2,
    UseThreads = 0x4,
    Faster = 0x8,
    BatchDoodads = 0x20,
    BatchParticles = 0x80,
    ForceAdditiveParticleSort = 0x100,
}

public enum M2RenderEntryFamily
{
    Core = 0,
    Projected = 1,
    Doodad = 2,
    Ribbon = 3,
    Particle = 4,
    Callback = 5,
    HitTest = 6,
}

public sealed class M2SceneFamilyPolicy
{
    public M2SceneFamilyPolicy(
        M2RenderEntryFamily family,
        string handlerName,
        M2RuntimeOptions batchingFlag,
        bool alwaysDirect,
        bool usesDedicatedStateScope)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(handlerName);

        Family = family;
        HandlerName = handlerName;
        BatchingFlag = batchingFlag;
        AlwaysDirect = alwaysDirect;
        UsesDedicatedStateScope = usesDedicatedStateScope;
    }

    public M2RenderEntryFamily Family { get; }

    public string HandlerName { get; }

    public M2RuntimeOptions BatchingFlag { get; }

    public bool AlwaysDirect { get; }

    public bool UsesDedicatedStateScope { get; }

    public bool RequiresDirectSubmission(M2RuntimeOptions options)
    {
        return AlwaysDirect || (BatchingFlag != M2RuntimeOptions.None && !options.HasFlag(BatchingFlag));
    }
}

public sealed class M2SceneSubmissionEntry
{
    public M2SceneSubmissionEntry(
        string entryKey,
        string modelKey,
        M2RenderEntryFamily family,
        string effectKey,
        int textureSortKey,
        int stateBucket,
        int vertexCount,
        int indexCount,
        float depthSortValue = 0.0f,
        bool isTransparent = false,
        bool isAdditive = false,
        bool forceDirect = false,
        int? sectionIndex = null,
        int? passIndex = null,
        int? batchIndex = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(entryKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(indexCount);

        EntryKey = entryKey;
        ModelKey = modelKey;
        Family = family;
        EffectKey = effectKey;
        TextureSortKey = textureSortKey;
        StateBucket = stateBucket;
        VertexCount = vertexCount;
        IndexCount = indexCount;
        DepthSortValue = depthSortValue;
        IsTransparent = isTransparent;
        IsAdditive = isAdditive;
        ForceDirect = forceDirect;
        SectionIndex = sectionIndex;
        PassIndex = passIndex;
        BatchIndex = batchIndex;
    }

    public string EntryKey { get; }

    public string ModelKey { get; }

    public M2RenderEntryFamily Family { get; }

    public string EffectKey { get; }

    public int TextureSortKey { get; }

    public int StateBucket { get; }

    public int VertexCount { get; }

    public int IndexCount { get; }

    public float DepthSortValue { get; }

    public bool IsTransparent { get; }

    public bool IsAdditive { get; }

    public bool ForceDirect { get; }

    public int? SectionIndex { get; }

    public int? PassIndex { get; }

    public int? BatchIndex { get; }
}

public sealed class M2SceneBatchLimits
{
    public M2SceneBatchLimits(int maxVertices = 65535, int maxIndices = 98304)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxVertices);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIndices);

        MaxVertices = maxVertices;
        MaxIndices = maxIndices;
    }

    public int MaxVertices { get; }

    public int MaxIndices { get; }
}

public sealed class M2SceneSubmissionPlan
{
    public M2SceneSubmissionPlan(
        M2RuntimeOptions options,
        IReadOnlyList<M2SceneSubmissionBatch> batches,
        int directEntryCount,
        int batchedEntryCount)
    {
        ArgumentNullException.ThrowIfNull(batches);

        Options = options;
        Batches = batches;
        DirectEntryCount = directEntryCount;
        BatchedEntryCount = batchedEntryCount;
    }

    public M2RuntimeOptions Options { get; }

    public IReadOnlyList<M2SceneSubmissionBatch> Batches { get; }

    public int DirectEntryCount { get; }

    public int BatchedEntryCount { get; }
}

public sealed class M2SceneSubmissionBatch
{
    public M2SceneSubmissionBatch(
        int batchIndex,
        M2RenderEntryFamily family,
        string modelKey,
        string effectKey,
        int textureSortKey,
        int stateBucket,
        bool isDirect,
        string handlerName,
        bool usesDedicatedStateScope,
        IReadOnlyList<M2SceneSubmissionEntry> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(handlerName);
        ArgumentNullException.ThrowIfNull(entries);

        BatchIndex = batchIndex;
        Family = family;
        ModelKey = modelKey;
        EffectKey = effectKey;
        TextureSortKey = textureSortKey;
        StateBucket = stateBucket;
        IsDirect = isDirect;
        HandlerName = handlerName;
        UsesDedicatedStateScope = usesDedicatedStateScope;
        Entries = entries;
    }

    public int BatchIndex { get; }

    public M2RenderEntryFamily Family { get; }

    public string ModelKey { get; }

    public string EffectKey { get; }

    public int TextureSortKey { get; }

    public int StateBucket { get; }

    public bool IsDirect { get; }

    public string HandlerName { get; }

    public bool UsesDedicatedStateScope { get; }

    public IReadOnlyList<M2SceneSubmissionEntry> Entries { get; }

    public int VertexCount => Entries.Sum(static entry => entry.VertexCount);

    public int IndexCount => Entries.Sum(static entry => entry.IndexCount);
}

public static class M2SceneSubmissionCoordinator
{
    public static M2SceneSubmissionPlan BuildPlan(
        IEnumerable<M2SceneSubmissionEntry> entries,
        M2RuntimeOptions options,
        M2SceneBatchLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(entries);
        limits ??= new M2SceneBatchLimits();

        List<M2SceneSubmissionEntry> ordered = entries
            .OrderBy(static entry => entry.Family)
            .ThenBy(static entry => entry.ModelKey, StringComparer.OrdinalIgnoreCase)
            .ThenBy(static entry => entry.TextureSortKey)
            .ThenBy(static entry => entry.EffectKey, StringComparer.OrdinalIgnoreCase)
            .ThenBy(static entry => entry.StateBucket)
            .ThenByDescending(entry => ShouldDepthSortEntry(entry, options) ? entry.DepthSortValue : 0.0f)
            .ThenBy(static entry => entry.EntryKey, StringComparer.OrdinalIgnoreCase)
            .ToList();

        List<M2SceneSubmissionBatch> batches = new();
        List<M2SceneSubmissionEntry> currentEntries = new();
        M2SceneSubmissionEntry? currentKey = null;
        bool currentDirect = false;
        int currentVertices = 0;
        int currentIndices = 0;
        int directCount = 0;
        int batchedCount = 0;

        foreach (M2SceneSubmissionEntry entry in ordered)
        {
            M2SceneFamilyPolicy policy = GetFamilyPolicy(entry.Family);
            bool isDirect = entry.ForceDirect || policy.RequiresDirectSubmission(options);
            bool startsNewBatch = currentKey is null
                || currentDirect != isDirect
                || !CanShareBatch(currentKey, entry)
                || currentVertices + entry.VertexCount > limits.MaxVertices
                || currentIndices + entry.IndexCount > limits.MaxIndices;

            if (startsNewBatch && currentKey is not null)
                FlushCurrent();

            currentKey ??= entry;
            currentDirect = isDirect;
            currentEntries.Add(entry);
            currentVertices += entry.VertexCount;
            currentIndices += entry.IndexCount;
            if (isDirect)
                directCount++;
            else
                batchedCount++;
        }

        if (currentKey is not null)
            FlushCurrent();

        return new M2SceneSubmissionPlan(options, batches, directCount, batchedCount);

        void FlushCurrent()
        {
            batches.Add(new M2SceneSubmissionBatch(
                batches.Count,
                currentKey!.Family,
                currentKey.ModelKey,
                currentKey.EffectKey,
                currentKey.TextureSortKey,
                currentKey.StateBucket,
                currentDirect,
                GetFamilyPolicy(currentKey.Family).HandlerName,
                GetFamilyPolicy(currentKey.Family).UsesDedicatedStateScope,
                currentEntries.ToArray()));

            currentEntries.Clear();
            currentKey = null;
            currentVertices = 0;
            currentIndices = 0;
        }
    }

    private static bool CanShareBatch(M2SceneSubmissionEntry left, M2SceneSubmissionEntry right)
    {
        return left.Family == right.Family
            && string.Equals(left.ModelKey, right.ModelKey, StringComparison.OrdinalIgnoreCase)
            && string.Equals(left.EffectKey, right.EffectKey, StringComparison.OrdinalIgnoreCase)
            && left.TextureSortKey == right.TextureSortKey
            && left.StateBucket == right.StateBucket
            && left.IsTransparent == right.IsTransparent
            && left.IsAdditive == right.IsAdditive;
    }

    private static bool ShouldDepthSortEntry(M2SceneSubmissionEntry entry, M2RuntimeOptions options)
    {
        return entry.IsTransparent || (entry.IsAdditive && options.HasFlag(M2RuntimeOptions.ForceAdditiveParticleSort));
    }

    public static M2SceneFamilyPolicy GetFamilyPolicy(M2RenderEntryFamily family)
    {
        return family switch
        {
            M2RenderEntryFamily.Core => new M2SceneFamilyPolicy(family, "core-batch", M2RuntimeOptions.None, alwaysDirect: false, usesDedicatedStateScope: false),
            M2RenderEntryFamily.Projected => new M2SceneFamilyPolicy(family, "projected-batch", M2RuntimeOptions.None, alwaysDirect: false, usesDedicatedStateScope: true),
            M2RenderEntryFamily.Doodad => new M2SceneFamilyPolicy(family, "doodad-batch", M2RuntimeOptions.BatchDoodads, alwaysDirect: false, usesDedicatedStateScope: false),
            M2RenderEntryFamily.Ribbon => new M2SceneFamilyPolicy(family, "ribbon-direct", M2RuntimeOptions.None, alwaysDirect: true, usesDedicatedStateScope: true),
            M2RenderEntryFamily.Particle => new M2SceneFamilyPolicy(family, "particle-dispatch", M2RuntimeOptions.BatchParticles, alwaysDirect: false, usesDedicatedStateScope: true),
            M2RenderEntryFamily.Callback => new M2SceneFamilyPolicy(family, "callback-direct", M2RuntimeOptions.None, alwaysDirect: true, usesDedicatedStateScope: true),
            M2RenderEntryFamily.HitTest => new M2SceneFamilyPolicy(family, "hit-test-direct", M2RuntimeOptions.None, alwaysDirect: true, usesDedicatedStateScope: true),
            _ => new M2SceneFamilyPolicy(family, "unknown-direct", M2RuntimeOptions.None, alwaysDirect: true, usesDedicatedStateScope: true),
        };
    }
}
