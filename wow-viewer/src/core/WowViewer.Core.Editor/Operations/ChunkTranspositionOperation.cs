namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Reversible editor operation representing a multi-tile chunk transposition / move / paste.
/// </summary>
public sealed class ChunkTranspositionOperation : EditorOperation
{
    public const string PluginId = "chunk.transposition";

    public IReadOnlyList<string> TargetAdtPaths { get; }
    public ChunkTranspositionOptions Options { get; }
    public GlobalChunkCoordinate SourceOrigin { get; }
    public GlobalChunkCoordinate TargetOrigin { get; }
    public ChunkTranspositionPayload BeforePayload { get; }
    public ChunkTranspositionPayload AfterPayload { get; }

    public override IReadOnlyList<string> AffectedPaths => TargetAdtPaths;

    public ChunkTranspositionOperation(
        string operationId,
        IReadOnlyList<string> targetAdtPaths,
        GlobalChunkCoordinate sourceOrigin,
        GlobalChunkCoordinate targetOrigin,
        ChunkTranspositionOptions options,
        ChunkTranspositionPayload beforePayload,
        ChunkTranspositionPayload afterPayload,
        string description = "Transpose chunks")
        : base(operationId, PluginId, description, undoable: true)
    {
        TargetAdtPaths = targetAdtPaths ?? Array.Empty<string>();
        SourceOrigin = sourceOrigin;
        TargetOrigin = targetOrigin;
        Options = options;
        BeforePayload = beforePayload;
        AfterPayload = afterPayload;
    }

    public override EditorOperation CreateReverse()
    {
        return new ChunkTranspositionOperation(
            Guid.NewGuid().ToString("N"),
            TargetAdtPaths,
            sourceOrigin: TargetOrigin,
            targetOrigin: SourceOrigin,
            options: Options,
            beforePayload: AfterPayload,
            afterPayload: BeforePayload,
            description: $"Undo transpose from {SourceOrigin} to {TargetOrigin}");
    }
}
