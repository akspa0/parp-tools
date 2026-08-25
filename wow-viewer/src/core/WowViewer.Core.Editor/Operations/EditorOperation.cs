namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// The data-based unit of editor work shared across plugins. An operation records what changed and
/// how to reverse it, which is what lets the session host (Spec 168) undo an operation without
/// knowing which plugin produced it. This is not a concession to any external consumer; it is the
/// reason editor work is reversible at all.
/// </summary>
public abstract class EditorOperation
{
    protected EditorOperation(string operationId, string originPluginId, string description, bool undoable)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(operationId);
        ArgumentException.ThrowIfNullOrWhiteSpace(originPluginId);

        OperationId = operationId;
        OriginPluginId = originPluginId;
        Description = description;
        Undoable = undoable;
    }

    public string OperationId { get; }

    public string OriginPluginId { get; }

    public string Description { get; }

    /// <summary>False means the operation declares itself non-reversible and the host records the
    /// point in history rather than offering a broken undo.</summary>
    public bool Undoable { get; }

    /// <summary>Distinct loose-file paths this operation reads or writes.</summary>
    public abstract IReadOnlyList<string> AffectedPaths { get; }

    public abstract EditorOperation CreateReverse();

    public override string ToString() => $"{OriginPluginId}: {Description}";
}