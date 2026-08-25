namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// The atomic reconciliation apply expressed as data (Spec 176 FR-015). The operation carries both
/// the bytes written to the output and the bytes that existed there before (null when the output
/// did not exist), so undo restores the prior output byte-identically — or removes the output and
/// its provenance sidecar when the apply created them. The session's applier materializes the
/// state; the operation itself never touches the file system.
/// </summary>
public sealed class ReconciliationApplyOperation : EditorOperation
{
    private readonly bool _reversed;

    public ReconciliationApplyOperation(
        string operationId,
        string originPluginId,
        string sourcePath,
        string outputPath,
        string reportPath,
        byte[]? priorOutputBytes,
        byte[] writtenOutputBytes,
        string reportJson)
        : this(operationId, originPluginId, sourcePath, outputPath, reportPath,
            priorOutputBytes, writtenOutputBytes, reportJson, reversed: false)
    {
    }

    private ReconciliationApplyOperation(
        string operationId,
        string originPluginId,
        string sourcePath,
        string outputPath,
        string reportPath,
        byte[]? priorOutputBytes,
        byte[] writtenOutputBytes,
        string reportJson,
        bool reversed)
        : base(
            operationId,
            originPluginId,
            $"PM4 reconciliation apply -> {Path.GetFileName(outputPath)}",
            undoable: true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(reportPath);
        ArgumentNullException.ThrowIfNull(writtenOutputBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(reportJson);

        SourcePath = sourcePath;
        OutputPath = outputPath;
        ReportPath = reportPath;
        PriorOutputBytes = priorOutputBytes;
        WrittenOutputBytes = writtenOutputBytes;
        ReportJson = reportJson;
        _reversed = reversed;
    }

    public string SourcePath { get; }

    public string OutputPath { get; }

    public string ReportPath { get; }

    /// <summary>The output content before the apply, or null when the apply created the output.</summary>
    public byte[]? PriorOutputBytes { get; }

    /// <summary>The output content written by the apply.</summary>
    public byte[] WrittenOutputBytes { get; }

    /// <summary>The provenance sidecar JSON written beside the output.</summary>
    public string ReportJson { get; }

    /// <summary>The bytes the applier must materialize at <see cref="OutputPath"/> for this
    /// application of the operation, or null when the applier must remove the output.</summary>
    public byte[]? BytesToWrite => _reversed ? PriorOutputBytes : WrittenOutputBytes;

    /// <summary>True when the sidecar report must exist after this application.</summary>
    public bool ReportExists => !_reversed;

    public override IReadOnlyList<string> AffectedPaths => [SourcePath, OutputPath, ReportPath];

    public override EditorOperation CreateReverse()
        => new ReconciliationApplyOperation(
            $"reverse:{OperationId}",
            OriginPluginId,
            SourcePath,
            OutputPath,
            ReportPath,
            PriorOutputBytes,
            WrittenOutputBytes,
            ReportJson,
            reversed: !_reversed);
}
