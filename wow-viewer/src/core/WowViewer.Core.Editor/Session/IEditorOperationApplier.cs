using WowViewer.Core.Editor.Operations;

namespace WowViewer.Core.Editor.Session;

/// <summary>
/// Applies an operation to both the source file and the live scene. The viewer shell implements
/// this; the session only records/reverses operations, which keeps the undo history decoupled from
/// any renderer or scene type (Spec 167 FR-002).
/// </summary>
public interface IEditorOperationApplier
{
    /// <summary>Applies the operation. Throws on failure so the session can abort the transition.</summary>
    void Apply(EditorOperation operation);
}