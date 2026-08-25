namespace WowViewer.Core.Editor.Bridge;

/// <summary>
/// The read side of the editor↔runtime bridge. The viewer shell implements this by adapting its
/// live scene into <see cref="EditorSceneSnapshot"/>; core and plugins depend only on the snapshot,
/// so no runtime/scene/renderer type ever needs to reference an editor type.
/// </summary>
public interface IEditorSceneReader
{
    EditorSceneSnapshot Capture();
}