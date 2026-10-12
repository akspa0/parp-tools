namespace WoWViewer.Terrain;

/// <summary>
/// Lightweight placement instance — just a model key and world transform.
/// The actual renderer is looked up from WorldAssetManager at render time.
/// </summary>
public enum ObjectType { None, Wmo, Mdx, WmoDoodad }
