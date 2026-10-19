namespace WowViewer.Core.IO.Population;

/// <summary>
/// Abstraction for resolving creature and gameobject display IDs into virtual client model paths (.m2/.mdx) and scales.
/// </summary>
public interface IDbcModelResolver
{
    /// <summary>
    /// Resolves a creature display ID to its model path (.m2) and scale.
    /// </summary>
    bool TryResolveCreature(int displayId, out string? modelPath, out float displayScale);

    /// <summary>
    /// Resolves a gameobject display ID to its model path (.mdx/.m2) and scale.
    /// </summary>
    bool TryResolveGameObject(int displayId, out string? modelPath, out float displayScale);
}
