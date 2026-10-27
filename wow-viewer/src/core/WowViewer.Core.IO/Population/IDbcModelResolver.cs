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
    /// Resolves a creature display ID to its model path (.m2), scale, and optional FileDataId.
    /// </summary>
    bool TryResolveCreature(int displayId, out string? modelPath, out float displayScale, out uint? fileDataId)
    {
        fileDataId = null;
        return TryResolveCreature(displayId, out modelPath, out displayScale);
    }

    /// <summary>
    /// Resolves a creature display ID to its model path (.m2), scale, optional FileDataId, and custom/baked textures.
    /// </summary>
    bool TryResolveCreature(int displayId, out string? modelPath, out float displayScale, out uint? fileDataId, out string[]? customTextures)
    {
        customTextures = null;
        return TryResolveCreature(displayId, out modelPath, out displayScale, out fileDataId);
    }

    /// <summary>
    /// Resolves a gameobject display ID to its model path (.mdx/.m2) and scale.
    /// </summary>
    bool TryResolveGameObject(int displayId, out string? modelPath, out float displayScale);

    /// <summary>
    /// Resolves a gameobject display ID to its model path (.mdx/.m2), scale, and optional FileDataId.
    /// </summary>
    bool TryResolveGameObject(int displayId, out string? modelPath, out float displayScale, out uint? fileDataId)
    {
        fileDataId = null;
        return TryResolveGameObject(displayId, out modelPath, out displayScale);
    }

    /// <summary>
    /// Resolves a map ID to its canonical human-readable name.
    /// </summary>
    string GetMapName(int mapId);
}
