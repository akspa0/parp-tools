using System.Numerics;

namespace WoWViewer.Rendering;

/// <summary>
/// Optional renderer contract for models that can emit point lights into the world scene.
/// </summary>
public interface ISceneLightEmitter
{
    void CollectSceneLights(Matrix4x4 modelMatrix, ICollection<SceneLight> lights, string sourceKey);
}
