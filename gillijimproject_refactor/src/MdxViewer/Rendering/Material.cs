using System.Numerics;

namespace MdxViewer.Rendering;

/// <summary>
/// Describes the visual properties of a renderable surface.
/// Mirrors the WoW Alpha 0.5.3 material system from Ghidra analysis (04_Shader_Rendering_System.md).
/// </summary>
public struct Material
{
    /// <summary>Blend mode (Opaque, Blend, Add, AlphaKey).</summary>
    public EGxBlend BlendMode;

    /// <summary>OpenGL texture handle for the diffuse map (0 = untextured).</summary>
    public uint DiffuseTexture;

    /// <summary>Diffuse color multiplier (default white).</summary>
    public Vector4 DiffuseColor;

    /// <summary>Whether to render both sides (disable back-face culling).</summary>
    public bool TwoSided;

    /// <summary>
    /// Sort key used by the render queue.
    /// Lower = rendered first in the opaque pass, higher = rendered first in transparent pass.
    /// </summary>
    public float SortDistance;

    /// <summary>
    /// Create a default opaque white material.
    /// </summary>
    public static Material Default => new()
    {
        BlendMode = EGxBlend.Opaque,
        DiffuseTexture = 0,
        DiffuseColor = Vector4.One,
        TwoSided = false,
        SortDistance = 0f
    };
}
