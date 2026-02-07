using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

/// <summary>
/// EGxBlend modes from WoW Alpha 0.5.3 (Ghidra: SetMaterialBlendMode @ 0x00448cb0).
/// Manages OpenGL blend/depth state to match original client behavior.
/// </summary>
public enum EGxBlend
{
    /// <summary>No blending, depth write ON.</summary>
    Opaque = 0,

    /// <summary>SrcAlpha / OneMinusSrcAlpha, depth write OFF.</summary>
    Blend = 1,

    /// <summary>SrcAlpha / One (additive), depth write OFF.</summary>
    Add = 2,

    /// <summary>SrcAlpha / OneMinusSrcAlpha + alpha test > 0.5, depth write ON.</summary>
    AlphaKey = 3
}

/// <summary>
/// Applies EGxBlend state to the GL context, matching the original Alpha client's
/// blend function and depth-write behavior per mode.
/// </summary>
public class BlendStateManager
{
    private readonly GL _gl;
    private EGxBlend _current = (EGxBlend)(-1); // force first apply

    public BlendStateManager(GL gl)
    {
        _gl = gl;
    }

    /// <summary>
    /// Current active blend mode.
    /// </summary>
    public EGxBlend Current => _current;

    /// <summary>
    /// Apply the given blend mode. No-op if already active.
    /// </summary>
    public void Apply(EGxBlend mode)
    {
        if (mode == _current) return;
        _current = mode;

        switch (mode)
        {
            case EGxBlend.Opaque:
                _gl.Disable(EnableCap.Blend);
                _gl.DepthMask(true);
                break;

            case EGxBlend.Blend:
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                _gl.DepthMask(false);
                break;

            case EGxBlend.Add:
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                _gl.DepthMask(false);
                break;

            case EGxBlend.AlphaKey:
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                _gl.DepthMask(true);
                // Alpha test is handled in shader: discard if alpha < 0.5
                break;
        }
    }

    /// <summary>
    /// Reset to a known default state (opaque).
    /// </summary>
    public void Reset()
    {
        _current = (EGxBlend)(-1);
        Apply(EGxBlend.Opaque);
    }

    /// <summary>
    /// Whether the given mode requires depth write.
    /// </summary>
    public static bool WritesDepth(EGxBlend mode) => mode == EGxBlend.Opaque || mode == EGxBlend.AlphaKey;

    /// <summary>
    /// Whether the given mode is considered transparent (needs back-to-front sorting).
    /// </summary>
    public static bool IsTransparent(EGxBlend mode) => mode == EGxBlend.Blend || mode == EGxBlend.Add;
}
