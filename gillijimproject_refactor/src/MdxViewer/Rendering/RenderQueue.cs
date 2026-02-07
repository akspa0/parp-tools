using System.Numerics;
using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

/// <summary>
/// Collects renderable items per frame and sorts them for correct draw order:
///   1. Opaque items — front-to-back (early-Z optimization)
///   2. Transparent items — back-to-front (correct alpha compositing)
/// Based on the WoW Alpha 0.5.3 transparency sorting from 04_Shader_Rendering_System.md.
/// </summary>
public class RenderQueue
{
    /// <summary>
    /// A single item in the render queue.
    /// </summary>
    public struct RenderItem
    {
        /// <summary>Material describing blend mode, texture, color.</summary>
        public Material Material;

        /// <summary>Model-to-world transform.</summary>
        public Matrix4x4 WorldTransform;

        /// <summary>
        /// Callback that performs the actual draw call.
        /// Receives the GL context and the shader program to use.
        /// </summary>
        public Action<GL, ShaderProgram> DrawCallback;

        /// <summary>Squared distance from camera for sorting.</summary>
        public float DistanceSq;
    }

    private readonly List<RenderItem> _opaqueItems = new();
    private readonly List<RenderItem> _transparentItems = new();

    /// <summary>Number of opaque items queued this frame.</summary>
    public int OpaqueCount => _opaqueItems.Count;

    /// <summary>Number of transparent items queued this frame.</summary>
    public int TransparentCount => _transparentItems.Count;

    /// <summary>
    /// Clear all items. Call at the start of each frame.
    /// </summary>
    public void Clear()
    {
        _opaqueItems.Clear();
        _transparentItems.Clear();
    }

    /// <summary>
    /// Submit an item. Automatically routes to opaque or transparent list based on blend mode.
    /// </summary>
    public void Submit(RenderItem item)
    {
        if (BlendStateManager.IsTransparent(item.Material.BlendMode))
            _transparentItems.Add(item);
        else
            _opaqueItems.Add(item);
    }

    /// <summary>
    /// Submit an item with automatic distance calculation from a camera position.
    /// </summary>
    public void Submit(RenderItem item, Vector3 cameraPosition)
    {
        var pos = item.WorldTransform.Translation;
        item.DistanceSq = Vector3.DistanceSquared(pos, cameraPosition);
        Submit(item);
    }

    /// <summary>
    /// Sort and draw all queued items.
    /// Opaque: front-to-back. Transparent: back-to-front.
    /// </summary>
    public void Flush(GL gl, ShaderProgram shader, BlendStateManager blendState)
    {
        // Sort opaque front-to-back (ascending distance)
        _opaqueItems.Sort((a, b) => a.DistanceSq.CompareTo(b.DistanceSq));

        // Sort transparent back-to-front (descending distance)
        _transparentItems.Sort((a, b) => b.DistanceSq.CompareTo(a.DistanceSq));

        // Draw opaque pass
        foreach (ref readonly var item in _opaqueItems.AsReadOnlySpan())
        {
            blendState.Apply(item.Material.BlendMode);
            ApplyMaterial(gl, shader, item.Material);
            shader.SetMat4("uModel", item.WorldTransform);
            item.DrawCallback(gl, shader);
        }

        // Draw transparent pass
        foreach (ref readonly var item in _transparentItems.AsReadOnlySpan())
        {
            blendState.Apply(item.Material.BlendMode);
            ApplyMaterial(gl, shader, item.Material);
            shader.SetMat4("uModel", item.WorldTransform);
            item.DrawCallback(gl, shader);
        }
    }

    private static void ApplyMaterial(GL gl, ShaderProgram shader, in Material mat)
    {
        shader.SetInt("uHasTexture", mat.DiffuseTexture != 0 ? 1 : 0);
        shader.SetVec4("uColor", mat.DiffuseColor);
        shader.SetFloat("uAlphaTest", mat.BlendMode == EGxBlend.AlphaKey ? WoWConstants.AlphaKeyThreshold : 0f);

        if (mat.DiffuseTexture != 0)
        {
            gl.ActiveTexture(TextureUnit.Texture0);
            gl.BindTexture(TextureTarget.Texture2D, mat.DiffuseTexture);
        }

        if (mat.TwoSided)
            gl.Disable(EnableCap.CullFace);
        else
            gl.Enable(EnableCap.CullFace);
    }
}

/// <summary>
/// Extension to allow ReadOnlySpan iteration over List without allocation.
/// </summary>
internal static class ListSpanExtensions
{
    public static ReadOnlySpan<T> AsReadOnlySpan<T>(this List<T> list)
    {
        return System.Runtime.InteropServices.CollectionsMarshal.AsSpan(list);
    }
}
