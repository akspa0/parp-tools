using System.Numerics;
using Silk.NET.OpenGL;

namespace WoWViewer.Rendering;

// Spec 256 P3: GPU instancing for the native backend. Every instance of one model shares its geometry and
// its animation state (CPU-skinned vertices are per model, updated once per frame by the world scene), so
// the per-instance state is only the model matrix and the distance fade. One instanced draw per visible
// section replaces one draw per instance per section.
//
// Parity with RenderInstance(transform, RenderPass.Opaque, fade): the opaque pass disables blending, so a
// fade only scales the section color (ComputeSectionColor); the shader applies the same scale per instance
// from aInstanceFade. Native RenderCore takes no per-instance lights, so nothing else varies per instance.
public sealed partial class M2Renderer
{
    private const int InstanceFloats = 17; // mat4 + fade
    private const uint InstanceStrideBytes = InstanceFloats * sizeof(float);
    private const uint FirstInstanceAttribute = 4;

    private static int _uInstanced;

    private uint _instanceVbo;
    private float[] _instanceUpload = new float[InstanceFloats * 16];
    private int _instanceCount;
    private bool _gpuInstanceBatchActive;

    /// <summary>
    /// Honors the existing opaque-model GPU instancing toggle (<see cref="MdxRenderer.GpuInstancingEnabled"/>),
    /// so one switch compares both backends' instanced and per-instance paths.
    /// </summary>
    private bool NativeSupportsGpuInstancedOpaque
        => _gl != null && !_disposed && _instanceVbo != 0 && !_wireframe && MdxRenderer.GpuInstancingEnabled;

    private void CreateInstanceBuffer()
    {
        _instanceVbo = _gl!.GenBuffer();
    }

    private void DeleteInstanceBuffer()
    {
        if (_instanceVbo != 0)
            _gl!.DeleteBuffer(_instanceVbo);
        _instanceVbo = 0;
    }

    // Called with a section VAO bound: points attributes 4–8 at the instance buffer, one step per instance.
    // They stay disabled, so non-instanced draws never read them (the shader ignores them when uInstanced is 0).
    private unsafe void ConfigureInstanceAttributes()
    {
        GL gl = _gl!;
        gl.BindBuffer(BufferTargetARB.ArrayBuffer, _instanceVbo);
        for (uint row = 0; row < 4; row++)
        {
            uint location = FirstInstanceAttribute + row;
            gl.VertexAttribPointer(location, 4, VertexAttribPointerType.Float, false, InstanceStrideBytes, (void*)(row * 4 * sizeof(float)));
            gl.VertexAttribDivisor(location, 1);
        }

        gl.VertexAttribPointer(FirstInstanceAttribute + 4, 1, VertexAttribPointerType.Float, false, InstanceStrideBytes, (void*)(16 * sizeof(float)));
        gl.VertexAttribDivisor(FirstInstanceAttribute + 4, 1);
        gl.BindBuffer(BufferTargetARB.ArrayBuffer, 0);
    }

    private void SetInstanceAttributesEnabled(bool enabled)
    {
        GL gl = _gl!;
        for (uint location = FirstInstanceAttribute; location <= FirstInstanceAttribute + 4; location++)
        {
            if (enabled)
                gl.EnableVertexAttribArray(location);
            else
                gl.DisableVertexAttribArray(location);
        }
    }

    private void BeginNativeGpuInstanceBatch(
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor)
    {
        BeginBatch(view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
        _instanceCount = 0;
        _gpuInstanceBatchActive = true;
    }

    private void QueueNativeGpuInstance(Matrix4x4 modelMatrix, float fadeAlpha)
    {
        if (!_gpuInstanceBatchActive)
            return;

        // Not instanceable right now (wireframe toggled mid-frame, instancing switched off): draw this one
        // the per-instance way instead of dropping it.
        if (!NativeSupportsGpuInstancedOpaque)
        {
            RenderInstance(modelMatrix, RenderPass.Opaque, fadeAlpha);
            return;
        }

        int offset = _instanceCount * InstanceFloats;
        if (offset + InstanceFloats > _instanceUpload.Length)
            Array.Resize(ref _instanceUpload, _instanceUpload.Length * 2);

        // Memory order of Matrix4x4 (rows), the same bytes RenderCore passes to UniformMatrix4 for uModel.
        _instanceUpload[offset + 0] = modelMatrix.M11;
        _instanceUpload[offset + 1] = modelMatrix.M12;
        _instanceUpload[offset + 2] = modelMatrix.M13;
        _instanceUpload[offset + 3] = modelMatrix.M14;
        _instanceUpload[offset + 4] = modelMatrix.M21;
        _instanceUpload[offset + 5] = modelMatrix.M22;
        _instanceUpload[offset + 6] = modelMatrix.M23;
        _instanceUpload[offset + 7] = modelMatrix.M24;
        _instanceUpload[offset + 8] = modelMatrix.M31;
        _instanceUpload[offset + 9] = modelMatrix.M32;
        _instanceUpload[offset + 10] = modelMatrix.M33;
        _instanceUpload[offset + 11] = modelMatrix.M34;
        _instanceUpload[offset + 12] = modelMatrix.M41;
        _instanceUpload[offset + 13] = modelMatrix.M42;
        _instanceUpload[offset + 14] = modelMatrix.M43;
        _instanceUpload[offset + 15] = modelMatrix.M44;
        _instanceUpload[offset + 16] = fadeAlpha;
        _instanceCount++;
    }

    private unsafe void EndNativeGpuInstanceBatch()
    {
        if (!_gpuInstanceBatchActive)
            return;

        _gpuInstanceBatchActive = false;
        int count = _instanceCount;
        _instanceCount = 0;
        if (count == 0 || _gl == null || _disposed || !_batchStateValid)
            return;

        GL gl = _gl;
        gl.BindBuffer(BufferTargetARB.ArrayBuffer, _instanceVbo);
        fixed (float* data = _instanceUpload)
        {
            gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(count * InstanceFloats * sizeof(float)), data, BufferUsageARB.StreamDraw);
        }
        gl.BindBuffer(BufferTargetARB.ArrayBuffer, 0);

        // RenderCore's shared state for an opaque, non-backdrop pass.
        Matrix4x4 view = _batchView;
        Matrix4x4 proj = _batchProj;
        gl.UseProgram(_shaderProgram);
        gl.Uniform1(_uInstanced, 1);
        ApplySkinningUniforms();
        gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        gl.Uniform3(_uFogColor, _batchFogColor.X, _batchFogColor.Y, _batchFogColor.Z);
        gl.Uniform1(_uFogStart, _batchFogStart);
        gl.Uniform1(_uFogEnd, _batchFogEnd);
        gl.Uniform3(_uCameraPos, _batchCameraPos.X, _batchCameraPos.Y, _batchCameraPos.Z);
        gl.Uniform3(_uLightDir, _batchLightDir.X, _batchLightDir.Y, _batchLightDir.Z);
        gl.Uniform3(_uLightColor, _batchLightColor.X, _batchLightColor.Y, _batchLightColor.Z);
        gl.Uniform3(_uAmbientColor, _batchAmbientColor.X, _batchAmbientColor.Y, _batchAmbientColor.Z);
        gl.Enable(EnableCap.DepthTest);
        gl.DepthFunc(DepthFunction.Lequal);
        gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
        gl.Disable(EnableCap.CullFace);
        gl.Disable(EnableCap.Blend);
        gl.DepthMask(true);

        foreach (SectionBuffers section in _sections)
        {
            if (!section.Visible || section.TexturePending || section.Material.IsTransparent)
                continue;

            // Fade 1.0 here; the shader applies each instance's fade (vInstanceFade).
            ApplySectionUniforms(section, ComputeSectionColor(section, 1.0f), section.AnimatedAlpha);

            gl.BindVertexArray(section.Vao);
            SetInstanceAttributesEnabled(true);
            gl.DrawElementsInstanced(PrimitiveType.Triangles, section.IndexCount, DrawElementsType.UnsignedInt, null, (uint)count);
            SetInstanceAttributesEnabled(false);
            ModelDrawCallCounter.Record();
        }

        gl.Uniform1(_uInstanced, 0);
        gl.BindTexture(TextureTarget.Texture2D, 0);
        gl.BindVertexArray(0);
        gl.Disable(EnableCap.Blend);
        gl.Enable(EnableCap.DepthTest);
        gl.DepthFunc(DepthFunction.Lequal);
        gl.DepthMask(true);
    }
}
