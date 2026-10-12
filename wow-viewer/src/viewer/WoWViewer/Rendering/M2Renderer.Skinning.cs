using System.Numerics;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

// Spec 256 amendment A: GPU skinning for native animated M2s. The CPU path rebuilt every vertex of every
// animated model every frame (M2SkinnedRenderModelBuilder), allocated the result, then re-uploaded the whole
// vertex buffer. Here the vertex buffer keeps the bind pose plus each vertex's model bone indices and weights,
// resolved once at load exactly as ApplyVertex resolves them per frame, and the vertex shader applies the
// bone matrices. Per frame only the matrices are evaluated (allocation-free) and uploaded as uniforms.
//
// Models with more bones than the shader array holds keep the CPU path unchanged.
public sealed partial class M2Renderer
{
    /// <summary>Size of <c>uBones</c> in the shader (same limit MdxRenderer's skinning uses).</summary>
    private const int MaxGpuSkinningBones = 128;

    /// <summary>Position, normal, uv0, uv1 (10 floats) + bone indices (4) + bone weights (4).</summary>
    private const int GpuSkinnedVertexFloats = 18;

    private const uint SkinBoneIndexAttribute = 9;
    private const uint SkinBoneWeightAttribute = 10;

    private static int _uSkinned;
    private static int _uBones;

    private bool _gpuSkinning;
    private bool _hasBonePose;
    private int _boneCount;
    private Matrix4x4[] _boneMatrices = Array.Empty<Matrix4x4>();
    private bool[] _boneSolvedScratch = Array.Empty<bool>();
    private Dictionary<int, M2StructuredRenderSection>? _skinSourcesBySection;

    private void InitGpuSkinning()
    {
        int boneCount = _runtimeModel?.Model.Bones.Count ?? 0;
        _gpuSkinning = _runtimeAnimator != null && boneCount > 0 && boneCount <= MaxGpuSkinningBones;
        if (!_gpuSkinning)
            return;

        _boneCount = boneCount;
        _boneMatrices = new Matrix4x4[boneCount];
        _boneSolvedScratch = new bool[boneCount];

        // The CPU path matched skinned sections to vertex buffers by section index (last one wins).
        _skinSourcesBySection = new Dictionary<int, M2StructuredRenderSection>();
        foreach (M2StructuredRenderSection structured in _runtimeModel!.StructuredSections)
            _skinSourcesBySection[structured.SectionIndex] = structured;
    }

    // The structured section the CPU path skinned into this buffer, or null when it never updated it
    // (no match or a different vertex count): such a buffer stays in its bind pose, as before.
    private M2StructuredRenderSection? FindSkinSource(M2StaticRenderSection section)
    {
        if (_skinSourcesBySection == null
            || !_skinSourcesBySection.TryGetValue(section.SectionIndex, out M2StructuredRenderSection? structured)
            || structured.Vertices.Count != section.Vertices.Count)
        {
            return null;
        }

        return structured;
    }

    private void WriteSkinningAttributes(float[] vertexData, int offset, M2StructuredRenderSection? skinSource, int vertexIndex)
    {
        // Bone indices 0, weights 0: the shader then keeps the bind-pose vertex.
        for (int component = 10; component < GpuSkinnedVertexFloats; component++)
            vertexData[offset + component] = 0.0f;

        if (skinSource == null)
            return;

        // The CPU path skinned the structured vertex; use its position and normal as the bind pose.
        M2StaticRenderVertex source = skinSource.Vertices[vertexIndex];
        vertexData[offset + 0] = source.Position.X;
        vertexData[offset + 1] = source.Position.Y;
        vertexData[offset + 2] = source.Position.Z;
        vertexData[offset + 3] = source.Normal.X;
        vertexData[offset + 4] = source.Normal.Y;
        vertexData[offset + 5] = source.Normal.Z;

        for (int influence = 0; influence < 4; influence++)
        {
            float weight = Component(source.BoneWeights, influence);
            if (weight <= 0.0f)
                continue;

            int boneIndex = M2SkinnedRenderModelBuilder.ResolveBoneIndex(_runtimeModel!, skinSource, (int)Component(source.BoneIndices, influence));
            if (boneIndex < 0 || boneIndex >= _boneCount)
                continue; // skipped by ApplyVertex too: weight stays 0

            vertexData[offset + 10 + influence] = boneIndex;
            vertexData[offset + 14 + influence] = weight;
        }
    }

    private static float Component(Vector4 value, int index) => index switch
    {
        0 => value.X,
        1 => value.Y,
        2 => value.Z,
        3 => value.W,
        _ => 0.0f,
    };

    // Before the first evaluated pose the buffer's bind pose is drawn, as the CPU path drew it.
    private unsafe void ApplySkinningUniforms()
    {
        bool skinned = _gpuSkinning && _hasBonePose;
        _gl!.Uniform1(_uSkinned, skinned ? 1 : 0);
        if (!skinned)
            return;

        fixed (Matrix4x4* matrices = _boneMatrices)
            _gl.UniformMatrix4(_uBones, (uint)_boneCount, false, (float*)matrices);
    }
}
