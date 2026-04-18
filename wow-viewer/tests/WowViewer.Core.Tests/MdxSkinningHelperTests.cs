using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxSkinningHelperTests
{
    [Fact]
    public void BuildSkinningVertexData_InterleavesIndicesAndWeightsPerVertex()
    {
        Vector4[] indices =
        [
            new Vector4(1.0f, 2.0f, 3.0f, 4.0f),
            new Vector4(5.0f, 6.0f, 7.0f, 8.0f),
        ];

        Vector4[] weights =
        [
            new Vector4(0.10f, 0.20f, 0.30f, 0.40f),
            new Vector4(0.25f, 0.25f, 0.25f, 0.25f),
        ];

        float[] packed = MdxSkinningHelper.BuildSkinningVertexData(indices, weights, vertexCount: 2);

        Assert.Equal(16, packed.Length);
        Assert.Equal(1.0f, packed[0]);
        Assert.Equal(2.0f, packed[1]);
        Assert.Equal(3.0f, packed[2]);
        Assert.Equal(4.0f, packed[3]);
        Assert.Equal(0.10f, packed[4]);
        Assert.Equal(0.20f, packed[5]);
        Assert.Equal(0.30f, packed[6]);
        Assert.Equal(0.40f, packed[7]);

        Assert.Equal(5.0f, packed[8]);
        Assert.Equal(6.0f, packed[9]);
        Assert.Equal(7.0f, packed[10]);
        Assert.Equal(8.0f, packed[11]);
        Assert.Equal(0.25f, packed[12]);
        Assert.Equal(0.25f, packed[13]);
        Assert.Equal(0.25f, packed[14]);
        Assert.Equal(0.25f, packed[15]);
    }

    [Fact]
    public void BuildSkinningVertexData_UsesZeroFallbackWhenVertexInputsAreMissing()
    {
        Vector4[] indices = [new Vector4(9.0f, 8.0f, 7.0f, 6.0f)];
        Vector4[] weights = [new Vector4(1.0f, 0.0f, 0.0f, 0.0f)];

        float[] packed = MdxSkinningHelper.BuildSkinningVertexData(indices, weights, vertexCount: 2);

        Assert.Equal(16, packed.Length);
        Assert.Equal(9.0f, packed[0]);
        Assert.Equal(8.0f, packed[1]);
        Assert.Equal(7.0f, packed[2]);
        Assert.Equal(6.0f, packed[3]);
        Assert.Equal(1.0f, packed[4]);
        Assert.Equal(0.0f, packed[5]);
        Assert.Equal(0.0f, packed[6]);
        Assert.Equal(0.0f, packed[7]);

        for (int index = 8; index < packed.Length; index++)
            Assert.Equal(0.0f, packed[index]);
    }

    [Fact]
    public void BuildSkinningVertexData_PreservesCpuSkinningParityWhenUnpacked()
    {
        MdxGeosetGeometry geoset = CreateSkinnedGeoset();
        IReadOnlyList<MdxBone> bones =
        [
            new MdxBone(0, "Bone5", 5, -1, 0u, uint.MaxValue, uint.MaxValue, Vector3.Zero, null, null, null),
            new MdxBone(1, "Bone9", 9, -1, 0u, uint.MaxValue, uint.MaxValue, Vector3.Zero, null, null, null),
        ];

        (Vector4[] indices, Vector4[] weights) = MdxSkinningHelper.BuildBoneWeights(geoset, bones);
        float[] packed = MdxSkinningHelper.BuildSkinningVertexData(indices, weights, geoset.VertexCount);

        Matrix4x4[] boneMatrices =
        [
            Matrix4x4.CreateTranslation(2.0f, 0.0f, 0.0f),
            Matrix4x4.CreateFromAxisAngle(Vector3.UnitZ, MathF.PI * 0.5f),
        ];

        for (int vertexIndex = 0; vertexIndex < geoset.VertexCount; vertexIndex++)
        {
            Vector3 cpuPosition = MdxSkinningHelper.ApplySkinning(
                geoset.Vertices[vertexIndex],
                indices[vertexIndex],
                weights[vertexIndex],
                boneMatrices);

            Vector3 cpuNormal = MdxSkinningHelper.ApplySkinningNormal(
                geoset.Normals[vertexIndex],
                indices[vertexIndex],
                weights[vertexIndex],
                boneMatrices);

            int packedOffset = vertexIndex * 8;
            Vector4 packedIndices = new(
                packed[packedOffset + 0],
                packed[packedOffset + 1],
                packed[packedOffset + 2],
                packed[packedOffset + 3]);
            Vector4 packedWeights = new(
                packed[packedOffset + 4],
                packed[packedOffset + 5],
                packed[packedOffset + 6],
                packed[packedOffset + 7]);

            Vector3 unpackedPosition = MdxSkinningHelper.ApplySkinning(
                geoset.Vertices[vertexIndex],
                packedIndices,
                packedWeights,
                boneMatrices);

            Vector3 unpackedNormal = MdxSkinningHelper.ApplySkinningNormal(
                geoset.Normals[vertexIndex],
                packedIndices,
                packedWeights,
                boneMatrices);

            Assert.Equal(cpuPosition, unpackedPosition, new Vector3EqualityComparer(0.0005f));
            Assert.Equal(cpuNormal, unpackedNormal, new Vector3EqualityComparer(0.0005f));
        }
    }

    private static MdxGeosetGeometry CreateSkinnedGeoset()
    {
        return new MdxGeosetGeometry(
            index: 0,
            vertices:
            [
                new Vector3(1.0f, 0.0f, 0.0f),
                new Vector3(0.0f, 1.0f, 0.0f),
            ],
            normals:
            [
                Vector3.UnitX,
                Vector3.UnitY,
            ],
            uvSets: [[]],
            primitiveTypes: [4],
            faceGroups: [3],
            indices: [0, 1, 0],
            vertexGroups: [0, 0],
            matrixGroups: [2u],
            matrixIndices: [5u, 9u],
            boneIndices: [],
            boneWeights: [],
            materialId: 0,
            selectionGroup: 0u,
            flags: 0u,
            boundsRadius: 1.0f,
            boundsMin: Vector3.Zero,
            boundsMax: Vector3.One,
            animationExtentCount: 0);
    }

    private sealed class Vector3EqualityComparer(float epsilon) : IEqualityComparer<Vector3>
    {
        public bool Equals(Vector3 x, Vector3 y)
        {
            return MathF.Abs(x.X - y.X) <= epsilon
                && MathF.Abs(x.Y - y.Y) <= epsilon
                && MathF.Abs(x.Z - y.Z) <= epsilon;
        }

        public int GetHashCode(Vector3 obj) => HashCode.Combine(obj.X, obj.Y, obj.Z);
    }
}
