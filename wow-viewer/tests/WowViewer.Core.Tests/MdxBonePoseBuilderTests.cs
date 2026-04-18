using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxBonePoseBuilderTests
{
    [Fact]
    public void Build_RotatesAroundPivotAndInheritsParentTransform()
    {
        MdxSummary summary = CreateSummary();
        MdxBoneFile boneFile = new(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            [
                new MdxBone(
                    0,
                    "Root",
                    0,
                    -1,
                    0u,
                    uint.MaxValue,
                    uint.MaxValue,
                    new Vector3(1.0f, 0.0f, 0.0f),
                    new MdxVector3NodeTrack("KGTR", MdxTrackInterpolationType.Linear, -1,
                    [
                        new MdxVector3Keyframe(100, Vector3.Zero, null, null),
                        new MdxVector3Keyframe(200, new Vector3(2.0f, 0.0f, 0.0f), null, null),
                    ]),
                    new MdxQuaternionNodeTrack("KGRT", MdxTrackInterpolationType.Linear, -1,
                    [
                        new MdxQuaternionKeyframe(100, Quaternion.Identity, null, null),
                        new MdxQuaternionKeyframe(200, Quaternion.CreateFromAxisAngle(Vector3.UnitZ, MathF.PI), null, null),
                    ]),
                    null),
                new MdxBone(
                    1,
                    "Child",
                    1,
                    0,
                    0u,
                    uint.MaxValue,
                    uint.MaxValue,
                    Vector3.Zero,
                    null,
                    null,
                    null),
            ]);

        Matrix4x4[] matrices = MdxBonePoseBuilder.Build(boneFile, summary, sequenceIndex: 0, timeMs: 50);
        Vector3 rotated = Vector3.Transform(new Vector3(2.0f, 0.0f, 0.0f), matrices[0]);
        Vector3 inherited = Vector3.Transform(Vector3.Zero, matrices[1]);

        Assert.Equal(new Vector3(2.0f, -1.0f, 0.0f), rotated, new Vector3EqualityComparer(0.0005f));
        Assert.Equal(new Vector3(2.0f, 1.0f, 0.0f), inherited, new Vector3EqualityComparer(0.0005f));
    }

    [Fact]
    public void ApplySkinning_UsesRemappedObjectIdsFromMatrixTable()
    {
        MdxGeosetGeometry geoset = new(
            0,
            [new Vector3(1.0f, 0.0f, 0.0f)],
            [Vector3.UnitX],
            [[]],
            [4],
            [3],
            [0, 0, 0],
            [0],
            [1u],
            [5u],
            [],
            [],
            0,
            0u,
            0u,
            1.0f,
            Vector3.Zero,
            Vector3.One,
            0);

        MdxBoneFile boneFile = new(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            [
                new MdxBone(0, "Bone5", 5, -1, 0u, uint.MaxValue, uint.MaxValue, Vector3.Zero, null, null, null),
            ]);

        (Vector4[] indices, Vector4[] weights) = MdxSkinningHelper.BuildBoneWeights(geoset, boneFile.Bones);
        Vector3 skinned = MdxSkinningHelper.ApplySkinning(
            new Vector3(1.0f, 0.0f, 0.0f),
            indices[0],
            weights[0],
            [Matrix4x4.CreateTranslation(3.0f, 0.0f, 0.0f)]);

        Assert.Equal(new Vector3(4.0f, 0.0f, 0.0f), skinned, new Vector3EqualityComparer(0.0005f));
    }

    [Fact]
    public void Build_SphericalBillboard_FacesBoneTowardCamera()
    {
        MdxSummary summary = CreateSummary();
        MdxBoneFile boneFile = new(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            [
                new MdxBone(
                    0,
                    "Billboard",
                    0,
                    -1,
                    MdxBone.SphericalBillboardFlag,
                    uint.MaxValue,
                    uint.MaxValue,
                    Vector3.Zero,
                    null,
                    null,
                    null),
            ]);

        Matrix4x4[] matrices = MdxBonePoseBuilder.Build(boneFile, summary, sequenceIndex: 0, timeMs: 0, cameraPosition: new Vector3(0.0f, 5.0f, 0.0f));
        Vector3 facing = Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitZ, matrices[0]));

        Assert.Equal(new Vector3(0.0f, 1.0f, 0.0f), facing, new Vector3EqualityComparer(0.0005f));
    }

    [Fact]
    public void Build_CylindricalBillboardLockZ_RotatesOnlyInHorizontalPlane()
    {
        MdxSummary summary = CreateSummary();
        MdxBoneFile boneFile = new(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            [
                new MdxBone(
                    0,
                    "BillboardZ",
                    0,
                    -1,
                    MdxBone.CylindricalBillboardLockZFlag,
                    uint.MaxValue,
                    uint.MaxValue,
                    Vector3.Zero,
                    null,
                    null,
                    null),
            ]);

        Matrix4x4[] matrices = MdxBonePoseBuilder.Build(boneFile, summary, sequenceIndex: 0, timeMs: 0, cameraPosition: new Vector3(0.0f, 2.0f, 3.0f));
        Vector3 facing = Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitZ, matrices[0]));

        Assert.Equal(new Vector3(0.0f, 1.0f, 0.0f), facing, new Vector3EqualityComparer(0.0005f));
    }

    private static MdxSummary CreateSummary()
    {
        return new MdxSummary(
            "synthetic.mdx",
            "MDLX",
            1300,
            "Synthetic",
            0,
            Vector3.Zero,
            Vector3.One,
            [],
            [new MdxSequenceSummary(0, "Stand", 100, 200, 0.0f, 0u, 1.0f, 0, 0, 0u, Vector3.Zero, Vector3.One, 1.0f)],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            null,
            [],
            [],
            [],
            [],
            0,
            0);
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
