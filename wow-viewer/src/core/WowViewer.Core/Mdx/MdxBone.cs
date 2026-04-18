using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxBone
{
    public const uint IgnoreParentTranslateFlag = 0x1;
    public const uint IgnoreParentScaleFlag = 0x2;
    public const uint IgnoreParentRotationFlag = 0x4;
    public const uint SphericalBillboardFlag = 0x8;
    public const uint CylindricalBillboardLockXFlag = 0x10;
    public const uint CylindricalBillboardLockYFlag = 0x20;
    public const uint CylindricalBillboardLockZFlag = 0x40;

    public MdxBone(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        uint geosetId,
        uint geosetAnimationId,
        Vector3 pivotPoint,
        MdxVector3NodeTrack? translationTrack,
        MdxQuaternionNodeTrack? rotationTrack,
        MdxVector3NodeTrack? scalingTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        GeosetId = geosetId;
        GeosetAnimationId = geosetAnimationId;
        PivotPoint = pivotPoint;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public uint GeosetId { get; }

    public uint GeosetAnimationId { get; }

    public Vector3 PivotPoint { get; }

    public bool HasParent => ParentId >= 0;

    public bool UsesGeoset => GeosetId != uint.MaxValue;

    public bool UsesGeosetAnimation => GeosetAnimationId != uint.MaxValue;

    public bool IgnoresParentTranslation => (Flags & IgnoreParentTranslateFlag) != 0;

    public bool IgnoresParentScale => (Flags & IgnoreParentScaleFlag) != 0;

    public bool IgnoresParentRotation => (Flags & IgnoreParentRotationFlag) != 0;

    public bool IsSphericalBillboard => (Flags & SphericalBillboardFlag) != 0;

    public bool IsCylindricalBillboardLockX => (Flags & CylindricalBillboardLockXFlag) != 0;

    public bool IsCylindricalBillboardLockY => (Flags & CylindricalBillboardLockYFlag) != 0;

    public bool IsCylindricalBillboardLockZ => (Flags & CylindricalBillboardLockZFlag) != 0;

    public bool IsBillboard => IsSphericalBillboard || IsCylindricalBillboardLockX || IsCylindricalBillboardLockY || IsCylindricalBillboardLockZ;

    public MdxVector3NodeTrack? TranslationTrack { get; }

    public MdxQuaternionNodeTrack? RotationTrack { get; }

    public MdxVector3NodeTrack? ScalingTrack { get; }
}
