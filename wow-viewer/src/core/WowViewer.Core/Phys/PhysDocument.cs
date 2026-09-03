using System.Numerics;

namespace WowViewer.Core.Phys;

/// <summary>
/// Shape kinds declared by a <c>SHAP</c> record. Values are MEASURED: each branch of the 5.0.1 shape
/// builder bounds-checks the array matching the value below.
/// </summary>
public enum PhysShapeKind : ushort
{
    Box = 0,
    Capsule = 1,
    Sphere = 2,
}

/// <summary>
/// Joint kinds declared by a <c>JOIN</c> record. Names are MEASURED from the client's own
/// <c>PhysData.h</c> bounds asserts (<c>m_sphericalJointCount</c>, <c>m_shoulderJointCount</c>,
/// <c>m_weldJointCount</c>), not inherited from a community layout.
/// </summary>
public enum PhysJointKind : ushort
{
    Spherical = 0,
    Shoulder = 1,
    Weld = 2,
}

/// <summary>Severity of a <see cref="PhysDiagnostic"/>.</summary>
public enum PhysDiagnosticSeverity
{
    Info = 0,
    Warning = 1,
    Error = 2,
}

/// <summary>
/// One explicit reason a construct was skipped, could not be read, or is not understood. The 5.0.1
/// client fails closed silently; spec 214 requires the same fallback but never the same silence, so
/// nothing is dropped without one of these.
/// </summary>
public readonly record struct PhysDiagnostic(
    PhysDiagnosticSeverity Severity,
    string Code,
    string Message);

/// <summary>
/// A <c>BOXS</c> record (60 bytes). Only the vector at +48 is read by the 5.0.1 shape builder.
/// </summary>
/// <param name="UnverifiedPrefix">
/// Bytes 0..47, preserved verbatim. The 5.0.1 shape builder never reads them on this path, so their
/// meaning is UNVERIFIED. A 4x3 transform is a plausible inference and is deliberately NOT applied.
/// </param>
/// <param name="HalfExtents">
/// MEASURED at +48 and passed to the shape descriptor. The offset and the read are measured; the
/// <em>half-extents</em> interpretation is INFERRED and has not been confirmed against real bytes.
/// </param>
public readonly record struct PhysBoxShape(
    ReadOnlyMemory<byte> UnverifiedPrefix,
    Vector3 HalfExtents);

/// <summary>
/// A <c>CAPS</c> record (28 bytes). The three reads are MEASURED; naming the vectors as segment
/// endpoints and the scalar as a radius is INFERRED.
/// </summary>
public readonly record struct PhysCapsuleShape(
    Vector3 PointA,
    Vector3 PointB,
    float Radius);

/// <summary>A <c>SPHS</c> record (16 bytes). Fully MEASURED.</summary>
public readonly record struct PhysSphereShape(
    Vector3 Center,
    float Radius);

/// <summary>
/// A <c>SHAP</c> record (20 bytes): the indirection from a body to a typed shape.
/// </summary>
/// <param name="Kind">MEASURED. Selects which typed array <paramref name="Index"/> addresses.</param>
/// <param name="Index">MEASURED index into the array named by <paramref name="Kind"/>.</param>
/// <param name="UnverifiedAt4">Bytes 4..7, not read by the 5.0.1 builder. Preserved, not interpreted.</param>
/// <param name="UnverifiedAt8">
/// MEASURED as passed to the shape descriptor; semantics UNKNOWN. The community layout calls
/// +8/+12/+16 friction, restitution and density. That is not measured here and is not asserted.
/// </param>
/// <param name="UnverifiedAt12">See <paramref name="UnverifiedAt8"/>.</param>
/// <param name="UnverifiedAt16">See <paramref name="UnverifiedAt8"/>.</param>
public readonly record struct PhysShape(
    PhysShapeKind Kind,
    ushort Index,
    uint UnverifiedAt4,
    uint UnverifiedAt8,
    uint UnverifiedAt12,
    uint UnverifiedAt16);

/// <summary>
/// A <c>BODY</c> record (28 bytes). Fully MEASURED from the 5.0.1 body builder.
/// </summary>
/// <param name="RawType">
/// Raw declared type. The client remaps 0 to 1, 1 to 0, and anything else to 2 before handing it to
/// the solver. The raw value is preserved here so the remap stays an adapter concern.
/// </param>
/// <param name="BoneIndex">
/// Model bone this body is bound to. <see cref="NoBone"/> is a real sentinel and must be preserved,
/// never clamped.
/// </param>
public readonly record struct PhysBody(
    ushort RawType,
    Vector3 Position,
    ushort BoneIndex,
    uint FirstShapeIndex,
    uint ShapeCount)
{
    /// <summary>MEASURED: the 5.0.1 driving loop skips bodies carrying this bone index.</summary>
    public const ushort NoBone = 0xFFFF;

    public bool HasBone => BoneIndex != NoBone;
}

/// <summary>
/// A <c>JOIN</c> record (16 bytes). Fully MEASURED. <paramref name="BodyAIndex"/> and
/// <paramref name="BodyBIndex"/> address the body array; <paramref name="Index"/> addresses the
/// typed joint array named by <paramref name="Kind"/>.
/// </summary>
public readonly record struct PhysJoint(
    uint BodyAIndex,
    uint BodyBIndex,
    uint UnverifiedAt8,
    PhysJointKind Kind,
    ushort Index);

/// <summary>
/// A typed joint payload whose per-field semantics are NOT decoded. The record length is validated
/// against the measured stride and the bytes are preserved verbatim, so presence and count are
/// reportable without inventing a layout.
/// </summary>
public readonly record struct PhysOpaqueRecord(ReadOnlyMemory<byte> Bytes);

/// <summary>
/// A parsed <c>.phys</c> sidecar.
/// </summary>
/// <remarks>
/// Layouts are recovered from the 5.0.1 client (see
/// <c>specs/214-mop-physics-domino/evidence/physics-contract.md</c>). Nothing here has been validated
/// against real client bytes yet; that is spec 214 task T002.
/// </remarks>
public sealed class PhysDocument
{
    /// <summary>MEASURED: 5.0.1 accepts only version 0 in the <c>PHYS</c> payload.</summary>
    public const ushort SupportedVersion = 0;

    public PhysDocument(
        ushort version,
        IReadOnlyList<PhysBoxShape> boxShapes,
        IReadOnlyList<PhysCapsuleShape> capsuleShapes,
        IReadOnlyList<PhysSphereShape> sphereShapes,
        IReadOnlyList<PhysShape> shapes,
        IReadOnlyList<PhysBody> bodies,
        IReadOnlyList<PhysJoint> joints,
        IReadOnlyList<PhysOpaqueRecord> sphericalJoints,
        IReadOnlyList<PhysOpaqueRecord> shoulderJoints,
        IReadOnlyList<PhysOpaqueRecord> weldJoints,
        IReadOnlyList<PhysDiagnostic> diagnostics)
    {
        Version = version;
        BoxShapes = boxShapes;
        CapsuleShapes = capsuleShapes;
        SphereShapes = sphereShapes;
        Shapes = shapes;
        Bodies = bodies;
        Joints = joints;
        SphericalJoints = sphericalJoints;
        ShoulderJoints = shoulderJoints;
        WeldJoints = weldJoints;
        Diagnostics = diagnostics;
    }

    public ushort Version { get; }

    public IReadOnlyList<PhysBoxShape> BoxShapes { get; }

    public IReadOnlyList<PhysCapsuleShape> CapsuleShapes { get; }

    public IReadOnlyList<PhysSphereShape> SphereShapes { get; }

    public IReadOnlyList<PhysShape> Shapes { get; }

    public IReadOnlyList<PhysBody> Bodies { get; }

    public IReadOnlyList<PhysJoint> Joints { get; }

    public IReadOnlyList<PhysOpaqueRecord> SphericalJoints { get; }

    public IReadOnlyList<PhysOpaqueRecord> ShoulderJoints { get; }

    public IReadOnlyList<PhysOpaqueRecord> WeldJoints { get; }

    /// <summary>Every skipped, malformed or unrecognised construct, with a reason. Never silent.</summary>
    public IReadOnlyList<PhysDiagnostic> Diagnostics { get; }

    public bool HasSimulatableContent => Bodies.Count > 0;
}
