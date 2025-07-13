using System;
using System.Numerics;

namespace WoWToolbox.Core.v2.Foundation.WMO.V17.Chunks
{
        using MOHDHeader = WoWToolbox.Core.v2.Foundation.WMO.V14.Models.MOHDHeader;

    // NOTE: Layouts mirror wow.export. These are plain data holders – parsing is handled elsewhere.

    public readonly record struct ChunkMver(uint Version);




    public readonly record struct MOMTMaterial(
        uint Flags,
        uint Shader,
        uint BlendMode,
        uint Texture1,
        uint Color1,
        uint Color1b,
        uint Texture2,
        uint Color2,
        uint GroupType,
        uint Texture3,
        uint Color3,
        uint Flags3,
        uint Runtime0,
        uint Runtime1,
        uint Runtime2,
        uint Runtime3);

    public readonly record struct MOGIEntry(
        uint Flags,
        Vector3 BoundingBox1,
        Vector3 BoundingBox2,
        int NameIndex);

    public readonly record struct PortalVertex(Vector3 Position);

    public readonly record struct PortalTriangle(
        ushort StartVertex,
        ushort Count,
        Vector4 Plane);

    public readonly record struct PortalReference(
        ushort PortalIndex,
        ushort GroupIndex,
        short Side);

    public readonly record struct FogEntry(
        uint Flags,
        Vector3 Position,
        float RadiusSmall,
        float RadiusLarge,
        float FogEnd,
        float FogStartScalar,
        uint FogColor,
        float UnderEnd,
        float UnderStartScalar,
        uint UnderColor);

    public readonly record struct DoodadSet(
        string Name,
        uint FirstInstanceIndex,
        uint DoodadCount,
        uint Unused);

    public readonly record struct DoodadDef(
        uint NameOffset24AndFlags8,
        Vector3 Position,
        Vector4 Rotation,
        float Scale,
        byte ColorA,
        byte ColorB,
        byte ColorC,
        byte ColorD);

    public readonly record struct MliqHeader(
        uint VertX,
        uint VertY,
        uint TileX,
        uint TileY,
        Vector3 Corner,
        ushort MaterialID);
}
