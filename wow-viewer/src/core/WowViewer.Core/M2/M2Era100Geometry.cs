using System.Numerics;

namespace WowViewer.Core.M2;

/// <summary>
/// Resolved geometry from a WoW 1.0.0 M2 model's embedded division (skin profile).
/// The vertex lookup has already been applied: RenderVertices are in division-local order,
/// ready for direct upload to a vertex buffer. Triangles index into RenderVertices.
/// </summary>
public sealed class M2Era100Geometry
{
    public M2Era100Geometry(
        IReadOnlyList<M2Era100Vertex> renderVertices,
        IReadOnlyList<ushort> triangles,
        IReadOnlyList<M2Era100Section> sections,
        IReadOnlyList<M2Era100Batch> batches,
        IReadOnlyList<M2Era100Texture> textures,
        IReadOnlyList<short> textureLookup,
        IReadOnlyList<M2Era100Material>? materials = null)
    {
        ArgumentNullException.ThrowIfNull(renderVertices);
        ArgumentNullException.ThrowIfNull(triangles);
        ArgumentNullException.ThrowIfNull(sections);
        ArgumentNullException.ThrowIfNull(batches);
        ArgumentNullException.ThrowIfNull(textures);
        ArgumentNullException.ThrowIfNull(textureLookup);

        RenderVertices = renderVertices;
        Triangles = triangles;
        Sections = sections;
        Batches = batches;
        Textures = textures;
        TextureLookup = textureLookup;
        Materials = materials ?? [];
    }

    /// <summary>Vertices in division-local order (vertexLookup already applied).</summary>
    public IReadOnlyList<M2Era100Vertex> RenderVertices { get; }

    /// <summary>Triangle index buffer (indices into RenderVertices).</summary>
    public IReadOnlyList<ushort> Triangles { get; }

    /// <summary>Render sections / geometry groups from the division.</summary>
    public IReadOnlyList<M2Era100Section> Sections { get; }

    /// <summary>Material/texture bindings from the division.</summary>
    public IReadOnlyList<M2Era100Batch> Batches { get; }

    public IReadOnlyList<M2Era100Texture> Textures { get; }

    /// <summary>
    /// textureCombos (header 0x94). Maps batch.textureComboIndex + stage → texture array index.
    /// SIGNED: a negative entry means a replaceable texture supplied at runtime (character skin,
    /// creature skin), resolved as slot <c>~value</c> rather than indexing <see cref="Textures"/>.
    /// </summary>
    public IReadOnlyList<short> TextureLookup { get; }

    /// <summary>M2Material[] (header 0x84) — render flags and blend mode per batch.MaterialIndex.</summary>
    public IReadOnlyList<M2Era100Material> Materials { get; }
}

/// <summary>
/// M2Material (header 0x84, 4 bytes): {uint16 flags, uint16 blendMode}.
/// FUN_0071a910 reads flags bit 0x01 = unlit and bit 0x02 = unfogged; blendMode 3/4 are additive.
/// </summary>
public readonly record struct M2Era100Material(ushort Flags, ushort BlendMode)
{
    public bool Unlit => (Flags & 0x01) != 0;
    public bool Unfogged => (Flags & 0x02) != 0;
    public bool TwoSided => (Flags & 0x04) != 0;
}

/// <summary>
/// A resolved render vertex from a 1.0.0 M2 model. The global M2Vertex (0x30 bytes)
/// has been looked up through the division's vertexLookup and split into its components.
/// </summary>
public readonly record struct M2Era100Vertex(
    Vector3 Position,
    Vector3 Normal,
    Vector2 TexCoord0,
    Vector2 TexCoord1,
    byte BoneWeight0,
    byte BoneWeight1,
    byte BoneWeight2,
    byte BoneWeight3,
    byte BoneIndex0,
    byte BoneIndex1,
    byte BoneIndex2,
    byte BoneIndex3);

/// <summary>
/// Render section / geometry group (0x20 bytes on disk). Defines a contiguous slice
/// of the vertex/index buffers that shares a material.
/// VertexStart and IndexStart are stored on disk as uint16 with their high bits in Level;
/// both are already combined here.
/// </summary>
public readonly record struct M2Era100Section(
    ushort SubmeshId,
    ushort Level,
    uint VertexStart,
    ushort VertexCount,
    uint IndexStart,
    uint IndexCount);

/// <summary>
/// Material/texture binding (0x18 = 24 bytes on disk). Associates a section with
/// textures and render state.
/// </summary>
public readonly record struct M2Era100Batch(
    byte Flags,
    byte PriorityPlane,
    ushort ShaderId,
    ushort SkinSectionIndex,
    ushort GeosetIndex,
    ushort ColorIndex,
    ushort MaterialIndex,
    ushort MaterialLayer,
    ushort TextureCount,
    ushort TextureComboIndex,
    ushort TextureCoordComboIndex,
    ushort TextureWeightComboIndex,
    ushort TextureTransformComboIndex);

/// <summary>
/// Texture definition (0x10 = 16 bytes on disk): {type, flags, nameLen, nameOfs}.
/// </summary>
public sealed class M2Era100Texture
{
    public M2Era100Texture(uint type, uint flags, string filename)
    {
        Type = type;
        Flags = flags;
        Filename = filename;
    }

    public uint Type { get; }
    public uint Flags { get; }
    public string Filename { get; }
}