using System.Numerics;

namespace WowViewer.Core.M2;

public enum M2BlendMode : byte
{
    Opaque = 0,
    AlphaKey = 1,
    AlphaBlend = 2,
    NoAlphaAdd = 3,
    Add = 4,
    Mod = 5,
    Mod2X = 6,
    BlendAdd = 7,
    Unknown = byte.MaxValue,
}

public sealed class M2GeometryDocument
{
    public M2GeometryDocument(
        M2ModelDocument model,
        IReadOnlyList<M2GeometryVertex> vertices,
        IReadOnlyList<M2GeometryTexture> textures,
        IReadOnlyList<M2GeometryRenderFlag> renderFlags,
        IReadOnlyList<M2GeometryTextureLookup> textureLookup)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(textures);
        ArgumentNullException.ThrowIfNull(renderFlags);
        ArgumentNullException.ThrowIfNull(textureLookup);

        Model = model;
        Vertices = vertices;
        Textures = textures;
        RenderFlags = renderFlags;
        TextureLookup = textureLookup;
    }

    public M2ModelDocument Model { get; }

    public IReadOnlyList<M2GeometryVertex> Vertices { get; }

    public IReadOnlyList<M2GeometryTexture> Textures { get; }

    public IReadOnlyList<M2GeometryRenderFlag> RenderFlags { get; }

    public IReadOnlyList<M2GeometryTextureLookup> TextureLookup { get; }
}

public readonly record struct M2GeometryVertex(
    Vector3 Position,
    Vector3 Normal,
    Vector2 TextureCoords0,
    Vector2 TextureCoords1,
    Vector4 BoneIndices,
    Vector4 BoneWeights);

public sealed class M2GeometryTexture
{
    public M2GeometryTexture(string? filename, uint replaceableId, uint flags)
    {
        Filename = string.IsNullOrWhiteSpace(filename)
            ? null
            : filename.Replace('/', '\\');
        ReplaceableId = replaceableId;
        Flags = flags;
    }

    public string? Filename { get; }

    public uint ReplaceableId { get; }

    public uint Flags { get; }
}

public sealed class M2GeometryRenderFlag
{
    public M2GeometryRenderFlag(ushort flags, ushort rawBlendMode)
    {
        Flags = flags;
        RawBlendMode = rawBlendMode;
        BlendMode = Enum.IsDefined(typeof(M2BlendMode), (byte)rawBlendMode)
            ? (M2BlendMode)(byte)rawBlendMode
            : M2BlendMode.Unknown;
    }

    public ushort Flags { get; }

    public ushort RawBlendMode { get; }

    public M2BlendMode BlendMode { get; }
}

public sealed class M2GeometryTextureLookup
{
    public M2GeometryTextureLookup(ushort textureId)
    {
        TextureId = textureId;
    }

    public ushort TextureId { get; }
}