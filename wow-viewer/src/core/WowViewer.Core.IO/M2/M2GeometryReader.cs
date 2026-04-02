using System.Numerics;
using Warcraft.NET.Files.M2;
using Warcraft.NET.Files.M2.Chunks;
using Warcraft.NET.Files.M2.Entries;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2GeometryReader
{
    public static M2GeometryDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static M2GeometryDocument Read(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        M2ModelDocument model = M2ModelReader.Read(stream, sourcePath);
        byte[] data = ReadAllBytes(stream);
        MD21 parsedModel = ParseModelData(data, sourcePath);

        List<M2GeometryVertex> vertices = new(parsedModel.Vertices?.Count ?? 0);
        if (parsedModel.Vertices != null)
        {
            foreach (VerticeStruct vertex in parsedModel.Vertices)
            {
                vertices.Add(new M2GeometryVertex(
                    vertex.Position,
                    vertex.Normal,
                    new Vector2(vertex.TextureCoordX, vertex.TextureCoordY),
                    new Vector2(vertex.TextureCoordX2, vertex.TextureCoordY2),
                    new Vector4(vertex.BoneIndices0, vertex.BoneIndices1, vertex.BoneIndices2, vertex.BoneIndices3),
                    new Vector4(vertex.BoneWeight0 / 255f, vertex.BoneWeight1 / 255f, vertex.BoneWeight2 / 255f, vertex.BoneWeight3 / 255f)));
            }
        }

        List<M2GeometryTexture> textures = new(parsedModel.Textures?.Count ?? 0);
        if (parsedModel.Textures != null)
        {
            foreach (TextureStruct texture in parsedModel.Textures)
                textures.Add(new M2GeometryTexture(texture.Filename, (uint)texture.Type, (uint)texture.Flags));
        }

        List<M2GeometryRenderFlag> renderFlags = new(parsedModel.RenderFlags?.Count ?? 0);
        if (parsedModel.RenderFlags != null)
        {
            foreach (RenderFlagStruct renderFlag in parsedModel.RenderFlags)
                renderFlags.Add(new M2GeometryRenderFlag(renderFlag.Flags, renderFlag.BlendingMode));
        }

        List<M2GeometryTextureLookup> textureLookup = new(parsedModel.TextureLookup?.Count ?? 0);
        if (parsedModel.TextureLookup != null)
        {
            foreach (TextureLookupStruct lookup in parsedModel.TextureLookup)
                textureLookup.Add(new M2GeometryTextureLookup(lookup.TextureID));
        }

        return new M2GeometryDocument(model, vertices, textures, renderFlags, textureLookup);
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] data = new byte[checked((int)stream.Length)];
            stream.ReadExactly(data);
            return data;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static MD21 ParseModelData(byte[] data, string sourcePath)
    {
        try
        {
            return new MD21(data);
        }
        catch (Exception md21Ex)
        {
            try
            {
                Model model = new(data);
                if (model.ModelInformation != null)
                    return model.ModelInformation;
            }
            catch (Exception wrapperEx)
            {
                throw new InvalidDataException(
                    $"Failed to parse M2 geometry payload for '{Path.GetFileName(sourcePath)}'.",
                    new AggregateException(md21Ex, wrapperEx));
            }

            throw new InvalidDataException($"M2 geometry payload for '{Path.GetFileName(sourcePath)}' did not expose MD21 model information.", md21Ex);
        }
    }
}