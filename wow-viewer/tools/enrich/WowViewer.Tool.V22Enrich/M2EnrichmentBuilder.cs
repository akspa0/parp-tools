using WowViewer.Core.IO.Maps;
using WowViewer.Core.M2;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.V22Enrich;

/// <summary>
/// Builds V22 enrichment entries from M2 geometry + skin data.
/// </summary>
static class M2EnrichmentBuilder
{
    /// <summary>
    /// Create an EnrichmentEntry for an M2 model. Flattens M2GeometryDocument
    /// and M2SkinDocument into named arrays matching the V22 spec (FR-008).
    /// </summary>
    public static EnrichmentEntry BuildEntry(string path, M2GeometryDocument geo, M2SkinDocument skin)
    {
        var arrays = new List<EnrichmentArray>();

        // ── Vertices (V, 3) ────────────────────────────────────
        int vCount = geo.Vertices.Count;
        float[] verts = new float[vCount * 3];
        for (int i = 0; i < vCount; i++)
        {
            verts[i * 3 + 0] = geo.Vertices[i].Position.X;
            verts[i * 3 + 1] = geo.Vertices[i].Position.Y;
            verts[i * 3 + 2] = geo.Vertices[i].Position.Z;
        }
        arrays.Add(new EnrichmentArray("vertices", [vCount, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(verts)));

        // ── Normals (V, 3) ─────────────────────────────────────
        float[] norms = new float[vCount * 3];
        for (int i = 0; i < vCount; i++)
        {
            norms[i * 3 + 0] = geo.Vertices[i].Normal.X;
            norms[i * 3 + 1] = geo.Vertices[i].Normal.Y;
            norms[i * 3 + 2] = geo.Vertices[i].Normal.Z;
        }
        arrays.Add(new EnrichmentArray("normals", [vCount, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(norms)));

        // ── Texcoords (V, 2) ───────────────────────────────────
        float[] uv0 = new float[vCount * 2];
        float[] uv1 = new float[vCount * 2];
        for (int i = 0; i < vCount; i++)
        {
            uv0[i * 2 + 0] = geo.Vertices[i].TextureCoords0.X;
            uv0[i * 2 + 1] = geo.Vertices[i].TextureCoords0.Y;
            uv1[i * 2 + 0] = geo.Vertices[i].TextureCoords1.X;
            uv1[i * 2 + 1] = geo.Vertices[i].TextureCoords1.Y;
        }
        arrays.Add(new EnrichmentArray("texcoords_0", [vCount, 2], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(uv0)));
        arrays.Add(new EnrichmentArray("texcoords_1", [vCount, 2], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(uv1)));

        // ── Bone indices (V, 4) uint8 ──────────────────────────
        byte[] boneIdx = new byte[vCount * 4];
        for (int i = 0; i < vCount; i++)
        {
            boneIdx[i * 4 + 0] = (byte)geo.Vertices[i].BoneIndices.X;
            boneIdx[i * 4 + 1] = (byte)geo.Vertices[i].BoneIndices.Y;
            boneIdx[i * 4 + 2] = (byte)geo.Vertices[i].BoneIndices.Z;
            boneIdx[i * 4 + 3] = (byte)geo.Vertices[i].BoneIndices.W;
        }
        arrays.Add(new EnrichmentArray("bone_indices", [vCount, 4], typeof(byte), boneIdx));

        // ── Bone weights (V, 4) float ──────────────────────────
        float[] boneW = new float[vCount * 4];
        for (int i = 0; i < vCount; i++)
        {
            boneW[i * 4 + 0] = geo.Vertices[i].BoneWeights.X;
            boneW[i * 4 + 1] = geo.Vertices[i].BoneWeights.Y;
            boneW[i * 4 + 2] = geo.Vertices[i].BoneWeights.Z;
            boneW[i * 4 + 3] = geo.Vertices[i].BoneWeights.W;
        }
        arrays.Add(new EnrichmentArray("bone_weights", [vCount, 4], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(boneW)));

        // ── Triangles (M, 3) int32 from skin ───────────────────
        int tCount = skin.TriangleIndices.Count / 3;
        int[] tris = new int[skin.TriangleIndices.Count];
        for (int i = 0; i < skin.TriangleIndices.Count; i++)
        {
            ushort index = skin.TriangleIndices[i];
            tris[i] = index < skin.VertexLookup.Count ? skin.VertexLookup[index] : index;
        }
        arrays.Add(new EnrichmentArray("triangles", [tCount, 3], typeof(int),
            EnrichmentArrayHelper.FlattenInts(tris)));

        // ── Render flags (R,) uint32 ───────────────────────────
        int rCount = geo.RenderFlags.Count;
        uint[] rFlags = new uint[rCount];
        for (int i = 0; i < rCount; i++)
            rFlags[i] = geo.RenderFlags[i].Flags;
        arrays.Add(new EnrichmentArray("render_flags", [rCount], typeof(uint),
            EnrichmentArrayHelper.FlattenUInts(rFlags)));

        // ── Blend modes (R,) uint8 ─────────────────────────────
        byte[] blendModes = new byte[rCount];
        for (int i = 0; i < rCount; i++)
            blendModes[i] = (byte)geo.RenderFlags[i].BlendMode;
        arrays.Add(new EnrichmentArray("blend_modes", [rCount], typeof(byte), blendModes));

        // ── Texture lookup (T,) uint16 ─────────────────────────
        int tlCount = geo.TextureLookup.Count;
        byte[] texLookup = new byte[tlCount * 2];
        for (int i = 0; i < tlCount; i++)
        {
            ushort val = geo.TextureLookup[i].TextureId;
            texLookup[i * 2 + 0] = (byte)(val & 0xFF);
            texLookup[i * 2 + 1] = (byte)((val >> 8) & 0xFF);
        }
        arrays.Add(new EnrichmentArray("texture_lookup", [tlCount], typeof(ushort), texLookup));

        int textureCount = geo.Textures.Count;
        uint[] replaceableIds = new uint[textureCount];
        uint[] textureFlags = new uint[textureCount];
        for (int i = 0; i < textureCount; i++)
        {
            replaceableIds[i] = geo.Textures[i].ReplaceableId;
            textureFlags[i] = geo.Textures[i].Flags;
        }
        arrays.Add(new EnrichmentArray("texture_replaceable_ids", [textureCount], typeof(uint),
            EnrichmentArrayHelper.FlattenUInts(replaceableIds)));
        arrays.Add(new EnrichmentArray("texture_flags", [textureCount], typeof(uint),
            EnrichmentArrayHelper.FlattenUInts(textureFlags)));

        // ── Texture paths (P,) string table encoded as uint8 blob ──
        string[] texturePaths = new string[textureCount];
        for (int i = 0; i < textureCount; i++)
            texturePaths[i] = geo.Textures[i].Filename ?? string.Empty;
        byte[] texturePathBlob = EnrichmentArrayHelper.FlattenStrings(texturePaths);
        arrays.Add(new EnrichmentArray("texture_paths", [texturePathBlob.Length], typeof(byte), texturePathBlob));

        // ── Transparency lookup (R,) uint16 ────────────────────
        int trlCount = geo.TransparencyLookup.Count;
        byte[] transpLookup = new byte[trlCount * 2];
        for (int i = 0; i < trlCount; i++)
        {
            ushort val = geo.TransparencyLookup[i].TransparencyIndex;
            transpLookup[i * 2 + 0] = (byte)(val & 0xFF);
            transpLookup[i * 2 + 1] = (byte)((val >> 8) & 0xFF);
        }
        arrays.Add(new EnrichmentArray("transparency_lookup", [trlCount], typeof(ushort), transpLookup));

        // ── Bone lookup (B,) uint16 ────────────────────────────
        int blCount = geo.BoneLookup.Count;
        byte[] boneLookup = new byte[blCount * 2];
        for (int i = 0; i < blCount; i++)
        {
            ushort val = geo.BoneLookup[i].BoneIndex;
            boneLookup[i * 2 + 0] = (byte)(val & 0xFF);
            boneLookup[i * 2 + 1] = (byte)((val >> 8) & 0xFF);
        }
        arrays.Add(new EnrichmentArray("bone_lookup", [blCount], typeof(ushort), boneLookup));

        // ── Bounds (2, 3) float32 ──────────────────────────────
        var boundsMin = geo.Model.BoundsMin;
        var boundsMax = geo.Model.BoundsMax;
        float[] bounds = [boundsMin.X, boundsMin.Y, boundsMin.Z, boundsMax.X, boundsMax.Y, boundsMax.Z];
        arrays.Add(new EnrichmentArray("bounds", [2, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(bounds)));

        return new EnrichmentEntry(path, AssetKind.M2, 0, arrays);
    }
}
