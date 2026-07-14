using System.Numerics;
using WowViewer.Core.M2;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Maps;

/// <summary>
/// Serialized payload of a single M2 or WMO model, designed for the V22 stream.
/// Flat byte arrays ready to write into the RawArraySerializer format.
/// </summary>
public sealed class V22ModelPayload
{
    public enum ModelKind : byte { Unknown, M2, Wmo }

    public ModelKind Kind { get; init; }
    public int LoadError { get; init; }
    public string CanonicalPath { get; init; } = "";
    public Dictionary<string, Array> RawArrays { get; init; } = new();
    public string[] TexturePaths { get; set; } = [];

    public static V22ModelPayload FromM2(
        string canonicalPath,
        M2ModelDocument model,
        M2GeometryDocument geometry,
        M2SkinDocument skin)
    {
        var payload = new V22ModelPayload
        {
            Kind = ModelKind.M2,
            CanonicalPath = canonicalPath,
            RawArrays = new Dictionary<string, Array>()
        };

        var verts = geometry.Vertices;
        int n = verts.Count;
        var positions = new float[n, 3];
        var normals = new float[n, 3];
        var tc0 = new float[n, 2];
        var tc1 = new float[n, 2];
        var boneIdx = new byte[n, 4];
        var boneWgt = new float[n, 4];
        for (int i = 0; i < n; i++)
        {
            positions[i, 0] = verts[i].Position.X;
            positions[i, 1] = verts[i].Position.Y;
            positions[i, 2] = verts[i].Position.Z;
            normals[i, 0] = verts[i].Normal.X;
            normals[i, 1] = verts[i].Normal.Y;
            normals[i, 2] = verts[i].Normal.Z;
            tc0[i, 0] = verts[i].TextureCoords0.X;
            tc0[i, 1] = verts[i].TextureCoords0.Y;
            tc1[i, 0] = verts[i].TextureCoords1.X;
            tc1[i, 1] = verts[i].TextureCoords1.Y;
            boneIdx[i, 0] = (byte)verts[i].BoneIndices.X;
            boneIdx[i, 1] = (byte)verts[i].BoneIndices.Y;
            boneIdx[i, 2] = (byte)verts[i].BoneIndices.Z;
            boneIdx[i, 3] = (byte)verts[i].BoneIndices.W;
            boneWgt[i, 0] = verts[i].BoneWeights.X;
            boneWgt[i, 1] = verts[i].BoneWeights.Y;
            boneWgt[i, 2] = verts[i].BoneWeights.Z;
            boneWgt[i, 3] = verts[i].BoneWeights.W;
        }
        payload.RawArrays["vertices"] = positions;
        payload.RawArrays["normals"] = normals;
        payload.RawArrays["texcoords_0"] = tc0;
        payload.RawArrays["texcoords_1"] = tc1;
        payload.RawArrays["bone_indices"] = boneIdx;
        payload.RawArrays["bone_weights"] = boneWgt;

        int tCount = skin.TriangleIndices.Count;
        var triangles = new int[tCount / 3, 3];
        for (int i = 0; i + 2 < tCount; i += 3)
        {
            triangles[i / 3, 0] = skin.TriangleIndices[i];
            triangles[i / 3, 1] = skin.TriangleIndices[i + 1];
            triangles[i / 3, 2] = skin.TriangleIndices[i + 2];
        }
        payload.RawArrays["triangles"] = triangles;

        int rCount = geometry.RenderFlags.Count;
        var renderFlags = new uint[rCount];
        var blendModes = new byte[rCount];
        var textureLookup = new ushort[rCount];
        var transparencyLookup = new ushort[rCount];
        for (int i = 0; i < rCount; i++)
        {
            renderFlags[i] = geometry.RenderFlags[i].Flags;
            blendModes[i] = (byte)geometry.RenderFlags[i].RawBlendMode;
            textureLookup[i] = i < geometry.TextureLookup.Count ? geometry.TextureLookup[i].TextureId : (ushort)0;
            transparencyLookup[i] = i < geometry.TransparencyLookup.Count ? geometry.TransparencyLookup[i].TransparencyIndex : (ushort)0;
        }
        payload.RawArrays["render_flags"] = renderFlags;
        payload.RawArrays["blend_modes"] = blendModes;
        payload.RawArrays["texture_lookup"] = textureLookup;
        payload.RawArrays["transparency_lookup"] = transparencyLookup;

        int pCount = geometry.Textures.Count;
        var texPaths = new string[pCount];
        var texReplaceableIds = new uint[pCount];
        var texFlags = new uint[pCount];
        for (int i = 0; i < pCount; i++)
        {
            texPaths[i] = geometry.Textures[i].Filename ?? "";
            texReplaceableIds[i] = geometry.Textures[i].ReplaceableId;
            texFlags[i] = geometry.Textures[i].Flags;
        }
        payload.TexturePaths = texPaths;
        payload.RawArrays["texture_replaceable_ids"] = texReplaceableIds;
        payload.RawArrays["texture_flags"] = texFlags;

        int bCount = geometry.BoneLookup.Count;
        var boneLookup = new ushort[bCount];
        for (int i = 0; i < bCount; i++)
            boneLookup[i] = geometry.BoneLookup[i].BoneIndex;
        payload.RawArrays["bone_lookup"] = boneLookup;

        payload.RawArrays["bounds"] = new float[,]
        {
            { model.BoundsMin.X, model.BoundsMin.Y, model.BoundsMin.Z },
            { model.BoundsMax.X, model.BoundsMax.Y, model.BoundsMax.Z }
        };

        return payload;
    }

    public static V22ModelPayload FromWmo(
        string canonicalPath,
        WmoRenderDocument document)
    {
        var payload = new V22ModelPayload
        {
            Kind = ModelKind.Wmo,
            CanonicalPath = canonicalPath,
            RawArrays = new Dictionary<string, Array>()
        };

        var allVerts = new List<Vector3>();
        var allTris = new List<int>();
        var groupCounts = new List<int>();
        var groupIndices = new List<int>();

        foreach (var group in document.Groups)
        {
            var mesh = group.Mesh;
            var gVerts = mesh.Vertices;
            var gTris = mesh.Indices;
            groupIndices.Add(allVerts.Count);
            groupCounts.Add(gVerts.Count);

            int baseIdx = allVerts.Count;
            foreach (var v in gVerts)
                allVerts.Add(v);
            foreach (var t in gTris)
                allTris.Add((int)t + baseIdx);
        }

        int vCount = allVerts.Count;
        var verts = new float[vCount, 3];
        for (int i = 0; i < vCount; i++)
        {
            verts[i, 0] = allVerts[i].X;
            verts[i, 1] = allVerts[i].Y;
            verts[i, 2] = allVerts[i].Z;
        }
        payload.RawArrays["vertices"] = verts;

        int triCount = allTris.Count / 3;
        var tris = new int[triCount, 3];
        for (int i = 0; i < triCount; i++)
        {
            tris[i, 0] = allTris[i * 3];
            tris[i, 1] = allTris[i * 3 + 1];
            tris[i, 2] = allTris[i * 3 + 2];
        }
        payload.RawArrays["triangles"] = tris;
        payload.RawArrays["group_counts"] = groupCounts.ToArray();
        payload.RawArrays["group_indices"] = groupIndices.ToArray();

        int mCount = document.Materials.Count;
        var materials = new int[mCount, 8];
        for (int i = 0; i < mCount; i++)
        {
            var m = document.Materials[i];
            materials[i, 0] = (int)m.Flags;
            materials[i, 1] = (int)m.Shader;
            materials[i, 2] = (int)m.BlendMode;
            materials[i, 3] = 0;
            materials[i, 4] = 0;
            materials[i, 5] = 0;
            materials[i, 6] = 0;
            materials[i, 7] = 0;
        }
        payload.RawArrays["materials"] = materials;

        payload.RawArrays["bounds"] = new float[,]
        {
            { document.Summary.BoundsMin.X, document.Summary.BoundsMin.Y, document.Summary.BoundsMin.Z },
            { document.Summary.BoundsMax.X, document.Summary.BoundsMax.Y, document.Summary.BoundsMax.Z }
        };

        payload.RawArrays["flags"] = new[] { document.Summary.Flags };
        payload.RawArrays["version"] = document.Version.HasValue ? new[] { document.Version.Value } : new uint[] { 0 };

        int pvCount = document.PortalVertices.Count;
        var pv = new float[pvCount, 3];
        for (int i = 0; i < pvCount; i++)
        {
            pv[i, 0] = document.PortalVertices[i].Position.X;
            pv[i, 1] = document.PortalVertices[i].Position.Y;
            pv[i, 2] = document.PortalVertices[i].Position.Z;
        }
        payload.RawArrays["portal_vertices"] = pv;

        // Portals are index pairs — write raw indices
        var portalIndices = new List<int>();
        foreach (var portal in document.Portals)
        {
            portalIndices.Add(portal.StartVertexIndex);
            portalIndices.Add(portal.VertexCount);
        }
        if (portalIndices.Count > 0)
            payload.RawArrays["portal_indices"] = portalIndices.ToArray();

        return payload;
    }
}