using System.Linq;
using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Wmo;

namespace WowViewer.Tool.V22Enrich;

/// <summary>
/// Builds V22 enrichment entries from WMO render data.
/// </summary>
static class WmoEnrichmentBuilder
{
    /// <summary>
    /// Create an EnrichmentEntry for a WMO model. Flattens WmoRenderDocument
    /// into named arrays matching the V22 spec (FR-009).
    /// </summary>
    public static EnrichmentEntry BuildEntry(string path, WmoRenderDocument wmo)
    {
        var arrays = new List<EnrichmentArray>();
        var groups = wmo.Groups;

        if (groups.Count == 0)
        {
            return new EnrichmentEntry(path, AssetKind.Wmo, 1, []);
        }

        // ── Merge all group vertices into one buffer ─────────────
        var allVerts = new List<Vector3>();
        var allNormals = new List<Vector3>();
        var allTris = new List<int>();
        int[] groupVertexCounts = new int[groups.Count];
        int[] groupTriCounts = new int[groups.Count];
        int vertexOffset = 0;

        for (int g = 0; g < groups.Count; g++)
        {
            var group = groups[g];
            groupVertexCounts[g] = group.Mesh.Vertices.Count;
            groupTriCounts[g] = group.Mesh.Indices.Count / 3;

            allVerts.AddRange(group.Mesh.Vertices);
            if (group.Mesh.Normals.Count == group.Mesh.Vertices.Count)
            {
                allNormals.AddRange(group.Mesh.Normals);
            }
            else
            {
                for (int i = 0; i < group.Mesh.Vertices.Count; i++)
                    allNormals.Add(Vector3.Zero);
            }

            for (int i = 0; i + 2 < group.Mesh.Indices.Count; i += 3)
            {
                allTris.Add(group.Mesh.Indices[i + 0] + vertexOffset);
                allTris.Add(group.Mesh.Indices[i + 1] + vertexOffset);
                allTris.Add(group.Mesh.Indices[i + 2] + vertexOffset);
            }

            vertexOffset += groupVertexCounts[g];
        }

        // ── Flatten vertices (N, 3) ─────────────────────────────
        float[] verts = new float[allVerts.Count * 3];
        for (int i = 0; i < allVerts.Count; i++)
        {
            verts[i * 3 + 0] = allVerts[i].X;
            verts[i * 3 + 1] = allVerts[i].Y;
            verts[i * 3 + 2] = allVerts[i].Z;
        }
        arrays.Add(new EnrichmentArray("vertices", [allVerts.Count, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(verts)));

        float[] normals = new float[allNormals.Count * 3];
        for (int i = 0; i < allNormals.Count; i++)
        {
            normals[i * 3 + 0] = allNormals[i].X;
            normals[i * 3 + 1] = allNormals[i].Y;
            normals[i * 3 + 2] = allNormals[i].Z;
        }
        arrays.Add(new EnrichmentArray("normals", [allNormals.Count, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(normals)));

        // ── Flatten triangles (M, 3) ────────────────────────────
        int[] tris = allTris.ToArray();
        int triCount = tris.Length / 3;
        arrays.Add(new EnrichmentArray("triangles", [triCount, 3], typeof(int),
            EnrichmentArrayHelper.FlattenInts(tris)));

        // ── Group counts (G,) int32 ─────────────────────────────
        arrays.Add(new EnrichmentArray("group_counts", [groups.Count], typeof(int),
            EnrichmentArrayHelper.FlattenInts(groupVertexCounts)));

        // ── Group indices (G,) int32 (start offsets) ────────────
        int[] groupOffsets = new int[groups.Count];
        int running = 0;
        for (int g = 0; g < groups.Count; g++)
        {
            groupOffsets[g] = running;
            running += groupVertexCounts[g];
        }
        arrays.Add(new EnrichmentArray("group_indices", [groups.Count], typeof(int),
            EnrichmentArrayHelper.FlattenInts(groupOffsets)));

        // ── Materials (K, 8) int32 ──────────────────────────────
        int matCount = wmo.Materials.Count;
        int[] mats = new int[matCount * 8];
        for (int i = 0; i < matCount; i++)
        {
            var mat = wmo.Materials[i];
            mats[i * 8 + 0] = unchecked((int)mat.Flags);
            mats[i * 8 + 1] = unchecked((int)mat.Shader);
            mats[i * 8 + 2] = unchecked((int)mat.BlendMode);
            mats[i * 8 + 3] = unchecked((int)mat.Texture1Offset);
            mats[i * 8 + 4] = unchecked((int)mat.Texture2Offset);
            mats[i * 8 + 5] = unchecked((int)mat.Texture3Offset);
            mats[i * 8 + 6] = mat.EntrySizeBytes;
            mats[i * 8 + 7] = mat.PayloadOffset;
        }
        arrays.Add(new EnrichmentArray("materials", [matCount, 8], typeof(int),
            EnrichmentArrayHelper.FlattenInts(mats)));

        // ── Bounds (2, 3) float32 ──────────────────────────────
        float[] bounds = [
            wmo.Summary.BoundsMin.X, wmo.Summary.BoundsMin.Y, wmo.Summary.BoundsMin.Z,
            wmo.Summary.BoundsMax.X, wmo.Summary.BoundsMax.Y, wmo.Summary.BoundsMax.Z,
        ];
        arrays.Add(new EnrichmentArray("bounds", [2, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(bounds)));

        // ── Portal vertices (PV, 3) ────────────────────────────
        int pvCount = wmo.PortalVertices.Count;
        float[] pv = new float[pvCount * 3];
        for (int i = 0; i < pvCount; i++)
        {
            pv[i * 3 + 0] = wmo.PortalVertices[i].Position.X;
            pv[i * 3 + 1] = wmo.PortalVertices[i].Position.Y;
            pv[i * 3 + 2] = wmo.PortalVertices[i].Position.Z;
        }
        arrays.Add(new EnrichmentArray("portal_vertices", [pvCount, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(pv)));

        // ── Portal indices (PI, 3) int32 ────────────────────────
        int piCount = wmo.Portals.Count;
        int[] pi = new int[piCount * 3];
        for (int i = 0; i < piCount; i++)
        {
            pi[i * 3 + 0] = wmo.Portals[i].PortalIndex;
            pi[i * 3 + 1] = wmo.Portals[i].StartVertexIndex;
            pi[i * 3 + 2] = wmo.Portals[i].VertexCount;
        }
        arrays.Add(new EnrichmentArray("portal_indices", [piCount, 3], typeof(int),
            EnrichmentArrayHelper.FlattenInts(pi)));

        // ── Flags uint32 ───────────────────────────────────────
        arrays.Add(new EnrichmentArray("flags", [1], typeof(uint),
            EnrichmentArrayHelper.FlattenUInts([wmo.Summary.Flags])));

        // ── Version uint32 ─────────────────────────────────────
        arrays.Add(new EnrichmentArray("version", [1], typeof(uint),
            EnrichmentArrayHelper.FlattenUInts([wmo.Version ?? 0])));

        // ── Material texture paths (P,) string table encoded as uint8 blob ──
        List<string> materialTexturePaths = [];
        foreach (var material in wmo.Materials)
        {
            if (!string.IsNullOrWhiteSpace(material.Texture1Name))
                materialTexturePaths.Add(material.Texture1Name);
            if (!string.IsNullOrWhiteSpace(material.Texture2Name))
                materialTexturePaths.Add(material.Texture2Name);
            if (!string.IsNullOrWhiteSpace(material.Texture3Name))
                materialTexturePaths.Add(material.Texture3Name);
        }
        byte[] materialTextureBlob = EnrichmentArrayHelper.FlattenStrings(materialTexturePaths);
        arrays.Add(new EnrichmentArray("material_texture_paths", [materialTextureBlob.Length], typeof(byte), materialTextureBlob));

        // ── Doodad set paths (DS,) string table encoded as uint8 blob ──
        string[] doodadSetPaths = wmo.DoodadSets.Select(set => set.Name).ToArray();
        byte[] doodadSetBlob = EnrichmentArrayHelper.FlattenStrings(doodadSetPaths);
        arrays.Add(new EnrichmentArray("doodad_set_paths", [doodadSetBlob.Length], typeof(byte), doodadSetBlob));

        return new EnrichmentEntry(path, AssetKind.Wmo, 0, arrays);
    }
}
