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
        var allTris = new List<int>();
        int[] groupVertexCounts = new int[groups.Count];
        int[] groupTriCounts = new int[groups.Count];
        int vertexOffset = 0;
        int triOffset = 0;

        for (int g = 0; g < groups.Count; g++)
        {
            var group = groups[g];
            groupVertexCounts[g] = group.Vertices?.Count ?? 0;
            groupTriCounts[g] = group.Triangles?.Count ?? 0;

            if (group.Vertices is not null)
                allVerts.AddRange(group.Vertices);

            if (group.Triangles is not null)
            {
                foreach (var tri in group.Triangles)
                {
                    allTris.Add(tri.Index0 + vertexOffset);
                    allTris.Add(tri.Index1 + vertexOffset);
                    allTris.Add(tri.Index2 + vertexOffset);
                }
            }

            vertexOffset += groupVertexCounts[g];
            triOffset += groupTriCounts[g] * 3;
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
            mats[i * 8 + 0] = mat.Flags;
            mats[i * 8 + 1] = mat.Shader;
            mats[i * 8 + 2] = mat.BlendMode;
            mats[i * 8 + 3] = mat.TextureIndex0;
            mats[i * 8 + 4] = mat.TextureIndex1;
            mats[i * 8 + 5] = mat.TextureIndex2;
            mats[i * 8 + 6] = mat.TextureIndex3;
            mats[i * 8 + 7] = mat.TextureIndex4;
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
            pv[i * 3 + 0] = wmo.PortalVertices[i].X;
            pv[i * 3 + 1] = wmo.PortalVertices[i].Y;
            pv[i * 3 + 2] = wmo.PortalVertices[i].Z;
        }
        arrays.Add(new EnrichmentArray("portal_vertices", [pvCount, 3], typeof(float),
            EnrichmentArrayHelper.FlattenFloats(pv)));

        // ── Portal indices (PI, 3) int32 ────────────────────────
        int piCount = wmo.PortalIndices.Count;
        int[] pi = new int[piCount * 3];
        for (int i = 0; i < piCount; i++)
        {
            pi[i * 3 + 0] = wmo.PortalIndices[i].Index0;
            pi[i * 3 + 1] = wmo.PortalIndices[i].Index1;
            pi[i * 3 + 2] = wmo.PortalIndices[i].Index2;
        }
        arrays.Add(new EnrichmentArray("portal_indices", [piCount, 3], typeof(int),
            EnrichmentArrayHelper.FlattenInts(pi)));

        // ── Flags uint32 ───────────────────────────────────────
        arrays.Add(new EnrichmentArray("flags", [1], typeof(uint),
            EnrichmentArrayHelper.FlattenUInts([wmo.Summary.Flags])));

        // ── Version uint32 ─────────────────────────────────────
        arrays.Add(new EnrichmentArray("version", [1], typeof(uint),
            EnrichmentArrayHelper.FlattenUInts([wmo.Version])));

        // ── Skip string arrays in this pass (doodad_set_paths, material_texture_paths) ──
        // These are recoverable from the model path or the V18 metadata.

        return new EnrichmentEntry(path, AssetKind.Wmo, 0, arrays);
    }
}
