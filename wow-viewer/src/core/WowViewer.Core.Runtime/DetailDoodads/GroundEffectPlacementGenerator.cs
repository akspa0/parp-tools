using System.Numerics;
using WowViewer.Core.IO.Dbc;

namespace WowViewer.Core.Runtime.DetailDoodads;

/// <summary>
/// Evaluates terrain chunk geometry, texture layers, and DBC rules to generate detail doodad instances.
/// Conforms to WoW client rules:
/// - Slope limit: normal Z >= 0.4 (~66.42 deg). Steeper slopes are culled.
/// - Flag 0x1 (AlignToNormal): aligns model up-vector to terrain triangle normal.
/// - Flag 0x2 (IgnoreMCCV): plain white (0xFFFFFFFF) when set; interpolates MCCV vertex colors when unset.
/// - MCSH shadow map: darkens RGB by 30% (scale 0.7) when shadowed.
/// </summary>
public static class GroundEffectPlacementGenerator
{
    public const float SlopeNormalZThreshold = 0.4f;
    private const float SubCellSize = 533.33333f / 16f / 8f; // ~4.166667f
    private const float ChunkSize = 533.33333f / 16f;        // ~33.333333f

    public delegate GroundEffectTextureRecord? TextureLookupDelegate(uint effectId);
    public delegate GroundEffectDoodadRecord? DoodadLookupDelegate(uint doodadId);

    /// <summary>
    /// Computes the quaternion that rotates the upright vector (0, 0, 1) to align with the given terrain normal.
    /// </summary>
    public static Quaternion ComputeNormalAlignment(Vector3 normal)
    {
        normal = Vector3.Normalize(normal);
        Vector3 up = Vector3.UnitZ;
        float dot = Vector3.Dot(up, normal);

        if (dot >= 0.9999f)
            return Quaternion.Identity;

        if (dot <= -0.9999f)
            return Quaternion.CreateFromAxisAngle(Vector3.UnitX, MathF.PI);

        Vector3 axis = Vector3.Cross(up, normal);
        float s = MathF.Sqrt((1.0f + dot) * 2.0f);
        float invS = 1.0f / s;
        return Quaternion.Normalize(new Quaternion(axis.X * invS, axis.Y * invS, axis.Z * invS, 0.5f * s));
    }

    /// <summary>
    /// Evaluates detail doodads for a terrain chunk.
    /// </summary>
    public static List<DetailDoodadInstance> GenerateChunkDoodads(
        TerrainChunkPlacementInput chunk,
        TextureLookupDelegate textureLookup,
        DoodadLookupDelegate doodadLookup,
        float densityMultiplier = 1.0f)
    {
        var results = new List<DetailDoodadInstance>();
        if (chunk.Heights.Length < 145 || chunk.Layers.Count == 0 || densityMultiplier <= 0f)
            return results;

        // Build 145 vertex world positions, normals, and colors
        Span<Vector3> positions = stackalloc Vector3[145];
        Span<Vector3> normals = stackalloc Vector3[145];
        Span<Vector4> colors = stackalloc Vector4[145];

        for (int i = 0; i < 145; i++)
        {
            GetVertexRowCol(i, out int row, out int col, out bool isInner);
            float lx = isInner ? (col + 0.5f) * SubCellSize : col * SubCellSize;
            float ly = isInner ? (row / 2 + 0.5f) * SubCellSize : (row / 2) * SubCellSize;

            float wx = chunk.WorldPosition.X - ly;
            float wy = chunk.WorldPosition.Y - lx;
            float wz = chunk.Heights[i];
            positions[i] = new Vector3(wx, wy, wz);

            normals[i] = i < chunk.Normals.Length ? Vector3.Normalize(chunk.Normals[i]) : Vector3.UnitZ;

            if (chunk.MccvColors != null && (i * 4 + 3) < chunk.MccvColors.Length)
            {
                int off = i * 4;
                colors[i] = new Vector4(
                    chunk.MccvColors[off + 2] / 255.0f, // R
                    chunk.MccvColors[off + 1] / 255.0f, // G
                    chunk.MccvColors[off + 0] / 255.0f, // B
                    chunk.MccvColors[off + 3] / 255.0f  // A
                );
            }
            else
            {
                colors[i] = new Vector4(0.5f, 0.5f, 0.5f, 1.0f);
            }
        }

        // Iterate through each layer that has ground effects
        for (int layerIdx = 0; layerIdx < chunk.Layers.Count; layerIdx++)
        {
            var layer = chunk.Layers[layerIdx];
            if (layer.EffectId == 0)
                continue;

            var texRecord = textureLookup(layer.EffectId);
            if (texRecord == null || texRecord.DoodadIds.Count == 0)
                continue;

            // Count valid doodads
            int validDoodadCount = 0;
            for (int k = 0; k < texRecord.DoodadIds.Count; k++)
            {
                if (texRecord.DoodadIds[k] != 0)
                    validDoodadCount++;
            }
            if (validDoodadCount <= 0)
                continue;

            // Deterministic random generator per chunk and layer
            int seed = ((chunk.TileX * 16 + chunk.ChunkX) * 397) ^ ((chunk.TileY * 16 + chunk.ChunkY) * 31) ^ (layerIdx * 17);
            var rng = new Random(seed);

            // Determine candidate count from density
            int nominalDensity = texRecord.Density > 0 ? (int)texRecord.Density : 16;
            int candidateCount = Math.Max(1, (int)MathF.Round(nominalDensity * densityMultiplier));

            for (int candidate = 0; candidate < candidateCount; candidate++)
            {
                int cellX = rng.Next(8);
                int cellY = rng.Next(8);

                // Hole check
                if (chunk.IsCellHoled(cellX, cellY))
                    continue;

                // Pick one of 4 triangles in the cell
                int triIndex = rng.Next(4);
                GetCellTriangleIndices(cellX, cellY, triIndex, out int idxA, out int idxB, out int idxC);

                Vector3 vA = positions[idxA];
                Vector3 vB = positions[idxB];
                Vector3 vC = positions[idxC];

                Vector3 nA = normals[idxA];
                Vector3 nB = normals[idxB];
                Vector3 nC = normals[idxC];

                // Uniform random point in triangle
                float r1 = rng.NextSingle();
                float r2 = rng.NextSingle();
                if (r1 + r2 > 1.0f)
                {
                    r1 = 1.0f - r1;
                    r2 = 1.0f - r2;
                }
                float u = 1.0f - r1 - r2;
                float v = r1;
                float w = r2;

                Vector3 pos = u * vA + v * vB + w * vC;
                Vector3 norm = Vector3.Normalize(u * nA + v * nB + w * nC);

                // Slope check: normal Z must be >= 0.4
                if (norm.Z < SlopeNormalZThreshold)
                    continue;

                // Alpha map sampling
                if (layer.AlphaMap != null && layer.AlphaMap.Length == 64 * 64)
                {
                    float localY = chunk.WorldPosition.X - pos.X;
                    float localX = chunk.WorldPosition.Y - pos.Y;
                    int ax = Math.Clamp((int)((localX / ChunkSize) * 64.0f), 0, 63);
                    int ay = Math.Clamp((int)((localY / ChunkSize) * 64.0f), 0, 63);
                    byte alphaByte = layer.AlphaMap[ay * 64 + ax];
                    if (alphaByte == 0 || rng.NextSingle() > (alphaByte / 255.0f))
                        continue;
                }

                // Random doodad selection
                int roll = rng.Next(validDoodadCount);
                uint chosenDoodadId = 0;
                int cur = 0;
                for (int k = 0; k < texRecord.DoodadIds.Count; k++)
                {
                    if (texRecord.DoodadIds[k] == 0) continue;
                    if (cur == roll)
                    {
                        chosenDoodadId = texRecord.DoodadIds[k];
                        break;
                    }
                    cur++;
                }
                if (chosenDoodadId == 0)
                    continue;

                var doodadRecord = doodadLookup(chosenDoodadId);
                GroundEffectDoodadFlags doodadFlags = doodadRecord?.Flags ?? GroundEffectDoodadFlags.None;

                // Random yaw
                float yaw = rng.NextSingle() * MathF.PI * 2.0f;
                var rotYaw = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, yaw);

                // Flag 0x1: Align to terrain normal vs upright
                Quaternion orientation;
                if ((doodadFlags & GroundEffectDoodadFlags.AlignToNormal) != 0)
                {
                    var rotAlign = ComputeNormalAlignment(norm);
                    orientation = rotAlign * rotYaw;
                }
                else
                {
                    orientation = rotYaw;
                }

                // Random scale variation: 0.85 to 1.15
                float scale = 0.85f + rng.NextSingle() * 0.3f;
                if (doodadRecord?.AnimScale > 0)
                    scale *= doodadRecord.AnimScale;

                // Flag 0x2: Ignore MCCV vs interpolate vertex colors
                Vector4 colorVec;
                if ((doodadFlags & GroundEffectDoodadFlags.IgnoreMCCV) != 0)
                {
                    colorVec = new Vector4(1.0f, 1.0f, 1.0f, 1.0f);
                }
                else
                {
                    colorVec = u * colors[idxA] + v * colors[idxB] + w * colors[idxC];
                }

                // MCSH Shadow attenuation (scale RGB by 0.7 if in shadow)
                if (chunk.ShadowMap != null && chunk.ShadowMap.Length == 64 * 64)
                {
                    float localY = chunk.WorldPosition.X - pos.X;
                    float localX = chunk.WorldPosition.Y - pos.Y;
                    int sx = Math.Clamp((int)((localX / ChunkSize) * 64.0f), 0, 63);
                    int sy = Math.Clamp((int)((localY / ChunkSize) * 64.0f), 0, 63);
                    if (chunk.ShadowMap[sy * 64 + sx] > 0)
                    {
                        colorVec.X *= 0.7f;
                        colorVec.Y *= 0.7f;
                        colorVec.Z *= 0.7f;
                    }
                }

                byte bB = (byte)Math.Clamp((int)(colorVec.Z * 255.0f), 0, 255);
                byte bG = (byte)Math.Clamp((int)(colorVec.Y * 255.0f), 0, 255);
                byte bR = (byte)Math.Clamp((int)(colorVec.X * 255.0f), 0, 255);
                byte bA = (byte)Math.Clamp((int)(colorVec.W * 255.0f), 0, 255);
                uint bgra = (uint)(bB | (bG << 8) | (bR << 16) | (bA << 24));

                results.Add(new DetailDoodadInstance(
                    pos,
                    orientation,
                    scale,
                    bgra,
                    chosenDoodadId,
                    doodadRecord?.FileDataId,
                    doodadRecord?.ModelPath,
                    doodadFlags));
            }
        }

        return results;
    }

    private static void GetVertexRowCol(int index, out int row, out int col, out bool isInner)
    {
        int rem = index;
        row = 0;
        col = 0;
        isInner = false;
        for (int r = 0; r < 17; r++)
        {
            int size = (r % 2 == 0) ? 9 : 8;
            if (rem < size)
            {
                row = r;
                col = rem;
                isInner = (r % 2 != 0);
                return;
            }
            rem -= size;
        }
    }

    private static void GetCellTriangleIndices(int cellX, int cellY, int triIndex, out int idxA, out int idxB, out int idxC)
    {
        int tl = cellY * 17 + cellX;
        int tr = cellY * 17 + cellX + 1;
        int bl = (cellY + 1) * 17 + cellX;
        int br = (cellY + 1) * 17 + cellX + 1;
        int center = cellY * 17 + 9 + cellX;

        switch (triIndex)
        {
            case 0: // Top
                idxA = center; idxB = tr; idxC = tl;
                break;
            case 1: // Right
                idxA = center; idxB = br; idxC = tr;
                break;
            case 2: // Bottom
                idxA = center; idxB = bl; idxC = br;
                break;
            case 3: // Left
            default:
                idxA = center; idxB = tl; idxC = bl;
                break;
        }
    }
}
