using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>
/// The axis-aligned tile-content transforms: 90-degree rotation, horizontal/vertical mirror, and
/// free rotation, applied to one MCNK chunk's content and to placement points.
/// </summary>
/// <remarks>
/// <para>
/// This seam is deliberately phase-agnostic (Spec 219 FR-019): it operates on
/// <see cref="TerrainChunkData"/> and raw positions/rotations only, so the phase map system is
/// just its first consumer and later editing tooling can call the same primitives directly.
/// </para>
/// <para>
/// Geometry facts this code is built on (all measured from the existing mesh builder, never
/// guessed):
/// <list type="bullet">
/// <item>One chunk mesh spans 33.33 yds with 145 vertices in the interleaved layout:
/// 17 rows alternating 9 outer (even rows) and 8 inner (odd rows). Outer index =
/// <c>outerRow * 17 + outerCol</c>; inner index = <c>innerRow * 17 + 9 + innerCol</c>.</item>
/// <item>In the 17x17 half-cell lattice, outer vertices sit at even (x, y) = (2c, 2r) and inner
/// vertices at odd (x, y) = (2c+1, 2r+1). Every transform here maps even/even to even/even and
/// odd/odd to odd/odd, so outer stays outer and inner stays inner.</item>
/// <item>The renderer maps local (x, y) to world via <c>wx = WP.X - y_local</c>,
/// <c>wy = WP.Y - x_local</c> — call that map M2(a, b) = (-b, -a); M2 is its own inverse. The
/// normal and point transforms below are the local-frame transforms conjugated through M2, so
/// chunk content and placement points transform consistently. Thus local CW becomes world
/// <c>(x, y) -&gt; (y, -x)</c>.</item>
/// <item><see cref="TerrainChunkData.HoleMask"/> is a 4x4 bit grid, bit = <c>holeY * 4 + holeX</c>,
/// each bit covering a 2x2 cell group in the 8x8 cell grid.</item>
/// <item>Alpha maps and the shadow map are 64x64 byte grids, row-major. Grids transform with their
/// own exact W-wide index formulas — never rescaled through the vertex lattice.</item>
/// </list>
/// </para>
/// <para>
/// The local-frame convention (the frame the mesh builder works in, +x right, +y down its rows):
/// Rotate90CW maps (x, y) -> (L - y, x); MirrorH maps (x, y) -> (L - x, y); MirrorV maps
/// (x, y) -> (x, L - y). All are exact-grid and involutions (or pair to their opposite turn).
/// Whether the "CW" button appears clockwise on screen depends on the renderer's ground-plane
/// orientation and is an operator verification item at Gate 2 — the math is self-consistent
/// either way, so a visual direction flip is a one-line convention change, not a redesign.
/// </para>
/// </remarks>
public static class TileContentTransform
{
    private const int VertexRowCount = 17;
    private const int LatticeMax = 16; // 17x17 half-cell lattice => max coordinate 16
    private const int GridSize = 64;   // alpha / shadow grid edge
    private const int HoleGridEdge = 4;

    /// <summary>Transform one chunk's content. Exact-grid for every kind; outer/inner vertices preserved.</summary>
    public static TerrainChunkData TransformChunk(TerrainChunkData chunk, TileTransformKind kind)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        return kind switch
        {
            TileTransformKind.Rotate90CW => TransformChunkVertices(chunk, kind, NormalCW),
            TileTransformKind.Rotate90CCW => TransformChunkVertices(chunk, kind, NormalCCW),
            TileTransformKind.Rotate180 => TransformChunkVertices(chunk, kind, Normal180),
            TileTransformKind.MirrorH => TransformChunkVertices(chunk, kind, NormalMirrorH),
            TileTransformKind.MirrorV => TransformChunkVertices(chunk, kind, NormalMirrorV),
            _ => chunk,
        };
    }

    /// <summary>
    /// Cartography (Spec 222): applies a composed transform sequence to every chunk of one donor
    /// tile — content transformed per kind in order, and each chunk re-slotted within the 16x16
    /// grid so content lands in the slot the composed transform dictates. Slots are exact-grid;
    /// no resampling. Returns the same list instance when no transforms are supplied.
    /// </summary>
    public static List<TerrainChunkData> TransformTileChunks(
        List<TerrainChunkData> chunks,
        IReadOnlyList<TileTransformKind> kinds)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        if (kinds.Count == 0)
            return chunks;

        var result = new List<TerrainChunkData>(chunks.Count);
        foreach (TerrainChunkData chunk in chunks)
        {
            TerrainChunkData transformed = chunk;
            foreach (TileTransformKind kind in kinds)
                transformed = TransformChunk(transformed, kind);

            int slotX = chunk.ChunkX;
            int slotY = chunk.ChunkY;
            foreach (TileTransformKind kind in kinds)
                (slotX, slotY) = TransformChunkSlot(slotX, slotY, kind);

            result.Add(WithSlots(transformed, slotX, slotY));
        }

        return result;
    }

    /// <summary>Rebuilds a chunk with different 16x16 slot coordinates, preserving everything else.</summary>
    private static TerrainChunkData WithSlots(TerrainChunkData chunk, int chunkX, int chunkY)
        => new()
        {
            McinIndex = chunk.McinIndex,
            TileX = chunk.TileX,
            TileY = chunk.TileY,
            ChunkX = chunkX,
            ChunkY = chunkY,
            Heights = chunk.Heights,
            Normals = chunk.Normals,
            HoleMask = chunk.HoleMask,
            Layers = chunk.Layers,
            AlphaMaps = chunk.AlphaMaps,
            ShadowMap = chunk.ShadowMap,
            MccvColors = chunk.MccvColors,
            Liquid = chunk.Liquid,
            WorldPosition = chunk.WorldPosition,
            AreaId = chunk.AreaId,
            McnkFlags = chunk.McnkFlags,
            AlphaSourceFlags = chunk.AlphaSourceFlags,
        };

    /// <summary>Which chunk slot (within a 16x16 ADT) the content of (cx, cy) lands in after the transform.</summary>
    public static (int ChunkX, int ChunkY) TransformChunkSlot(int chunkX, int chunkY, TileTransformKind kind)
    {
        const int max = 15;
        return kind switch
        {
            TileTransformKind.Rotate90CW => (max - chunkY, chunkX),
            TileTransformKind.Rotate90CCW => (chunkY, max - chunkX),
            TileTransformKind.Rotate180 => (max - chunkX, max - chunkY),
            TileTransformKind.MirrorH => (max - chunkX, chunkY),
            TileTransformKind.MirrorV => (chunkX, max - chunkY),
            _ => (chunkX, chunkY),
        };
    }

    /// <summary>
    /// Transform a placement point about a world-space origin. Exact for every kind. Derived by
    /// conjugating the local-frame transform through the renderer's local-to-world mapping
    /// M2(a, b) = (-b, -a): local CW becomes world (a, b) -> (b, -a); local MirrorH becomes
    /// world (a, b) -> (a, -b); local MirrorV becomes world (a, b) -> (-a, b).
    /// </summary>
    public static Vector2 TransformPoint(Vector2 point, TileTransformKind kind, Vector2 origin)
    {
        var d = point - origin;
        var t = kind switch
        {
            TileTransformKind.Rotate90CW => new Vector2(d.Y, -d.X),
            TileTransformKind.Rotate90CCW => new Vector2(-d.Y, d.X),
            TileTransformKind.Rotate180 => new Vector2(-d.X, -d.Y),
            TileTransformKind.MirrorH => new Vector2(d.X, -d.Y),
            TileTransformKind.MirrorV => new Vector2(-d.X, d.Y),
            _ => d,
        };
        return origin + t;
    }

    /// <summary>
    /// Transform a placement's Z (yaw) rotation, in degrees, for the given kind. Mirrors flip
    /// handedness and negate the yaw; 90-degree turns add their angle. The sign convention of the
    /// placement yaw is verified against real data at Phase 2's gate.
    /// </summary>
    public static float TransformYawDegrees(float yawDegrees, TileTransformKind kind)
        => kind switch
        {
            TileTransformKind.Rotate90CW => yawDegrees - 90f,
            TileTransformKind.Rotate90CCW => yawDegrees + 90f,
            TileTransformKind.Rotate180 => yawDegrees + 180f,
            TileTransformKind.MirrorH => -yawDegrees,
            TileTransformKind.MirrorV => 180f - yawDegrees,
            _ => yawDegrees,
        };

    /// <summary>Transform a complete MDDF placement (position and orientation) as tile content.</summary>
    public static MddfPlacement TransformPlacement(
        MddfPlacement placement,
        TileTransformKind kind,
        Vector2 origin)
    {
        Vector2 position = TransformPoint(new Vector2(placement.Position.X, placement.Position.Y), kind, origin);
        Vector3 rotation = TransformPlacementRotation(placement.Rotation, kind);
        return placement with
        {
            Position = new Vector3(position, placement.Position.Z),
            Rotation = rotation,
        };
    }

    /// <summary>
    /// Transform a complete MODF placement, including its world-space axis-aligned bounds.
    /// </summary>
    public static ModfPlacement TransformPlacement(
        ModfPlacement placement,
        TileTransformKind kind,
        Vector2 origin)
    {
        Vector2 position = TransformPoint(new Vector2(placement.Position.X, placement.Position.Y), kind, origin);
        Vector3 rotation = TransformPlacementRotation(placement.Rotation, kind);
        (Vector3 boundsMin, Vector3 boundsMax) = TransformBounds(
            placement.BoundsMin,
            placement.BoundsMax,
            kind,
            origin);

        return placement with
        {
            Position = new Vector3(position, placement.Position.Z),
            Rotation = rotation,
            BoundsMin = boundsMin,
            BoundsMax = boundsMax,
        };
    }

    /// <summary>
    /// Transform the placement Euler data in the horizontal plane. X/Y tilt is preserved; Z is
    /// the yaw used by the viewer. A mirror reverses handedness, represented by the matching yaw
    /// reflection. Mirroring the model mesh itself, if desired by a future editor, requires a
    /// renderer/editor reflection matrix in addition to this placement-data transform.
    /// </summary>
    public static Vector3 TransformPlacementRotation(Vector3 rotation, TileTransformKind kind)
        => new(rotation.X, rotation.Y, TransformYawDegrees(rotation.Z, kind));

    /// <summary>Rotate a point about an origin by an arbitrary angle (degrees, world XZ plane).</summary>
    public static Vector2 RotatePoint(Vector2 point, float degrees, Vector2 origin)
    {
        var d = point - origin;
        double rad = degrees * Math.PI / 180.0;
        double cos = Math.Cos(rad);
        double sin = Math.Sin(rad);
        return origin + new Vector2(
            (float)((d.X * cos) + (d.Y * sin)),
            (float)((-d.X * sin) + (d.Y * cos)));
    }

    // Exact index transform shared by the 17x17 vertex lattice, 4x4 hole grid, and 64x64 maps.
    // Each caller supplies its own maximum coordinate; no grid is rescaled through another.
    private static (int X, int Y) TransformIndex(int x, int y, int max, TileTransformKind kind)
        => kind switch
        {
            TileTransformKind.Rotate90CW => (max - y, x),
            TileTransformKind.Rotate90CCW => (y, max - x),
            TileTransformKind.Rotate180 => (max - x, max - y),
            TileTransformKind.MirrorH => (max - x, y),
            TileTransformKind.MirrorV => (x, max - y),
            _ => (x, y),
        };

    // World-frame normal transforms: the local transform conjugated through M2(a,b,c) = (-b,-a,c).

    private static Vector3 NormalCW(Vector3 n) => new(n.Y, -n.X, n.Z);
    private static Vector3 NormalCCW(Vector3 n) => new(-n.Y, n.X, n.Z);
    private static Vector3 Normal180(Vector3 n) => new(-n.X, -n.Y, n.Z);
    private static Vector3 NormalMirrorH(Vector3 n) => new(n.X, -n.Y, n.Z);
    private static Vector3 NormalMirrorV(Vector3 n) => new(-n.X, n.Y, n.Z);

    private static TerrainChunkData TransformChunkVertices(
        TerrainChunkData chunk,
        TileTransformKind kind,
        Func<Vector3, Vector3> normalTransform)
    {
        int[] indexMap = BuildVertexIndexMap(kind);

        return new TerrainChunkData
        {
            McinIndex = chunk.McinIndex,
            TileX = chunk.TileX,
            TileY = chunk.TileY,
            ChunkX = chunk.ChunkX,
            ChunkY = chunk.ChunkY,
            WorldPosition = chunk.WorldPosition,

            Heights = TransformHeights(chunk.Heights, indexMap),
            Normals = TransformNormals(chunk.Normals, indexMap, normalTransform),
            HoleMask = TransformHoleMask(chunk.HoleMask, kind),
            Layers = chunk.Layers,
            AlphaMaps = TransformAlphaMaps(chunk.AlphaMaps, kind),
            AlphaSourceFlags = chunk.AlphaSourceFlags,
            ShadowMap = TransformGrid64(chunk.ShadowMap, kind),
            MccvColors = TransformMccv(chunk.MccvColors, indexMap),
            Liquid = TransformLiquid(chunk.Liquid, kind),
            AreaId = chunk.AreaId,
            McnkFlags = chunk.McnkFlags,
        };
    }

    /// <summary>
    /// Build the 145-entry index map: <c>out[indexMap[i]] = in[i]</c> — vertex i's value moves to
    /// the slot its transformed lattice position occupies.
    /// </summary>
    private static int[] BuildVertexIndexMap(TileTransformKind kind)
    {
        var map = new int[145];
        for (int i = 0; i < 145; i++)
        {
            GetLattice(i, out int x, out int y);
            (int tx, int ty) = TransformIndex(x, y, LatticeMax, kind);
            map[i] = LatticeToIndex(tx, ty);
        }

        return map;
    }

    /// <summary>Half-cell lattice coordinates for a vertex index in the interleaved 145 layout.</summary>
    private static void GetLattice(int index, out int x, out int y)
    {
        // 17 rows alternating 9 outer (even) and 8 inner (odd); see TerrainMeshBuilder.GetVertexPosition.
        int remaining = index;
        int row = 0;
        int col = 0;
        bool isInner = false;
        for (int r = 0; r < VertexRowCount; r++)
        {
            int rowSize = (r % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = r;
                col = remaining;
                isInner = r % 2 != 0;
                break;
            }

            remaining -= rowSize;
        }

        if (isInner)
        {
            x = (2 * col) + 1;
            y = row; // odd rows: y_hc = r (already odd)
        }
        else
        {
            x = 2 * col;
            y = row; // even rows: y_hc = r (already even)
        }
    }

    /// <summary>Vertex index for a half-cell lattice coordinate (inverse of <see cref="GetLattice"/>).</summary>
    private static int LatticeToIndex(int x, int y)
    {
        if (y % 2 == 0)
        {
            // Outer: outerRow = y / 2; index = outerRow * 17 + outerCol (outerCol = x / 2).
            return ((y / 2) * VertexRowCount) + (x / 2);
        }

        // Inner: innerRow = (y - 1) / 2; index = innerRow * 17 + 9 + innerCol ((x - 1) / 2).
        return (((y - 1) / 2) * VertexRowCount) + 9 + ((x - 1) / 2);
    }

    private static float[] TransformHeights(float[] heights, int[] indexMap)
    {
        if (heights.Length == 0)
            return heights;

        var result = new float[heights.Length];
        for (int i = 0; i < heights.Length && i < indexMap.Length; i++)
            result[indexMap[i]] = heights[i];

        return result;
    }

    private static Vector3[] TransformNormals(Vector3[] normals, int[] indexMap, Func<Vector3, Vector3> normalTransform)
    {
        if (normals.Length == 0)
            return normals;

        var result = new Vector3[normals.Length];
        for (int i = 0; i < normals.Length && i < indexMap.Length; i++)
            result[indexMap[i]] = normalTransform(normals[i]);

        return result;
    }

    private static byte[]? TransformMccv(byte[]? mccv, int[] indexMap)
    {
        if (mccv == null || mccv.Length == 0)
            return mccv;

        int vertexStride = 4; // BGRA per vertex
        int vertexCount = mccv.Length / vertexStride;
        var result = new byte[mccv.Length];
        for (int i = 0; i < vertexCount && i < indexMap.Length; i++)
        {
            int src = i * vertexStride;
            int dst = indexMap[i] * vertexStride;
            for (int b = 0; b < vertexStride && dst + b < result.Length; b++)
                result[dst + b] = mccv[src + b];
        }

        return result;
    }

    /// <summary>
    /// Transform the 4x4 hole-bit grid with its own exact W-wide index formula. Each bit moves to
    /// one bit: CW (hx,hy) -> (3-hy, hx), MirrorH -> (3-hx, hy), MirrorV -> (hx, 3-hy).
    /// </summary>
    private static int TransformHoleMask(int holeMask, TileTransformKind kind)
    {
        if (holeMask == 0)
            return holeMask;

        int result = 0;
        for (int hy = 0; hy < HoleGridEdge; hy++)
        {
            for (int hx = 0; hx < HoleGridEdge; hx++)
            {
                int bit = 1 << ((hy * HoleGridEdge) + hx);
                if ((holeMask & bit) == 0)
                    continue;

                (int tx, int ty) = TransformIndex(hx, hy, HoleGridEdge - 1, kind);
                result |= 1 << ((ty * HoleGridEdge) + tx);
            }
        }

        return result;
    }

    private static Dictionary<int, byte[]> TransformAlphaMaps(
        Dictionary<int, byte[]> alphaMaps,
        TileTransformKind kind)
    {
        if (alphaMaps.Count == 0)
            return alphaMaps;

        var result = new Dictionary<int, byte[]>(alphaMaps.Count);
        foreach ((int layer, byte[] data) in alphaMaps)
            result[layer] = TransformGrid64(data, kind) ?? data;

        return result;
    }

    private static LiquidChunkData? TransformLiquid(LiquidChunkData? liquid, TileTransformKind kind)
    {
        if (liquid == null)
            return null;

        return new LiquidChunkData
        {
            LiquidType = liquid.LiquidType,
            MinHeight = liquid.MinHeight,
            MaxHeight = liquid.MaxHeight,
            TileFlags = TransformSquareGrid(liquid.TileFlags, 8, kind),
        };
    }

    private static (Vector3 Min, Vector3 Max) TransformBounds(
        Vector3 min,
        Vector3 max,
        TileTransformKind kind,
        Vector2 origin)
    {
        Span<Vector2> corners = stackalloc Vector2[4]
        {
            new(min.X, min.Y),
            new(max.X, min.Y),
            new(min.X, max.Y),
            new(max.X, max.Y),
        };

        Vector2 transformedMin = new(float.PositiveInfinity, float.PositiveInfinity);
        Vector2 transformedMax = new(float.NegativeInfinity, float.NegativeInfinity);
        foreach (Vector2 corner in corners)
        {
            Vector2 transformed = TransformPoint(corner, kind, origin);
            transformedMin = Vector2.Min(transformedMin, transformed);
            transformedMax = Vector2.Max(transformedMax, transformed);
        }

        return (
            new Vector3(transformedMin, min.Z),
            new Vector3(transformedMax, max.Z));
    }

    /// <summary>
    /// Transform a 64x64 row-major byte grid with its own exact index formula. A texel maps to
    /// exactly one texel, so the transform is lossless and requires no resampling.
    /// </summary>
    private static byte[]? TransformGrid64(byte[]? grid, TileTransformKind kind)
        => TransformSquareGrid(grid, GridSize, kind);

    private static byte[]? TransformSquareGrid(byte[]? grid, int edge, TileTransformKind kind)
    {
        if (grid == null || grid.Length != edge * edge)
            return grid;

        var result = new byte[grid.Length];
        for (int y = 0; y < edge; y++)
        {
            for (int x = 0; x < edge; x++)
            {
                (int tx, int ty) = TransformIndex(x, y, edge - 1, kind);
                result[(ty * edge) + tx] = grid[(y * edge) + x];
            }
        }

        return result;
    }
}

/// <summary>The axis-aligned tile-content transforms, as named tools and for composition.</summary>
public enum TileTransformKind
{
    None = 0,
    Rotate90CW,
    Rotate90CCW,
    Rotate180,
    MirrorH,
    MirrorV,
}
