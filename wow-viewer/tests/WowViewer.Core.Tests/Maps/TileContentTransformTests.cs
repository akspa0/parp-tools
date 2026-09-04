using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 219 Phase 1: the TileContentTransform seam and the rotation/mirror/per-tile policy.
/// These tests exercise the seam directly with no phase-system type in scope where possible
/// (SC-010), and prove the exact-grid invariants (SC-002 round-trip, SC-009 mirror involution).
/// </summary>
public sealed class TileContentTransformTests
{
    /// <summary>A chunk whose heights encode position: heights[i] = i. Index maps are then
    /// directly observable through the transformed values.</summary>
    private static TerrainChunkData MakeIndexedChunk() => new()
    {
        Heights = Enumerable.Range(0, 145).Select(i => (float)i).ToArray(),
        Normals = Enumerable.Range(0, 145).Select(i => new Vector3(i, -i, 1f)).ToArray(),
        HoleMask = 0,
    };

    [Fact]
    public void Rotate90_CW_Then_CCW_IsIdentity()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData roundTrip = TileContentTransform.TransformChunk(
            TileContentTransform.TransformChunk(chunk, TileTransformKind.Rotate90CW),
            TileTransformKind.Rotate90CCW);

        Assert.Equal(chunk.Heights, roundTrip.Heights);
        Assert.Equal(chunk.Normals, roundTrip.Normals);
    }

    [Fact]
    public void Rotate90_CCW_Then_CW_IsIdentity()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData roundTrip = TileContentTransform.TransformChunk(
            TileContentTransform.TransformChunk(chunk, TileTransformKind.Rotate90CCW),
            TileTransformKind.Rotate90CW);

        Assert.Equal(chunk.Heights, roundTrip.Heights);
    }

    [Fact]
    public void MirrorH_Twice_IsIdentity()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData roundTrip = TileContentTransform.TransformChunk(
            TileContentTransform.TransformChunk(chunk, TileTransformKind.MirrorH),
            TileTransformKind.MirrorH);

        Assert.Equal(chunk.Heights, roundTrip.Heights);
        Assert.Equal(chunk.Normals, roundTrip.Normals);
    }

    [Fact]
    public void MirrorV_Twice_IsIdentity()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData roundTrip = TileContentTransform.TransformChunk(
            TileContentTransform.TransformChunk(chunk, TileTransformKind.MirrorV),
            TileTransformKind.MirrorV);

        Assert.Equal(chunk.Heights, roundTrip.Heights);
    }

    [Fact]
    public void Rotate180_IsIdentity_WhenAppliedTwice()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData roundTrip = TileContentTransform.TransformChunk(
            TileContentTransform.TransformChunk(chunk, TileTransformKind.Rotate180),
            TileTransformKind.Rotate180);

        Assert.Equal(chunk.Heights, roundTrip.Heights);
    }

    /// <summary>
    /// The corner vertex test: index 0 is the outer vertex at lattice (0, 0). Under CW it must
    /// move to lattice (16, 0) = outer row 0, col 8 = index 8.
    /// </summary>
    [Fact]
    public void Rotate90CW_MovesCornerVertex_ToTheKnownSlot()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData rotated = TileContentTransform.TransformChunk(chunk, TileTransformKind.Rotate90CW);

        // Vertex 0 (lattice 0,0) -> lattice (16, 0) -> outer row 0, col 8 -> index 8.
        Assert.Equal(0f, rotated.Heights[8]);
        // Vertex 8 (lattice 16, 0) -> lattice (16, 16) -> outer row 8, col 8 -> index 144.
        Assert.Equal(8f, rotated.Heights[144]);
    }

    /// <summary>Inner vertices stay inner (odd lattice parity preserved by every transform).</summary>
    [Fact]
    public void Transforms_PreserveOuterInnerParity()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        foreach (TileTransformKind kind in new[]
                 {
                     TileTransformKind.Rotate90CW,
                     TileTransformKind.Rotate90CCW,
                     TileTransformKind.Rotate180,
                     TileTransformKind.MirrorH,
                     TileTransformKind.MirrorV,
                 })
        {
            TerrainChunkData transformed = TileContentTransform.TransformChunk(chunk, kind);

            // Outer vertex 0 must land on an outer index; inner vertex 9 (first inner) likewise.
            int outerLandsAt = IndexOf(transformed.Heights, 0f);
            int innerLandsAt = IndexOf(transformed.Heights, 9f);
            Assert.True(IsOuterIndex(outerLandsAt), $"{kind}: vertex 0 landed at {outerLandsAt}");
            Assert.False(IsOuterIndex(innerLandsAt), $"{kind}: vertex 9 landed at {innerLandsAt}");
        }
    }

    [Fact]
    public void HoleMask_MirrorH_MirrorsColumns()
    {
        // Bit (hx=0, hy=0) -> column mirrored to hx=3: bit index hy*4+hx = 0 -> 0*4+3 = 3.
        TerrainChunkData chunk = new() { HoleMask = 1 << 0 };

        TerrainChunkData mirrored = TileContentTransform.TransformChunk(chunk, TileTransformKind.MirrorH);

        Assert.Equal(1 << 3, mirrored.HoleMask);
    }

    [Fact]
    public void HoleMask_Rotate90CW_Rotates()
    {
        // (hx=0, hy=0) -> (3-hy, hx) = (3, 0) -> bit = 0*4+3 = 3.
        TerrainChunkData chunk = new() { HoleMask = 1 << 0 };

        TerrainChunkData rotated = TileContentTransform.TransformChunk(chunk, TileTransformKind.Rotate90CW);

        Assert.Equal(1 << 3, rotated.HoleMask);
    }

    [Fact]
    public void Grid64_MirrorH_MirrorsColumns()
    {
        var alpha = new byte[64 * 64];
        alpha[10 * 64 + 3] = 0xAB;

        TerrainChunkData chunk = new() { AlphaMaps = new Dictionary<int, byte[]> { [0] = alpha } };

        TerrainChunkData mirrored = TileContentTransform.TransformChunk(chunk, TileTransformKind.MirrorH);

        Assert.Equal(0xAB, mirrored.AlphaMaps[0][(10 * 64) + (64 - 1 - 3)]);
        Assert.Equal(0, mirrored.AlphaMaps[0][(10 * 64) + 3]);
    }

    [Fact]
    public void Grid64_Rotate90CW_IsExact()
    {
        var alpha = new byte[64 * 64];
        alpha[10 * 64 + 3] = 0xCD;

        TerrainChunkData chunk = new() { AlphaMaps = new Dictionary<int, byte[]> { [0] = alpha } };

        TerrainChunkData rotated = TileContentTransform.TransformChunk(chunk, TileTransformKind.Rotate90CW);

        // CW: (x, y) -> (63 - y, x). Source (3, 10) lands at (53, 3).
        Assert.Equal(0xCD, rotated.AlphaMaps[0][(3 * 64) + (64 - 1 - 10)]);
    }

    [Fact]
    public void Normals_TransformWithTheContent()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData rotated = TileContentTransform.TransformChunk(chunk, TileTransformKind.Rotate90CW);

        // Vertex 0's normal (0, 0, 1) -> (-0, 0, 1); vertex 0 moves to slot 8 with the rotated normal.
        Assert.Equal(new Vector3(0f, 0f, 1f), rotated.Normals[8]);
        // Vertex 1 (lattice 2,0) lands at lattice (16,2), index 25. In renderer world axes its
        // normal (1,-1,1) -> (-1,-1,1): X/Y rotate with the placement point, Z stays unchanged.
        Assert.Equal(new Vector3(-1f, -1f, 1f), rotated.Normals[25]);
    }

    // ---- Point transforms (world frame). ----

    [Fact]
    public void TransformPoint_MirrorH_IsAnInvolution()
    {
        Vector2 origin = new(100f, 200f);
        Vector2 point = new(150f, 260f);

        Vector2 once = TileContentTransform.TransformPoint(point, TileTransformKind.MirrorH, origin);
        Vector2 twice = TileContentTransform.TransformPoint(once, TileTransformKind.MirrorH, origin);

        Assert.Equal(point, twice);
    }

    [Fact]
    public void TransformPoint_Rotate90_PairIsIdentity()
    {
        Vector2 origin = new(0f, 0f);
        Vector2 point = new(30f, -70f);

        Vector2 once = TileContentTransform.TransformPoint(point, TileTransformKind.Rotate90CW, origin);
        Vector2 twice = TileContentTransform.TransformPoint(once, TileTransformKind.Rotate90CCW, origin);

        Assert.Equal(point, twice);
    }

    [Fact]
    public void TransformPoint_MirrorH_FlipsTheMatchingWorldAxisAboutOrigin()
    {
        Vector2 origin = new(100f, 100f);
        Vector2 point = new(130f, 140f);

        Vector2 mirrored = TileContentTransform.TransformPoint(point, TileTransformKind.MirrorH, origin);

        // "Horizontal" follows the tile-image/local-X axis. The renderer maps local X to world Y,
        // so the matching placement transform preserves world X and reflects world Y.
        Assert.Equal(new Vector2(130f, 60f), mirrored);
    }

    [Fact]
    public void TransformYaw_MirrorsNegate()
    {
        Assert.Equal(-45f, TileContentTransform.TransformYawDegrees(45f, TileTransformKind.MirrorH));
        Assert.Equal(135f, TileContentTransform.TransformYawDegrees(45f, TileTransformKind.MirrorV));
    }

    [Fact]
    public void TransformYaw_Rotate90CW_Subtracts90()
    {
        Assert.Equal(0f, TileContentTransform.TransformYawDegrees(90f, TileTransformKind.Rotate90CW));
    }

    [Fact]
    public void TransformMddfPlacement_TransformsPositionAndYaw()
    {
        var placement = new MddfPlacement(
            1, "tree.mdx", 2,
            new Vector3(10f, 0f, 7f),
            new Vector3(1f, 2f, 30f),
            1f);

        MddfPlacement transformed = TileContentTransform.TransformPlacement(
            placement,
            TileTransformKind.Rotate90CW,
            Vector2.Zero);

        Assert.Equal(new Vector3(0f, -10f, 7f), transformed.Position);
        Assert.Equal(new Vector3(1f, 2f, -60f), transformed.Rotation);
    }

    [Fact]
    public void TransformModfPlacement_TransformsBounds()
    {
        var placement = new ModfPlacement(
            1, "keep.wmo", 2,
            new Vector3(5f, 5f, 3f),
            new Vector3(0f, 0f, 0f),
            new Vector3(2f, 3f, 1f),
            new Vector3(8f, 9f, 6f),
            0);

        ModfPlacement transformed = TileContentTransform.TransformPlacement(
            placement,
            TileTransformKind.MirrorV,
            Vector2.Zero);

        Assert.Equal(new Vector3(-8f, 3f, 1f), transformed.BoundsMin);
        Assert.Equal(new Vector3(-2f, 9f, 6f), transformed.BoundsMax);
    }

    [Fact]
    public void LiquidTileFlags_RotateWithChunk()
    {
        var flags = new byte[64];
        flags[(2 * 8) + 1] = 0xA5;
        TerrainChunkData chunk = new()
        {
            Liquid = new LiquidChunkData { LiquidType = 3, TileFlags = flags },
        };

        TerrainChunkData transformed = TileContentTransform.TransformChunk(
            chunk,
            TileTransformKind.Rotate90CW);

        Assert.NotNull(transformed.Liquid?.TileFlags);
        // 8x8 CW: (1,2) -> (5,1).
        Assert.Equal(0xA5, transformed.Liquid!.TileFlags![(1 * 8) + 5]);
    }

    [Fact]
    public void RotatePoint_FreeAngle_90DegreesMatchesTheKind()
    {
        Vector2 origin = new(0f, 0f);
        Vector2 point = new(10f, 0f);

        Vector2 free = TileContentTransform.RotatePoint(point, 90f, origin);
        Vector2 kind = TileContentTransform.TransformPoint(point, TileTransformKind.Rotate90CW, origin);

        Assert.Equal(kind.X, free.X, 5);
        Assert.Equal(kind.Y, free.Y, 5);
    }

    [Fact]
    public void TransformChunkSlot_PairsAreInverse()
    {
        (int cx, int cy) = (3, 7);

        var cw = TileContentTransform.TransformChunkSlot(cx, cy, TileTransformKind.Rotate90CW);
        var back = TileContentTransform.TransformChunkSlot(cw.Item1, cw.Item2, TileTransformKind.Rotate90CCW);

        Assert.Equal((cx, cy), back);
        Assert.Equal((8, 3), cw);
    }

    [Fact]
    public void None_IsANoOp()
    {
        TerrainChunkData chunk = MakeIndexedChunk();

        TerrainChunkData result = TileContentTransform.TransformChunk(chunk, TileTransformKind.None);

        Assert.Equal(chunk.Heights, result.Heights);
        Assert.Equal(chunk.Normals, result.Normals);
        Assert.Equal(chunk.HoleMask, result.HoleMask);
    }

    private static int IndexOf(float[] values, float value)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] == value)
                return i;
        }

        return -1;
    }

    /// <summary>Outer index = outerRow * 17 + outerCol, outerRow 0..8, outerCol 0..8.</summary>
    private static bool IsOuterIndex(int index)
    {
        if (index < 0 || index >= 145)
            return false;

        int col = index % 17;
        // Each 17-entry pair contains one 9-vertex outer row then one 8-vertex inner row.
        // The final outer row is indices 136..144; in every case, col < 9 means outer.
        return col < 9;
    }
}
