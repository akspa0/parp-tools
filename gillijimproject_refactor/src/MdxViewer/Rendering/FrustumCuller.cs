using System.Numerics;

namespace MdxViewer.Rendering;

/// <summary>
/// View-frustum culling using 6 planes extracted from the view-projection matrix.
/// Based on Ghidra analysis: CWorldScene::UpdateFrustum @ 0x0066a460.
/// </summary>
public class FrustumCuller
{
    private readonly Plane[] _planes = new Plane[6];

    /// <summary>
    /// Extract and normalize the 6 frustum planes from a combined view-projection matrix.
    /// Plane order: Left, Right, Top, Bottom, Near, Far.
    /// </summary>
    public void Update(Matrix4x4 vp)
    {
        // Left: row4 + row1
        _planes[0] = NormalizePlane(new Plane(
            vp.M14 + vp.M11, vp.M24 + vp.M21, vp.M34 + vp.M31, vp.M44 + vp.M41));

        // Right: row4 - row1
        _planes[1] = NormalizePlane(new Plane(
            vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31, vp.M44 - vp.M41));

        // Top: row4 - row2
        _planes[2] = NormalizePlane(new Plane(
            vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32, vp.M44 - vp.M42));

        // Bottom: row4 + row2
        _planes[3] = NormalizePlane(new Plane(
            vp.M14 + vp.M12, vp.M24 + vp.M22, vp.M34 + vp.M32, vp.M44 + vp.M42));

        // Near: row4 + row3
        _planes[4] = NormalizePlane(new Plane(
            vp.M14 + vp.M13, vp.M24 + vp.M23, vp.M34 + vp.M33, vp.M44 + vp.M43));

        // Far: row4 - row3
        _planes[5] = NormalizePlane(new Plane(
            vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33, vp.M44 - vp.M43));
    }

    /// <summary>
    /// Test whether a point is inside the frustum.
    /// </summary>
    public bool TestPoint(Vector3 point)
    {
        for (int i = 0; i < 6; i++)
        {
            if (SignedDistance(_planes[i], point) < 0f)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Test whether a sphere intersects or is inside the frustum.
    /// </summary>
    public bool TestSphere(Vector3 center, float radius)
    {
        for (int i = 0; i < 6; i++)
        {
            if (SignedDistance(_planes[i], center) < -radius)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Test whether an axis-aligned bounding box intersects or is inside the frustum.
    /// Uses the "test all 8 corners against each plane" approach from the original client.
    /// </summary>
    public bool TestAABB(Vector3 min, Vector3 max)
    {
        Span<Vector3> corners = stackalloc Vector3[8];
        corners[0] = new Vector3(min.X, min.Y, min.Z);
        corners[1] = new Vector3(max.X, min.Y, min.Z);
        corners[2] = new Vector3(min.X, max.Y, min.Z);
        corners[3] = new Vector3(max.X, max.Y, min.Z);
        corners[4] = new Vector3(min.X, min.Y, max.Z);
        corners[5] = new Vector3(max.X, min.Y, max.Z);
        corners[6] = new Vector3(min.X, max.Y, max.Z);
        corners[7] = new Vector3(max.X, max.Y, max.Z);

        for (int i = 0; i < 6; i++)
        {
            int inside = 0;
            for (int j = 0; j < 8; j++)
            {
                if (SignedDistance(_planes[i], corners[j]) >= 0f)
                    inside++;
            }
            if (inside == 0)
                return false;
        }
        return true;
    }

    private static float SignedDistance(Plane p, Vector3 point)
    {
        return p.Normal.X * point.X + p.Normal.Y * point.Y + p.Normal.Z * point.Z + p.D;
    }

    private static Plane NormalizePlane(Plane p)
    {
        float len = p.Normal.Length();
        if (len < 1e-8f) return p;
        return new Plane(p.Normal / len, p.D / len);
    }
}
