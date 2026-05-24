using System.Numerics;

namespace WowViewer.Core.Renderer.Scene;

public sealed class FrustumCuller
{
    private readonly Vector4[] _planes = new Vector4[6];

    public void ComputePlanes(Matrix4x4 viewProj)
    {
        // Left
        _planes[0] = new Vector4(
            viewProj.M14 + viewProj.M11,
            viewProj.M24 + viewProj.M21,
            viewProj.M34 + viewProj.M31,
            viewProj.M44 + viewProj.M41);
        // Right
        _planes[1] = new Vector4(
            viewProj.M14 - viewProj.M11,
            viewProj.M24 - viewProj.M21,
            viewProj.M34 - viewProj.M31,
            viewProj.M44 - viewProj.M41);
        // Bottom
        _planes[2] = new Vector4(
            viewProj.M14 + viewProj.M12,
            viewProj.M24 + viewProj.M22,
            viewProj.M34 + viewProj.M32,
            viewProj.M44 + viewProj.M42);
        // Top
        _planes[3] = new Vector4(
            viewProj.M14 - viewProj.M12,
            viewProj.M24 - viewProj.M22,
            viewProj.M34 - viewProj.M32,
            viewProj.M44 - viewProj.M42);
        // Near
        _planes[4] = new Vector4(
            viewProj.M13,
            viewProj.M23,
            viewProj.M33,
            viewProj.M43);
        // Far
        _planes[5] = new Vector4(
            viewProj.M14 - viewProj.M13,
            viewProj.M24 - viewProj.M23,
            viewProj.M34 - viewProj.M33,
            viewProj.M44 - viewProj.M43);

        for (int i = 0; i < 6; i++)
        {
            float len = new Vector3(_planes[i].X, _planes[i].Y, _planes[i].Z).Length();
            _planes[i] /= len;
        }
    }

    public bool TestAABB(Vector3 min, Vector3 max)
    {
        for (int i = 0; i < 6; i++)
        {
            float px = _planes[i].X >= 0 ? max.X : min.X;
            float py = _planes[i].Y >= 0 ? max.Y : min.Y;
            float pz = _planes[i].Z >= 0 ? max.Z : min.Z;
            float dist = px * _planes[i].X + py * _planes[i].Y + pz * _planes[i].Z + _planes[i].W;
            if (dist < 0)
                return false;
        }
        return true;
    }
}
