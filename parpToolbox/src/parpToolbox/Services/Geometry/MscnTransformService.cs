using System;
using System.Collections.Generic;
using System.Numerics;

namespace ParpToolbox.Services.Geometry
{
    public static class MscnTransformService
    {
        public enum MscnBasis
        {
            Legacy,
            Remap
        }

        [Flags]
        public enum FlipAxes
        {
            None = 0,
            X = 1,
            Y = 2,
            Z = 4,
            XY = X | Y,
            XZ = X | Z,
            YZ = Y | Z
        }

        public static void CanonicalizeAxes(List<Vector3> verts, MscnBasis basis)
        {
            if (verts == null || verts.Count == 0) return;
            for (int i = 0; i < verts.Count; i++)
            {
                var v = verts[i];
                switch (basis)
                {
                    case MscnBasis.Remap:
                        verts[i] = new Vector3(v.Y, v.X, v.Z);
                        break;
                    case MscnBasis.Legacy:
                    default:
                        verts[i] = new Vector3(-v.X, -v.Y, v.Z);
                        break;
                }
            }
        }

        public static void PreTransform(List<Vector3> verts, int rotZDeg, FlipAxes flip, MscnBasis basis)
        {
            if (verts == null || verts.Count == 0) return;

            // First, bring MSCN points into the canonical basis used by meshes
            CanonicalizeAxes(verts, basis);

            // Normalize rotation to right angles (multiples of 90 deg)
            int r = rotZDeg % 360; if (r < 0) r += 360;
            int rNorm = ((int)MathF.Round(r / 90f)) * 90; rNorm %= 360;

            // Build Z-rotation
            if (rNorm != 0)
            {
                float rad = rNorm * (MathF.PI / 180f);
                var rz = Matrix4x4.CreateRotationZ(rad);
                for (int i = 0; i < verts.Count; i++)
                {
                    verts[i] = Vector3.Transform(verts[i], rz);
                }
            }

            // Apply flips
            bool fx = (flip & FlipAxes.X) != 0;
            bool fy = (flip & FlipAxes.Y) != 0;
            bool fz = (flip & FlipAxes.Z) != 0;
            if (fx || fy || fz)
            {
                for (int i = 0; i < verts.Count; i++)
                {
                    var v = verts[i];
                    if (fx) v.X = -v.X;
                    if (fy) v.Y = -v.Y;
                    if (fz) v.Z = -v.Z;
                    verts[i] = v;
                }
            }
        }
    }
}
