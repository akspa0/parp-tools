using System;
using System.Collections.Generic;
using System.Numerics;

namespace ParpToolbox.Services.Geometry
{
    public sealed class GeometryTransformService
    {
        public sealed class TransformOptions
        {
            public bool ProjectLocal { get; set; }
            public bool FlipXEnabled { get; set; }
            public bool FlipYEnabled { get; set; }
            public float RotXDeg { get; set; }
            public float RotYDeg { get; set; }
            public float RotZDeg { get; set; }
            public float TranslateX { get; set; }
            public float TranslateY { get; set; }
            public float TranslateZ { get; set; }
        }

        public static void ApplyProjectLocal(List<Vector3> verts, bool projectLocal)
        {
            if (!projectLocal || verts == null || verts.Count == 0) return;
            double sx = 0, sy = 0, sz = 0;
            for (int i = 0; i < verts.Count; i++)
            {
                sx += verts[i].X;
                sy += verts[i].Y;
                sz += verts[i].Z;
            }
            double inv = 1.0 / Math.Max(1, verts.Count);
            var mean = new Vector3((float)(sx * inv), (float)(sy * inv), (float)(sz * inv));
            for (int i = 0; i < verts.Count; i++)
            {
                verts[i] = verts[i] - mean;
            }
        }

        public static void ApplyGlobal(List<Vector3> verts, TransformOptions opts)
        {
            if (verts == null || verts.Count == 0 || opts == null) return;

            bool doFlipX = opts.FlipXEnabled;
            bool doFlipY = opts.FlipYEnabled;
            bool doRotX = MathF.Abs(opts.RotXDeg) > 1e-6f;
            bool doRotY = MathF.Abs(opts.RotYDeg) > 1e-6f;
            bool doRotZ = MathF.Abs(opts.RotZDeg) > 1e-6f;
            bool doTrans = MathF.Abs(opts.TranslateX) > 1e-6f || MathF.Abs(opts.TranslateY) > 1e-6f || MathF.Abs(opts.TranslateZ) > 1e-6f;

            Matrix4x4 rx = doRotX ? Matrix4x4.CreateRotationX(opts.RotXDeg * (MathF.PI / 180f)) : Matrix4x4.Identity;
            Matrix4x4 ry = doRotY ? Matrix4x4.CreateRotationY(opts.RotYDeg * (MathF.PI / 180f)) : Matrix4x4.Identity;
            Matrix4x4 rz = doRotZ ? Matrix4x4.CreateRotationZ(opts.RotZDeg * (MathF.PI / 180f)) : Matrix4x4.Identity;
            Vector3 t = doTrans ? new Vector3(opts.TranslateX, opts.TranslateY, opts.TranslateZ) : default;

            for (int i = 0; i < verts.Count; i++)
            {
                var v = verts[i];
                if (doFlipX) v = new Vector3(-v.X, v.Y, v.Z);
                if (doFlipY) v = new Vector3(v.X, -v.Y, v.Z);
                if (doRotX) v = Vector3.Transform(v, rx);
                if (doRotY) v = Vector3.Transform(v, ry);
                if (doRotZ) v = Vector3.Transform(v, rz);
                if (doTrans) v += t;
                verts[i] = v;
            }
        }
    }
}
