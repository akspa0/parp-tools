using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder
{
    /// <summary>
    /// Variant transform: swap Y/Z, then flip X, Y and Z. Useful for testing when geometry appears mirrored along X and Y but Z needs inversion as well.
    /// </summary>
    internal static class TransformUtilsFlipXYZ
    {
        public static void Apply(Pm4Scene scene)
        {
            const float scale = 1f / 4096f;

            for (int i = 0; i < scene.Vertices.Count; i++)
            {
                Vector3 v = scene.Vertices[i] * scale;
                (v.Y, v.Z) = (v.Z, v.Y); // swap Y/Z
                v.X = -v.X;
                v.Y = -v.Y;
                v.Z = -v.Z;
                scene.Vertices[i] = v;
            }

            for (int i = 0; i < scene.MscnVertices.Count; i++)
            {
                Vector3 v = scene.MscnVertices[i];
                (v.Y, v.Z) = (v.Z, v.Y);
                v.X = -v.X;
                v.Y = -v.Y;
                v.Z = -v.Z;
                scene.MscnVertices[i] = v;
            }
        }
    }
}
