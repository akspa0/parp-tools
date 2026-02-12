using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder
{
    /// <summary>
    /// Utility class that performs coordinate-system corrections after a PM4 scene is loaded.
    /// </summary>
    internal static class TransformUtils
    {
        // Blizzard PM4 chunks encode MSVT vertices in full-resolution world units already.
        // Coordinate unification for OBJ export is therefore strictly:
        // 1) Swap Y ↔ Z so Y becomes up (OBJ expects Y-up)
        // 2) Flip X and Y to correct handedness/rotation (empirically confirmed)
        public static void ApplyCoordinateUnification(Pm4Scene scene)
        {


            // Transform primary vertex buffer
            for (int i = 0; i < scene.Vertices.Count; i++)
            {
                Vector3 v = scene.Vertices[i];
                // Do NOT scale MSVT/MSUR vertices – they are already in full-resolution units
                (v.Y, v.Z) = (v.Z, v.Y); // swap Y/Z (Y up)
                v.X = -v.X;          // flip X (east-west)
                v.Y = -v.Y;          // flip Y (north-south)
                scene.Vertices[i] = v;
            }

            // Transform MSCN vertices (no scale – raw units already correct)
            // Based on coordinate analysis: MSCN needs X unmirror + coordinate swap for alignment
            for (int i = 0; i < scene.MscnVertices.Count; i++)
            {
                Vector3 v = scene.MscnVertices[i];
                // do NOT apply scale (per spec)
                
                // Apply corrected MSCN transformation for perfect MSVT alignment:
                // 1. Swap Y/Z (same as MSVT)
                (v.Y, v.Z) = (v.Z, v.Y);
                // 2. Swap X/Z to correct the rotation
                (v.X, v.Z) = (v.Z, v.X);
                // 3. Apply flips to align with MSVT coordinate system
                v.X = -v.X;  // flip X 
                v.Y = -v.Y;  // flip Y (same as MSVT)
                
                scene.MscnVertices[i] = v;
            }
        }
    }
}
