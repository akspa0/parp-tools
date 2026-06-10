using System.Numerics;
using System;

namespace ParpToolbox.Services.Coordinate
{
    /// <summary>
    /// Provides consistent coordinate transformation services across the parpToolbox.
    /// This ensures all components use the same coordinate system transformations.
    /// </summary>
    public static class CoordinateTransformationService
    {
        /// <summary>
        /// Applies the standard coordinate transformation for PM4 data.
        /// This flips the X and Z axes to match the expected coordinate system.
        /// </summary>
        /// <param name="vertex">The input vertex</param>
        /// <returns>Transformed vertex with flipped X and Z axes</returns>
        public static Vector3 ApplyPm4Transformation(Vector3 vertex)
        {
            return new Vector3(-vertex.X, vertex.Y, -vertex.Z);
        }

        /// <summary>
        /// Applies the standard coordinate transformation for MSCN data.
        /// This converts from WoW coordinates (Y,X,Z) with X-axis flip.
        /// </summary>
        /// <param name="xVal">X value from MSCN data</param>
        /// <param name="yVal">Y value from MSCN data</param>
        /// <param name="zVal">Z value from MSCN data</param>
        /// <returns>Transformed vertex in standard coordinate system</returns>
        public static Vector3 ApplyMscnTransformation(float xVal, float yVal, float zVal)
        {
            // Transform from WoW coordinates (Y,X,Z) with X-axis flip
            float x = -yVal;  // Flip X-axis
            float y = xVal;   // Y becomes X
            float z = zVal;   // Z remains Z
            
            return new Vector3(x, y, z);
        }

        /// <summary>
        /// Applies a 180-degree rotation around the Z-axis.
        /// </summary>
        /// <param name="vertex">The input vertex</param>
        /// <returns>Rotated vertex</returns>
        public static Vector3 Apply180DegreeRotation(Vector3 vertex)
        {
            return new Vector3(-vertex.X, -vertex.Y, vertex.Z);
        }

        /// <summary>
        /// Applies a 180-degree rotation around the Z-axis to a vertex relative to a center point.
        /// </summary>
        /// <param name="vertex">The input vertex</param>
        /// <param name="center">The center point for rotation</param>
        /// <returns>Rotated vertex</returns>
        public static Vector3 Apply180DegreeRotation(Vector3 vertex, Vector3 center)
        {
            var relative = vertex - center;
            var rotated = new Vector3(-relative.X, -relative.Y, relative.Z);
            return rotated + center;
        }
    }
}
