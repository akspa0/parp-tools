using System;
using System.Numerics;

namespace WCAnalyzer.Core.Models
{
    /// <summary>
    /// Represents a 3D bounding box with minimum and maximum coordinates.
    /// </summary>
    public class BoundingBox
    {
        /// <summary>
        /// Gets or sets the minimum coordinates of the bounding box.
        /// </summary>
        public Vector3 Min { get; set; }

        /// <summary>
        /// Gets or sets the maximum coordinates of the bounding box.
        /// </summary>
        public Vector3 Max { get; set; }

        /// <summary>
        /// Gets the width of the bounding box along the X axis.
        /// </summary>
        public float Width => Max.X - Min.X;

        /// <summary>
        /// Gets the height of the bounding box along the Y axis.
        /// </summary>
        public float Height => Max.Y - Min.Y;

        /// <summary>
        /// Gets the depth of the bounding box along the Z axis.
        /// </summary>
        public float Depth => Max.Z - Min.Z;

        /// <summary>
        /// Gets the center of the bounding box.
        /// </summary>
        public Vector3 Center => new Vector3(
            (Min.X + Max.X) * 0.5f,
            (Min.Y + Max.Y) * 0.5f,
            (Min.Z + Max.Z) * 0.5f);

        /// <summary>
        /// Initializes a new instance of the <see cref="BoundingBox"/> class.
        /// The min and max are set to extremes so the first UpdateBounds call will reset them.
        /// </summary>
        public BoundingBox()
        {
            Min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BoundingBox"/> class.
        /// </summary>
        /// <param name="min">The minimum coordinates.</param>
        /// <param name="max">The maximum coordinates.</param>
        public BoundingBox(Vector3 min, Vector3 max)
        {
            Min = min;
            Max = max;
        }

        /// <summary>
        /// Updates the bounding box to include the specified point.
        /// </summary>
        /// <param name="point">The point to include in the bounding box.</param>
        public void UpdateBounds(Vector3 point)
        {
            Min = Vector3.Min(Min, point);
            Max = Vector3.Max(Max, point);
        }

        /// <summary>
        /// Updates the bounding box to include the specified coordinates.
        /// </summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        /// <param name="z">The z coordinate.</param>
        public void UpdateBounds(float x, float y, float z)
        {
            UpdateBounds(new Vector3(x, y, z));
        }

        /// <summary>
        /// Returns a string that represents the current bounding box.
        /// </summary>
        /// <returns>A string that represents the current bounding box.</returns>
        public override string ToString()
        {
            return $"BoundingBox: Min={Min}, Max={Max}, Size=({Width}, {Height}, {Depth})";
        }

        /// <summary>
        /// Creates a copy of this bounding box.
        /// </summary>
        /// <returns>A new bounding box with the same min and max values.</returns>
        public BoundingBox Clone()
        {
            return new BoundingBox(Min, Max);
        }

        /// <summary>
        /// Checks if this bounding box intersects with another bounding box.
        /// </summary>
        /// <param name="other">The other bounding box.</param>
        /// <returns>True if the bounding boxes intersect, otherwise false.</returns>
        public bool Intersects(BoundingBox other)
        {
            return Min.X <= other.Max.X && Max.X >= other.Min.X &&
                   Min.Y <= other.Max.Y && Max.Y >= other.Min.Y &&
                   Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
        }

        /// <summary>
        /// Merges this bounding box with another bounding box.
        /// </summary>
        /// <param name="other">The other bounding box.</param>
        public void Merge(BoundingBox other)
        {
            Min = Vector3.Min(Min, other.Min);
            Max = Vector3.Max(Max, other.Max);
        }
    }
} 