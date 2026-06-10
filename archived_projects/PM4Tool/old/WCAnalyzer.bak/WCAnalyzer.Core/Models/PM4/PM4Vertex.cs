using System;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Represents a vertex in a PM4 file with 3D coordinates.
    /// </summary>
    public class PM4Vertex
    {
        /// <summary>
        /// Gets or sets the X coordinate of the vertex.
        /// </summary>
        public float X { get; set; }
        
        /// <summary>
        /// Gets or sets the Y coordinate of the vertex.
        /// </summary>
        public float Y { get; set; }
        
        /// <summary>
        /// Gets or sets the Z coordinate of the vertex.
        /// </summary>
        public float Z { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="PM4Vertex"/> class.
        /// </summary>
        public PM4Vertex()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="PM4Vertex"/> class with the specified coordinates.
        /// </summary>
        /// <param name="x">The X coordinate.</param>
        /// <param name="y">The Y coordinate.</param>
        /// <param name="z">The Z coordinate.</param>
        public PM4Vertex(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }
        
        /// <summary>
        /// Calculates the Euclidean distance between this vertex and another vertex.
        /// </summary>
        /// <param name="other">The other vertex.</param>
        /// <returns>The Euclidean distance between the two vertices.</returns>
        public float DistanceTo(PM4Vertex other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));
                
            float dx = X - other.X;
            float dy = Y - other.Y;
            float dz = Z - other.Z;
            
            return (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
        
        /// <summary>
        /// Returns a string representation of this vertex.
        /// </summary>
        /// <returns>A string representation of this vertex.</returns>
        public override string ToString()
        {
            return $"({X:F2}, {Y:F2}, {Z:F2})";
        }
    }
} 