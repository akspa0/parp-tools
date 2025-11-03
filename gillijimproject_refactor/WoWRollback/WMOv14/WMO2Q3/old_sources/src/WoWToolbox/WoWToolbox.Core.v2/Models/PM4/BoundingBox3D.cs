using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace WoWToolbox.Core.v2.Models.PM4
{
    public struct BoundingBox3D
    {
        public Vector3 Min { get; }
        public Vector3 Max { get; }

        public BoundingBox3D(Vector3 min, Vector3 max)
        {
            Min = min;
            Max = max;
        }

        public static BoundingBox3D FromVertices(IEnumerable<Vector3> vertices)
        {
            if (vertices == null || !vertices.Any())
            {
                return new BoundingBox3D(Vector3.Zero, Vector3.Zero);
            }

            var minX = float.MaxValue;
            var minY = float.MaxValue;
            var minZ = float.MaxValue;
            var maxX = float.MinValue;
            var maxY = float.MinValue;
            var maxZ = float.MinValue;

            foreach (var vertex in vertices)
            {
                if (vertex.X < minX) minX = vertex.X;
                if (vertex.Y < minY) minY = vertex.Y;
                if (vertex.Z < minZ) minZ = vertex.Z;
                if (vertex.X > maxX) maxX = vertex.X;
                if (vertex.Y > maxY) maxY = vertex.Y;
                if (vertex.Z > maxZ) maxZ = vertex.Z;
            }

            return new BoundingBox3D(new Vector3(minX, minY, minZ), new Vector3(maxX, maxY, maxZ));
        }
    }
}
