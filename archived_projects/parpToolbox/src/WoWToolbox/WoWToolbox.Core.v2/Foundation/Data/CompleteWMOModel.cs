using System.Numerics;
using System.Runtime.CompilerServices;

namespace WoWToolbox.Core.v2.Foundation.Data
{
    /// <summary>
    /// Optimized complete WMO (World Model Object) extracted from PM4 navigation data.
    /// Contains full geometry, normals, texture coordinates, and metadata for 3D rendering
    /// with memory-efficient data structures and enhanced performance.
    /// </summary>
    public class CompleteWMOModel : IDisposable
    {
        private List<Vector3>? _vertices;
        private List<int>? _triangleIndices;
        private List<Vector3>? _normals;
        private List<Vector2>? _texCoords;
        private Dictionary<string, object>? _metadata;

        #region Properties

        /// <summary>Gets or sets the filename for export</summary>
        public string FileName { get; set; } = "";

        /// <summary>Gets or sets the building category classification</summary>
        public string Category { get; set; } = "";

        /// <summary>Gets the vertex list with lazy initialization</summary>
        public List<Vector3> Vertices => _vertices ??= new List<Vector3>();

        /// <summary>Gets the triangle indices with lazy initialization</summary>
        public List<int> TriangleIndices => _triangleIndices ??= new List<int>();

        /// <summary>Gets the normals list with lazy initialization</summary>
        public List<Vector3> Normals => _normals ??= new List<Vector3>();

        /// <summary>Gets the texture coordinates with lazy initialization</summary>
        public List<Vector2> TexCoords => _texCoords ??= new List<Vector2>();

        /// <summary>Gets or sets the material name for export</summary>
        public string MaterialName { get; set; } = "WMO_Material";

        /// <summary>Gets the metadata dictionary with lazy initialization</summary>
        public Dictionary<string, object> Metadata => _metadata ??= new Dictionary<string, object>();

        #endregion

        #region Performance Properties

        /// <summary>Gets the vertex count efficiently</summary>
        public int VertexCount => _vertices?.Count ?? 0;

        /// <summary>Gets the face count efficiently</summary>
        public int FaceCount => (_triangleIndices?.Count ?? 0) / 3;

        /// <summary>Checks if the model has any geometry</summary>
        public bool HasGeometry => VertexCount > 0 && FaceCount > 0;

        /// <summary>Checks if normals are available</summary>
        public bool HasNormals => (_normals?.Count ?? 0) > 0;

        /// <summary>Checks if texture coordinates are available</summary>
        public bool HasTexCoords => (_texCoords?.Count ?? 0) > 0;

        #endregion

        #region Efficient Bulk Operations

        /// <summary>
        /// Efficiently adds multiple vertices using spans for performance.
        /// </summary>
        /// <param name="vertices">Span of vertices to add</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddVertices(ReadOnlySpan<Vector3> vertices)
        {
            var vertexList = Vertices;
            var initialCapacity = vertexList.Count + vertices.Length;
            
            if (vertexList.Capacity < initialCapacity)
                vertexList.Capacity = initialCapacity;
            
            foreach (var vertex in vertices)
            {
                vertexList.Add(vertex);
            }
        }

        /// <summary>
        /// Efficiently adds multiple triangle indices.
        /// </summary>
        /// <param name="indices">Span of indices to add</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddTriangleIndices(ReadOnlySpan<int> indices)
        {
            var indexList = TriangleIndices;
            var initialCapacity = indexList.Count + indices.Length;
            
            if (indexList.Capacity < initialCapacity)
                indexList.Capacity = initialCapacity;
            
            foreach (var index in indices)
            {
                indexList.Add(index);
            }
        }

        /// <summary>
        /// Efficiently adds multiple normals.
        /// </summary>
        /// <param name="normals">Span of normals to add</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddNormals(ReadOnlySpan<Vector3> normals)
        {
            var normalList = Normals;
            var initialCapacity = normalList.Count + normals.Length;
            
            if (normalList.Capacity < initialCapacity)
                normalList.Capacity = initialCapacity;
            
            foreach (var normal in normals)
            {
                normalList.Add(normal);
            }
        }

        /// <summary>
        /// Pre-allocates capacity for known data sizes to improve performance.
        /// </summary>
        /// <param name="vertexCount">Expected vertex count</param>
        /// <param name="faceCount">Expected face count</param>
        public void PreAllocate(int vertexCount, int faceCount = 0)
        {
            if (vertexCount > 0)
            {
                _vertices ??= new List<Vector3>(vertexCount);
                if (_vertices.Capacity < vertexCount)
                    _vertices.Capacity = vertexCount;
            }

            if (faceCount > 0)
            {
                var triangleIndexCount = faceCount * 3;
                _triangleIndices ??= new List<int>(triangleIndexCount);
                if (_triangleIndices.Capacity < triangleIndexCount)
                    _triangleIndices.Capacity = triangleIndexCount;
            }
        }

        #endregion

        #region Spatial Operations

        /// <summary>
        /// Calculates the bounding box of the model efficiently.
        /// </summary>
        /// <returns>Bounding box or null if no vertices</returns>
        public BoundingBox3D? CalculateBoundingBox()
        {
            if (VertexCount == 0) return null;

            var vertices = _vertices!;
            var min = vertices[0];
            var max = vertices[0];

            for (int i = 1; i < vertices.Count; i++)
            {
                var vertex = vertices[i];
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            return new BoundingBox3D(min, max);
        }

        /// <summary>
        /// Gets the center point of the model.
        /// </summary>
        /// <returns>Center point or zero if no vertices</returns>
        public Vector3 GetCenter()
        {
            var bounds = CalculateBoundingBox();
            return bounds?.Center ?? Vector3.Zero;
        }

        /// <summary>
        /// Transforms all vertices by the given matrix efficiently.
        /// </summary>
        /// <param name="transform">Transformation matrix</param>
        public void Transform(Matrix4x4 transform)
        {
            if (_vertices == null) return;

            for (int i = 0; i < _vertices.Count; i++)
            {
                _vertices[i] = Vector3.Transform(_vertices[i], transform);
            }

            // Transform normals too (only rotation part)
            if (_normals != null)
            {
                // Extract just the rotation/scale part for normals (ignore translation)
                for (int i = 0; i < _normals.Count; i++)
                {
                    _normals[i] = Vector3.TransformNormal(_normals[i], transform);
                    _normals[i] = Vector3.Normalize(_normals[i]);
                }
            }
        }

        #endregion

        #region Validation

        /// <summary>
        /// Validates the model geometry for common issues.
        /// </summary>
        /// <returns>Validation result</returns>
        public ModelValidationResult Validate()
        {
            var result = new ModelValidationResult();

            if (VertexCount == 0)
            {
                result.AddError("Model has no vertices");
                return result;
            }

            if (FaceCount == 0)
            {
                result.AddError("Model has no faces");
                return result;
            }

            // Check for degenerate triangles
            var triangles = _triangleIndices!;
            for (int i = 0; i < triangles.Count; i += 3)
            {
                if (i + 2 >= triangles.Count) break;

                var idx1 = triangles[i];
                var idx2 = triangles[i + 1];
                var idx3 = triangles[i + 2];

                if (idx1 >= VertexCount || idx2 >= VertexCount || idx3 >= VertexCount)
                {
                    result.AddError($"Triangle {i / 3}: Index out of bounds");
                }

                if (idx1 == idx2 || idx2 == idx3 || idx1 == idx3)
                {
                    result.AddWarning($"Triangle {i / 3}: Degenerate triangle (duplicate indices)");
                }
            }

            return result;
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Disposes resources and clears collections to free memory.
        /// </summary>
        public void Dispose()
        {
            _vertices?.Clear();
            _triangleIndices?.Clear();
            _normals?.Clear();
            _texCoords?.Clear();
            _metadata?.Clear();

            _vertices = null;
            _triangleIndices = null;
            _normals = null;
            _texCoords = null;
            _metadata = null;

            GC.SuppressFinalize(this);
        }

        #endregion
    }

    /// <summary>
    /// Represents a 3D bounding box for spatial calculations.
    /// </summary>
    public readonly struct BoundingBox3D
    {
        public readonly Vector3 Min;
        public readonly Vector3 Max;

        public BoundingBox3D(Vector3 min, Vector3 max)
        {
            Min = min;
            Max = max;
        }

        public Vector3 Center => (Min + Max) * 0.5f;
        public Vector3 Size => Max - Min;
        public float Volume => Size.X * Size.Y * Size.Z;

        public bool Contains(Vector3 point)
        {
            return point.X >= Min.X && point.X <= Max.X &&
                   point.Y >= Min.Y && point.Y <= Max.Y &&
                   point.Z >= Min.Z && point.Z <= Max.Z;
        }

        public bool Intersects(BoundingBox3D other)
        {
            return Min.X <= other.Max.X && Max.X >= other.Min.X &&
                   Min.Y <= other.Max.Y && Max.Y >= other.Min.Y &&
                   Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
        }
    }

    /// <summary>
    /// Result of model validation with errors and warnings.
    /// </summary>
    public class ModelValidationResult
    {
        private readonly List<string> _errors = new();
        private readonly List<string> _warnings = new();

        public IReadOnlyList<string> Errors => _errors;
        public IReadOnlyList<string> Warnings => _warnings;
        public bool IsValid => _errors.Count == 0;
        public bool HasWarnings => _warnings.Count > 0;

        public void AddError(string error) => _errors.Add(error);
        public void AddWarning(string warning) => _warnings.Add(warning);
    }
} 