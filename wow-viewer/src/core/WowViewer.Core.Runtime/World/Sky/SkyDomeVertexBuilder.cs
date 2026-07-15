namespace WowViewer.Core.Runtime.World.Sky;

/// <summary>Builds a Z-up hemisphere for the active world renderer.</summary>
public static class SkyDomeVertexBuilder
{
    public static SkyDomeMeshData Build(int segments, int rings, float radius)
    {
        if (segments < 3)
            throw new ArgumentOutOfRangeException(nameof(segments));
        if (rings < 1)
            throw new ArgumentOutOfRangeException(nameof(rings));
        if (!float.IsFinite(radius) || radius <= 0f)
            throw new ArgumentOutOfRangeException(nameof(radius));

        int vertexCount = (rings + 1) * (segments + 1);
        float[] vertices = new float[vertexCount * 4];
        int vertexOffset = 0;
        for (int ring = 0; ring <= rings; ring++)
        {
            // World/terrain up is +Z. The old Y-up dome rotated this gradient onto the horizon.
            float phi = (float)ring / rings * MathF.PI * 0.5f;
            float z = MathF.Sin(phi) * radius;
            float ringRadius = MathF.Cos(phi) * radius;
            float heightFactor = (float)ring / rings;

            for (int segment = 0; segment <= segments; segment++)
            {
                float theta = (float)segment / segments * MathF.Tau;
                vertices[vertexOffset++] = MathF.Cos(theta) * ringRadius;
                vertices[vertexOffset++] = MathF.Sin(theta) * ringRadius;
                vertices[vertexOffset++] = z;
                vertices[vertexOffset++] = heightFactor;
            }
        }

        var indices = new List<ushort>(rings * segments * 6);
        for (int ring = 0; ring < rings; ring++)
        {
            for (int segment = 0; segment < segments; segment++)
            {
                int current = (ring * (segments + 1)) + segment;
                int next = current + segments + 1;
                indices.Add(checked((ushort)current));
                indices.Add(checked((ushort)next));
                indices.Add(checked((ushort)(current + 1)));
                indices.Add(checked((ushort)(current + 1)));
                indices.Add(checked((ushort)next));
                indices.Add(checked((ushort)(next + 1)));
            }
        }

        return new SkyDomeMeshData(vertices, indices.ToArray());
    }
}

public sealed record SkyDomeMeshData(float[] Vertices, ushort[] Indices);
