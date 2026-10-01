using System.Numerics;
using WowViewer.Core.IO.Converters;
using WoWViewer.Logging;

namespace WoWViewer.Rendering;

/// <summary>
/// Geometry calculation and normal/lighting utilities for WMO rendering.
/// </summary>
internal static class WmoGeometryHelper
{
    public static List<Vector3> BuildRenderNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        if (group.Normals.Count == group.Vertices.Count && group.Normals.Count > 0)
        {
            var normalized = new List<Vector3>(group.Normals.Count);
            bool hasUsableNormal = false;

            for (int i = 0; i < group.Normals.Count; i++)
            {
                Vector3 n = group.Normals[i];
                if (!float.IsFinite(n.X) || !float.IsFinite(n.Y) || !float.IsFinite(n.Z))
                {
                    normalized.Add(Vector3.UnitY);
                    continue;
                }

                float lengthSq = n.LengthSquared();
                if (lengthSq > 1e-8f)
                {
                    normalized.Add(Vector3.Normalize(n));
                    hasUsableNormal = true;
                }
                else
                {
                    normalized.Add(Vector3.UnitY);
                }
            }

            if (hasUsableNormal)
                return normalized;
        }

        return GenerateNormals(group);
    }

    public static List<Vector3> GenerateNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            int i0 = group.Indices[i], i1 = group.Indices[i + 1], i2 = group.Indices[i + 2];
            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                continue;
            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var n = Vector3.Normalize(Vector3.Cross(e1, e2));
            if (float.IsNaN(n.X)) continue;
            normals[i0] += n;
            normals[i1] += n;
            normals[i2] += n;
        }
        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitY).ToList();
    }

    public static Vector4[] BuildVertexLightColors(WmoV14ToV17Converter.WmoGroupData group)
    {
        int vertexCount = group.Vertices.Count;
        var vertexLightColors = new Vector4[vertexCount];
        if (vertexCount == 0)
            return vertexLightColors;

        if (TryCopyParsedVertexColors(group, vertexLightColors))
            return vertexLightColors;

        if (TrySampleVertexColorsFromLightmaps(group, vertexLightColors))
            return vertexLightColors;

        for (int i = 0; i < vertexCount; i++)
            vertexLightColors[i] = Vector4.One;

        return vertexLightColors;
    }

    private static bool TryCopyParsedVertexColors(WmoV14ToV17Converter.WmoGroupData group, Vector4[] vertexLightColors)
    {
        if (group.VertexColors.Count != vertexLightColors.Length || group.VertexColors.Count == 0)
            return false;

        double averageLuminosity = 0.0;
        foreach (uint packedColor in group.VertexColors)
        {
            byte blue = (byte)(packedColor & 0xFF);
            byte green = (byte)((packedColor >> 8) & 0xFF);
            byte red = (byte)((packedColor >> 16) & 0xFF);
            averageLuminosity += (red + green + blue) / 3.0;
        }

        averageLuminosity /= group.VertexColors.Count;
        if (averageLuminosity < 10.0)
            return false;

        for (int i = 0; i < vertexLightColors.Length; i++)
            vertexLightColors[i] = DecodePackedBgra(group.VertexColors[i]);

        return true;
    }

    private static bool TrySampleVertexColorsFromLightmaps(WmoV14ToV17Converter.WmoGroupData group, Vector4[] vertexLightColors)
    {
        if (group.LightmapData.Length == 0 || group.LightmapUVs.Count == 0 || group.LightmapInfos.Count == 0)
            return false;

        int vertexCount = vertexLightColors.Length;
        var redSums = new float[vertexCount];
        var greenSums = new float[vertexCount];
        var blueSums = new float[vertexCount];
        var sampleCounts = new int[vertexCount];
        int faceCount = group.Indices.Count / 3;
        bool hasSamples = false;

        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            var lightmapInfo = group.LightmapInfos[Math.Min(faceIndex, group.LightmapInfos.Count - 1)];
            if (lightmapInfo.Width == 0 || lightmapInfo.Height == 0)
                continue;

            for (int corner = 0; corner < 3; corner++)
            {
                int uvIndex = faceIndex * 3 + corner;
                int indexOffset = faceIndex * 3 + corner;
                if (uvIndex >= group.LightmapUVs.Count || indexOffset >= group.Indices.Count)
                    continue;

                int vertexIndex = group.Indices[indexOffset];
                if ((uint)vertexIndex >= (uint)vertexCount)
                    continue;

                Vector2 uv = group.LightmapUVs[uvIndex];
                if (!float.IsFinite(uv.X) || !float.IsFinite(uv.Y))
                    continue;

                float u = Math.Clamp(uv.X, 0f, 1f);
                float v = Math.Clamp(uv.Y, 0f, 1f);
                int pixelX = (int)(u * (lightmapInfo.Width - 1));
                int pixelY = (int)(v * (lightmapInfo.Height - 1));

                long pixelOffset = (long)lightmapInfo.DataOffset + (((long)pixelY * lightmapInfo.Width) + pixelX) * 4L;
                if (pixelOffset < 0 || pixelOffset + 4 > group.LightmapData.LongLength)
                    continue;

                int pixelOffsetInt = (int)pixelOffset;
                blueSums[vertexIndex] += group.LightmapData[pixelOffsetInt + 0] / 255f;
                greenSums[vertexIndex] += group.LightmapData[pixelOffsetInt + 1] / 255f;
                redSums[vertexIndex] += group.LightmapData[pixelOffsetInt + 2] / 255f;
                sampleCounts[vertexIndex]++;
                hasSamples = true;
            }
        }

        if (!hasSamples)
            return false;

        double averageLuminosity = 0.0;
        for (int i = 0; i < vertexCount; i++)
        {
            if (sampleCounts[i] > 0)
            {
                float invCount = 1f / sampleCounts[i];
                float red = redSums[i] * invCount;
                float green = greenSums[i] * invCount;
                float blue = blueSums[i] * invCount;
                vertexLightColors[i] = new Vector4(red, green, blue, 1f);
                averageLuminosity += (red + green + blue) / 3.0;
            }
            else
            {
                vertexLightColors[i] = Vector4.One;
                averageLuminosity += 1.0;
            }
        }

        averageLuminosity /= vertexCount;
        if (averageLuminosity < 0.08)
            return false;

        return true;
    }

    public static Vector4 DecodePackedBgra(uint packedColor)
    {
        float blue = (packedColor & 0xFF) / 255f;
        float green = ((packedColor >> 8) & 0xFF) / 255f;
        float red = ((packedColor >> 16) & 0xFF) / 255f;
        float alpha = ((packedColor >> 24) & 0xFF) / 255f;
        return new Vector4(red, green, blue, alpha > 0f ? alpha : 1f);
    }

    public static void TransformAabb(Vector3 min, Vector3 max, in Matrix4x4 transform, out Vector3 outMin, out Vector3 outMax)
    {
        outMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        outMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        Span<float> xs = stackalloc float[] { min.X, max.X };
        Span<float> ys = stackalloc float[] { min.Y, max.Y };
        Span<float> zs = stackalloc float[] { min.Z, max.Z };

        foreach (float x in xs)
        foreach (float y in ys)
        foreach (float z in zs)
        {
            Vector3 p = Vector3.Transform(new Vector3(x, y, z), transform);
            outMin = Vector3.Min(outMin, p);
            outMax = Vector3.Max(outMax, p);
        }
    }

    public static EGxBlend ResolveWmoBlendMode(uint rawBlendMode)
    {
        return rawBlendMode switch
        {
            0 => EGxBlend.Opaque,
            // WMO MOMT blend-mode mapping parity with Alpha-era EGx semantics:
            // 0 = Opaque, 1 = AlphaKey (cutout), 2 = Blend, 3 = Add.
            // Treating mode 1 as full Blend causes shell cutouts (e.g., windows/cloth)
            // to render in transparent pass and can expose interior surfaces through walls.
            1 => EGxBlend.AlphaKey,
            2 => EGxBlend.Blend,
            3 => EGxBlend.Add,
            _ => EGxBlend.Blend,
        };
    }
}
