using System.Numerics;

namespace MdxViewer.Terrain;

internal static class MccvColorDecoder
{
    private const float MccvToModulationScale = 2.0f / 255.0f;

    public static Vector4 DecodeModulation(byte[]? mccvColors, int vertexIndex)
    {
        if (mccvColors == null)
            return Vector4.One;

        int offset = vertexIndex * 4;
        if (offset + 3 >= mccvColors.Length)
            return Vector4.One;

        float blue = mccvColors[offset + 0] * MccvToModulationScale;
        float green = mccvColors[offset + 1] * MccvToModulationScale;
        float red = mccvColors[offset + 2] * MccvToModulationScale;

        return new Vector4(
            Math.Clamp(red, 0.0f, 2.0f),
            Math.Clamp(green, 0.0f, 2.0f),
            Math.Clamp(blue, 0.0f, 2.0f),
            1.0f);
    }
}