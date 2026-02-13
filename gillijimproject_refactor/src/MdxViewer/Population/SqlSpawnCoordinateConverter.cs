using System.Numerics;
using MdxViewer.Rendering;

namespace MdxViewer.Population;

public static class SqlSpawnCoordinateConverter
{
    public static Vector3 ToRendererPosition(Vector3 wowPosition)
    {
        return wowPosition;
    }
}
