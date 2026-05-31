using System.Numerics;
using WoWViewer.Rendering;

namespace WoWViewer.Population;

public static class SqlSpawnCoordinateConverter
{
    public static Vector3 ToRendererPosition(Vector3 wowPosition)
    {
        return wowPosition;
    }
}
