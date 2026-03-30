using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4PlacementContract
{
    public static Pm4PlanarTransform GetDefaultPlanarTransform(Pm4CoordinateMode coordinateMode)
    {
        return coordinateMode == Pm4CoordinateMode.TileLocal
            ? new Pm4PlanarTransform(false, true, true)
            : new Pm4PlanarTransform(false, false, false);
    }

    public static IReadOnlyList<Pm4PlanarTransform> EnumeratePlanarTransforms(Pm4CoordinateMode coordinateMode)
    {
        return coordinateMode == Pm4CoordinateMode.TileLocal
        ?
        [
            new Pm4PlanarTransform(false, true, true),
            new Pm4PlanarTransform(false, false, false)
        ]
        :
        [
            new Pm4PlanarTransform(false, false, false),
            new Pm4PlanarTransform(false, true, true),
            new Pm4PlanarTransform(true, true, false),
            new Pm4PlanarTransform(true, false, true)
        ];
    }
}