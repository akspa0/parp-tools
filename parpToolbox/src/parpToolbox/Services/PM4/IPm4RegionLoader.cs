using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Interface for loading PM4 files with cross-tile vertex reference resolution.
    /// </summary>
    public interface IPm4RegionLoader
    {
        /// <summary>
        /// Loads a PM4 tile with cross-tile vertex reference resolution.
        /// </summary>
        /// <param name="centerTilePath">Path to the center tile to load</param>
        /// <param name="regionSize">Size of the region to load (e.g., 3 = 3x3 grid, 5 = 5x5 grid)</param>
        /// <returns>PM4 scene with resolved cross-tile vertex references</returns>
        Pm4Scene LoadWithCrossTileReferences(string centerTilePath, int regionSize = 3);

        /// <summary>
        /// Loads multiple PM4 tiles in a region and combines their vertex pools.
        /// </summary>
        /// <param name="tileCoordX">X coordinate of center tile</param>
        /// <param name="tileCoordY">Y coordinate of center tile</param>
        /// <param name="basePath">Base path pattern for PM4 files (e.g., "path/to/tiles/{0:D2}_{1:D2}.pm4")</param>
        /// <param name="regionSize">Size of the region to load</param>
        /// <returns>PM4 scene with combined vertex pools from multiple tiles</returns>
        Pm4Scene LoadRegion(int tileCoordX, int tileCoordY, string basePath, int regionSize = 3);
    }
}
