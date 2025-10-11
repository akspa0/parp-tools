using System.Collections.Generic;
using WoWToolbox.Core.v2.Models.PM4;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// A data transfer object for holding the various output paths for the MSLK exporter.
    /// </summary>
    public class MslkExportPaths
    {
        public string? GeometryObjPath { get; set; }
        public string? NodesObjPath { get; set; }
        public string? HierarchyJsonPath { get; set; }
        public string? DoodadCsvPath { get; set; }
        public string? SkippedLogPath { get; set; }
    }

    /// <summary>
    /// Provides functionality to export complex MSLK data from a PM4 file.
    /// </summary>
    public interface IMslkExporter
    {
        /// <summary>
        /// Exports MSLK-related data to various output files.
        /// </summary>
        /// <param name="pm4File">The loaded PM4 file.</param>
        /// <param name="inputFilePath">The path to the original input file, for logging.</param>
        /// <param name="paths">An object containing all the required output paths.</param>
        /// <param name="uniqueBuildingIds">A hash set to collect unique building IDs found during processing.</param>
        void Export(PM4File pm4File, string inputFilePath, MslkExportPaths paths, ISet<int> uniqueBuildingIds);
    }
}
