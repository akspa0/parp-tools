using System.Collections.Generic;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Provides functionality to export MPRL data from a PM4 file.
    /// </summary>
    public interface IMprlExporter
    {
        /// <summary>
        /// Exports the MPRL point data from a PM4 file to an OBJ file.
        /// </summary>
        /// <param name="pm4File">The PM4 file to process.</param>
        /// <param name="inputFilePath">The original path of the input PM4 file, used for logging in the output.</param>
        /// <param name="outputPath">The path to the output OBJ file.</param>
        void Export(PM4File pm4File, string inputFilePath, string outputPath);
    }
}
