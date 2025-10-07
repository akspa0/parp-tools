using ParpToolbox.Formats.PM4;
using System.Threading.Tasks;

namespace ParpToolbox.Services.PM4.Core
{
    /// <summary>
    /// Defines a unified service for exporting PM4 data using various strategies.
    /// </summary>
    public interface IPm4ExportService
    {
        /// <summary>
        /// Exports a PM4 scene based on the provided options.
        /// </summary>
        /// <param name="scene">The loaded PM4 scene object.</param>
        /// <param name="options">The options defining the export strategy and parameters.</param>
        /// <returns>A task representing the asynchronous export operation.</returns>
        Task ExportAsync(Pm4Scene scene, Pm4ExportOptions options);
    }

    /// <summary>
    /// Defines the options for a PM4 export operation.
    /// </summary>
    public class Pm4ExportOptions
    {
        public required string OutputPath { get; set; }
        public ExportStrategy Strategy { get; set; }
        // Add other options like 'IncludeCollision', 'SplitGroups', etc. as needed.
    }

    public enum ExportStrategy
    {
        /// <summary>
        /// Exports all surfaces as a single, raw geometry dump.
        /// </summary>
        RawSurfaces,
        /// <summary>
        /// Exports objects grouped by their MSUR SurfaceGroupKey.
        /// </summary>
        PerSurfaceGroup
    }
}
