using parpToolbox.Models.PM4.Export;
using System.Collections.Generic;
using ParpToolbox.Formats.P4.Chunks.Common;
using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4.Core
{
    /// <summary>
    /// Defines a service for explicitly mapping raw PM4 chunk data to structured data models.
    /// </summary>
    public interface IPm4FieldMappingService
    {
        /// <summary>
        /// Builds a collection of hierarchically structured object groups from the complete PM4 scene data.
        /// </summary>
        /// <param name="scene">The loaded PM4 scene, providing access to all chunks (MSLK, MSUR, etc.).</param>
        /// <returns>A collection of assembled object groups, each containing their constituent surfaces.</returns>
        IEnumerable<Pm4ObjectGroup> BuildObjectGroups(Pm4Scene scene);
    }
}
