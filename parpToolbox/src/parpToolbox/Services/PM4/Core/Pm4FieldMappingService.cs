using parpToolbox.Models.PM4.Export;
using System.Collections.Generic;
using System.Linq;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4.Core
{
    /// <summary>
    /// Service for explicitly mapping raw PM4 chunk data to structured data models.
    /// </summary>
    public class Pm4FieldMappingService : IPm4FieldMappingService
    {
        private readonly IPm4ChunkAccessService _chunkAccessService;

        public Pm4FieldMappingService(IPm4ChunkAccessService chunkAccessService)
        {
            _chunkAccessService = chunkAccessService;
        }

        /// <summary>
        /// Builds a collection of hierarchically structured object groups from the complete PM4 scene data.
        /// This implementation uses the MSUR.GroupKey for primary grouping.
        /// </summary>
        public IEnumerable<Pm4ObjectGroup> BuildObjectGroups(Pm4Scene scene)
        {
            var msurEntries = _chunkAccessService.GetMsurChunks(scene);

            // Group surfaces by the MSUR GroupKey, which is the most reliable grouping field.
            var groupedSurfaces = msurEntries
                .Select((entry, index) => new { entry, index })
                .GroupBy(item => item.entry.GroupKey)
                .Select(group => new Pm4ObjectGroup
                {
                    GroupId = group.Key,
                    Surfaces = group.Select(item => new Pm4SurfaceData
                    {
                        SurfaceId = (uint)item.index,
                        MsviFirstIndex = (int)item.entry.MsviFirstIndex,
                        IndexCount = item.entry.IndexCount,
                        SurfaceGroupKey = item.entry.SurfaceGroupKey
                    }).ToList()
                });

            return groupedSurfaces;
        }
    }
}
