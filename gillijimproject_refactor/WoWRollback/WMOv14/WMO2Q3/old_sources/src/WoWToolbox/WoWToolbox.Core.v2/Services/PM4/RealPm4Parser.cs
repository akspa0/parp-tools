using System.Collections.Generic;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4;
using WoWToolbox.Core.v2.Models.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Concrete implementation that extracts navigation mesh objects from a PM4 file
    /// using decoded chunk relationships (MSLK → MSPI → MSVI → MSVT).
    /// </summary>
    public class RealPm4Parser : IRealPm4Parser
    {
        private readonly ICoordinateService _coordinateService;

        public RealPm4Parser(ICoordinateService coordinateService)
        {
            _coordinateService = coordinateService;
        }

        public IEnumerable<IndividualNavigationObject> ParseIndividualObjects(PM4File pm4)
        {
            var results = new List<IndividualNavigationObject>();

            if (pm4.MSLK == null || pm4.MSPI == null || pm4.MSVI == null || pm4.MSVT == null)
                return results; // insufficient data

            // Fast look-ups
            var mspi = pm4.MSPI.Indices;
            var msvi = pm4.MSVI.Indices;
            var msvt = pm4.MSVT.Vertices;

            int objCounter = 0;
            foreach (var entry in pm4.MSLK.Entries.Where(e => e.MspiIndexCount > 0))
            {
                var navObj = new IndividualNavigationObject
                {
                    ObjectId = $"OBJ_{objCounter:000}",
                    BuildingId = entry.GroupObjectId,
                    LinkIdRaw = entry.MaterialColorId // Unknown_0x0C field as per CSV (LinkIdHex)
                };
                if (Utilities.LinkIdDecoder.TryDecode(navObj.LinkIdRaw, out int tx, out int ty))
                {
                    navObj.TileX = tx;
                    navObj.TileY = ty;
                }
                objCounter++;

                // MSPI section contains uint offsets into MSVI for this object.
                int mspiStart = entry.MspiFirstIndex;
                int mspiCount = entry.MspiIndexCount;
                if (mspiStart < 0 || mspiStart + mspiCount > mspi.Count) continue;

                // Each MSPI value is an index into MSVI list (uint), MSVI list then indexes into MSVT vertices (ushort?)
                for (int mspiIdx = 0; mspiIdx < mspiCount; mspiIdx++)
                {
                    int msviIndex = (int)mspi[mspiStart + mspiIdx];
                    if (msviIndex < 0 || msviIndex >= msvi.Count) continue;

                    int msvtIndex = (int)msvi[msviIndex];
                    if (msvtIndex < 0 || msvtIndex >= msvt.Count) continue;

                    var vertexRaw = msvt[msvtIndex];
                    var world = _coordinateService.FromMsvtVertexSimple(vertexRaw);
                    navObj.Vertices.Add(world);
                    navObj.Indices.Add(navObj.Vertices.Count - 1);
                }

                // Ensure at least one triangle (>=3 vertices)
                if (navObj.Vertices.Count >= 3)
                    results.Add(navObj);
            }

            return results;
        }
    }
}
