using System.Collections.Generic;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4;
using WoWToolbox.Core.v2.Services.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class PM4BuildingExtractionService : IBuildingExtractionService
    {
        private readonly ICoordinateService _coordinateService;

        public PM4BuildingExtractionService(ICoordinateService coordinateService)
        {
            _coordinateService = coordinateService;
        }

        public IEnumerable<BuildingFragment> ExtractBuildings(PM4File pm4File)
        {
            var fragments = new List<BuildingFragment>();

            if (pm4File.MSUR == null || pm4File.MSVI == null || pm4File.MSVT == null || pm4File.MDOS == null || pm4File.MDSF == null)
            {
                return fragments; // Not enough data to extract buildings
            }

            var mdsfLookup = pm4File.MDSF.Entries.ToDictionary(e => e.msur_index, e => e.mdos_index);

            for (uint msurIndex = 0; msurIndex < pm4File.MSUR.Entries.Count; msurIndex++)
            {
                var msurEntry = pm4File.MSUR.Entries[(int)msurIndex];

                if (!mdsfLookup.TryGetValue(msurIndex, out uint linkedMdosIndex))
                {
                    continue; // No link to a building
                }

                if (linkedMdosIndex >= pm4File.MDOS.Entries.Count)
                {
                    continue; // Invalid MDOS index
                }

                var linkedMdosEntry = pm4File.MDOS.Entries[(int)linkedMdosIndex];

                if (linkedMdosEntry.destruction_state != 0)
                {
                    continue; // Not the default, intact state
                }

                var buildingId = linkedMdosEntry.m_destructible_building_index;
                var fragment = fragments.FirstOrDefault(f => f.BuildingId == buildingId);
                if (fragment == null)
                {
                    fragment = new BuildingFragment { BuildingId = buildingId };
                    fragments.Add(fragment);
                }

                var firstIndex = (int)msurEntry.MsviFirstIndex;
                var indexCount = msurEntry.IndexCount;

                if (firstIndex < 0 || firstIndex + indexCount > pm4File.MSVI.Indices.Count)
                {
                    continue; // Invalid MSVI range
                }

                var msviIndicesForFace = pm4File.MSVI.Indices.GetRange(firstIndex, indexCount);
                var objFaceIndices = new List<int>();
                var baseVertexIndex = fragment.Vertices.Count;

                foreach (var msviIdx in msviIndicesForFace)
                {
                    if (msviIdx >= pm4File.MSVT.Vertices.Count)
                    {
                        continue; // Invalid MSVT index
                    }
                    var vertex = pm4File.MSVT.Vertices[(int)msviIdx];
                    var worldCoords = _coordinateService.FromMsvtVertexSimple(vertex);
                    fragment.Vertices.Add(worldCoords);
                    objFaceIndices.Add(baseVertexIndex + objFaceIndices.Count);
                }

                if (objFaceIndices.Count >= 3)
                {
                    for (int i = 1; i < objFaceIndices.Count - 1; i++)
                    {
                        fragment.Indices.Add(objFaceIndices[0]);
                        fragment.Indices.Add(objFaceIndices[i]);
                        fragment.Indices.Add(objFaceIndices[i + 1]);
                    }
                }
            }

            return fragments;
        }
    }
}