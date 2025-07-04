using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class WmoMatcher : IWmoMatcher
    {
        private readonly string _wmoDataPath;

        public WmoMatcher(string wmoDataPath)
        {
            _wmoDataPath = wmoDataPath;
        }

        public IEnumerable<WmoMatchResult> Match(IEnumerable<BuildingFragment> fragments)
        {
            var results = new List<WmoMatchResult>();
            var wmoGeometries = LoadAllWmoGeometry();

            if (!wmoGeometries.Any())
            {
                return results; // No WMOs to match against
            }

            foreach (var fragment in fragments)
            {
                WmoGeometryData? bestMatch = null;
                var bestMatchScore = float.MaxValue;

                foreach (var wmo in wmoGeometries)
                {
                    var score = Vector3.Distance(fragment.BoundingBox.Min, wmo.BoundingBox.Min) + Vector3.Distance(fragment.BoundingBox.Max, wmo.BoundingBox.Max);
                    if (score < bestMatchScore)
                    {
                        bestMatchScore = score;
                        bestMatch = wmo;
                    }
                }

                if (bestMatch != null)
                {
                    results.Add(new WmoMatchResult
                    {
                        BuildingId = fragment.BuildingId,
                        WmoFilePath = bestMatch.FilePath,
                        Confidence = 1.0f / (1.0f + bestMatchScore) // Simple confidence score
                    });
                }
            }

            return results;
        }

        private List<WmoGeometryData> LoadAllWmoGeometry()
        {
            var wmoDataList = new List<WmoGeometryData>();
            var wmoRootFiles = Directory.GetFiles(_wmoDataPath, "*.wmo", SearchOption.AllDirectories);

            foreach (var wmoFilePath in wmoRootFiles)
            {
                wmoDataList.Add(LoadWmoGeometry(wmoFilePath));
            }

            return wmoDataList;
        }

        private WmoGeometryData LoadWmoGeometry(string wmoFilePath)
        {
            var allVertices = new List<System.Numerics.Vector3>();
            var (groupCount, groupNames) = WoWToolbox.Core.WMO.WmoRootLoader.LoadGroupInfo(wmoFilePath);

            foreach (var groupName in groupNames)
            {
                var groupPath = Path.Combine(Path.GetDirectoryName(wmoFilePath)!, groupName);
                if (File.Exists(groupPath))
                {
                    using var stream = File.OpenRead(groupPath);
                    var groupMesh = WoWToolbox.Core.WMO.WmoGroupMesh.LoadFromStream(stream, groupPath);
                    if (groupMesh != null)
                    {
                        allVertices.AddRange(groupMesh.Vertices.Select(v => v.Position));
                    }
                }
            }

            return new WmoGeometryData
            {
                FilePath = wmoFilePath,
                Vertices = allVertices
            };
        }
    }
}
