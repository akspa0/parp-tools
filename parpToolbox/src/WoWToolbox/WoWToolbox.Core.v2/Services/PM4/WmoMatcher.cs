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
                    // Compute center points
                    Vector3 Center(Vector3 min, Vector3 max) => (min + max) * 0.5f;
                    var fragCenter = Center(fragment.BoundingBox.Min, fragment.BoundingBox.Max);
                    var wmoCenter  = Center(wmo.BoundingBox.Min, wmo.BoundingBox.Max);

                    // Euclidean distance between centers (positional error)
                    var positionalError = Vector3.Distance(fragCenter, wmoCenter);

                    // Compare bounding-box volumes (size error)
                    float Volume(BoundingBox3D bb)
                    {
                        var size = bb.Max - bb.Min;
                        return size.X * size.Y * size.Z;
                    }
                    var fragVol = Volume(fragment.BoundingBox);
                    var wmoVol  = Volume(wmo.BoundingBox);
                    var sizeError = MathF.Abs(fragVol - wmoVol) / MathF.Max(fragVol, wmoVol);

                    // Final score with weights (lower is better)
                    var score = positionalError + sizeError * 50f; // weight size difference more
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
                        WMOFilePath = bestMatch.FilePath,
                        Confidence = 1.0f / (1.0f + bestMatchScore) // Simple confidence score
                    });
                }
            }

            return results;
        }

        private List<WmoGeometryData> LoadAllWmoGeometry()
        {
            var wmoDataList = new List<WmoGeometryData>();
            if (!Directory.Exists(_wmoDataPath))
            {
                return wmoDataList; // No WMO directory present
            }

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
