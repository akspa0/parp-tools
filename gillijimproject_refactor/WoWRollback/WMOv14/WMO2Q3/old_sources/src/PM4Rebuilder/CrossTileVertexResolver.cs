using System;
using System.Collections.Generic;
using System.Linq;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder
{
    /// <summary>
    /// Stub for cross-tile vertex resolution. Currently provides auditing of index usage
    /// across regular and MSCN pools and counts out-of-bounds references.
    /// Future: Implement remapping when adjacent tiles are loaded.
    /// </summary>
    public static class CrossTileVertexResolver
    {
        public static (int Total, int Regular, int Mscn, int Invalid, int MaxIndex) AuditVertexIndices(Pm4Scene scene, IEnumerable<int> indices)
        {
            int regular = 0, mscn = 0, invalid = 0, total = 0, maxIndex = -1;
            int regularMax = scene.Vertices.Count - 1;
            int mscnMax = scene.Vertices.Count + scene.MscnVertices.Count - 1;

            foreach (var idx in indices)
            {
                total++;
                if (idx > maxIndex) maxIndex = idx;

                if (idx >= 0 && idx <= regularMax)
                {
                    regular++;
                }
                else if (idx > regularMax && idx <= mscnMax)
                {
                    mscn++;
                }
                else
                {
                    invalid++;
                }
            }

            return (total, regular, mscn, invalid, maxIndex);
        }
    }
}
