using System.Collections.Generic;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Remaps indices that point into MSCN vertex blocks by appending those vertices
/// to the global scene vertex list and rewriting indices in-place.
/// </summary>
internal static class MscnRemapper
{
    public static void Apply(Pm4Scene scene, MscnChunk mscn)
    {
        if (mscn == null || mscn.Vertices.Count == 0)
            return;

        int originalCount = scene.Vertices.Count;
        // Append MSCN vertices (apply same Y,X,Z transform convention)
        foreach (var v in mscn.Vertices)
        {
            scene.Vertices.Add(new System.Numerics.Vector3(v.Y, v.X, v.Z));
        }

        // Remap indices that exceed the original vertex count to reference MSCN vertices
        // These out-of-bounds indices are cross-tile references that should map to MSCN block
        int mscnStartIndex = originalCount;
        int remappedCount = 0;
        
        for (int i = 0; i < scene.Indices.Count; i++)
        {
            int idx = scene.Indices[i];
            if (idx >= originalCount)
            {
                // Map out-of-bounds index to MSCN vertex range
                int mscnOffset = (idx - originalCount) % mscn.Vertices.Count;
                scene.Indices[i] = mscnStartIndex + mscnOffset;
                remappedCount++;
            }
        }
        
        ConsoleLogger.WriteLine($"MSCN remapper processed {remappedCount} cross-tile vertex references");
    }
}
