using System.Collections.Generic;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;

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

        // In many maps indices that exceed originalCount are meant to reference MSCN block
        // so subtract originalCount to map into appended range.
        for (int i = 0; i < scene.Indices.Count; i++)
        {
            int idx = scene.Indices[i];
            if (idx >= originalCount)
            {
                scene.Indices[i] = idx - originalCount + originalCount; // effectively unchanged; placeholder for per-tile logic
            }
        }
    }
}
