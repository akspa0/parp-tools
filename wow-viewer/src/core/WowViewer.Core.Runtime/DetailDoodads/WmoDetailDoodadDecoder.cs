using System.Numerics;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Runtime.DetailDoodads;

/// <summary>
/// Decodes and places WMO detail doodads from parsed MDDL group streams onto WMO group geometry.
/// </summary>
public static class WmoDetailDoodadDecoder
{
    public delegate GroundEffectDoodadRecord? DoodadLookupDelegate(uint doodadId);

    /// <summary>
    /// Decodes WMO detail doodad placements for a specific WMO group into world-space instances.
    /// </summary>
    public static List<DetailDoodadInstance> DecodeGroupDoodads(
        WmoGroupPlacementInput input,
        DoodadLookupDelegate doodadLookup)
    {
        var instances = new List<DetailDoodadInstance>();
        if (input.Layers.Count == 0 || input.Commands.Count == 0 || input.Vertices.Length == 0)
            return instances;

        var rng = new Random(input.GroupIndex * 1013);

        foreach (var cmd in input.Commands)
        {
            if (cmd.LayerIndex >= input.Layers.Count)
                continue;

            var layer = input.Layers[cmd.LayerIndex];
            if (layer.Doodads.Count == 0)
                continue;

            // Pick doodad from layer.Doodads using weight
            int totalWeight = 0;
            for (int i = 0; i < layer.Doodads.Count; i++)
                totalWeight += layer.Doodads[i].Weight > 0 ? layer.Doodads[i].Weight : 1;

            foreach (int loc in cmd.Locations)
            {
                Vector3 localPos;
                Vector3 localNorm;

                if (loc >= 0 && loc < input.Vertices.Length)
                {
                    localPos = input.Vertices[loc];
                    localNorm = loc < input.Normals.Length ? input.Normals[loc] : Vector3.UnitZ;
                }
                else
                {
                    continue;
                }

                // Transform to world space
                Vector3 worldPos = Vector3.Transform(localPos, input.WorldTransform);
                Vector3 worldNorm = Vector3.Normalize(Vector3.TransformNormal(localNorm, input.WorldTransform));

                // Slope check: normal Z must be >= 0.4
                if (worldNorm.Z < GroundEffectPlacementGenerator.SlopeNormalZThreshold)
                    continue;

                // Pick doodad
                int roll = rng.Next(Math.Max(1, totalWeight));
                uint chosenDoodadId = layer.Doodads[0].DoodadId;
                int accum = 0;
                for (int i = 0; i < layer.Doodads.Count; i++)
                {
                    accum += layer.Doodads[i].Weight > 0 ? layer.Doodads[i].Weight : 1;
                    if (roll < accum)
                    {
                        chosenDoodadId = layer.Doodads[i].DoodadId;
                        break;
                    }
                }

                var doodad = doodadLookup(chosenDoodadId);
                GroundEffectDoodadFlags flags = doodad?.Flags ?? GroundEffectDoodadFlags.None;

                float yaw = rng.NextSingle() * MathF.PI * 2.0f;
                var rotYaw = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, yaw);

                Quaternion orientation;
                if ((flags & GroundEffectDoodadFlags.AlignToNormal) != 0)
                {
                    var rotAlign = GroundEffectPlacementGenerator.ComputeNormalAlignment(worldNorm);
                    orientation = rotAlign * rotYaw;
                }
                else
                {
                    orientation = rotYaw;
                }

                float scale = 0.9f + rng.NextSingle() * 0.2f;
                if (doodad?.AnimScale > 0)
                    scale *= doodad.AnimScale;

                uint colorBgra = 0xFFFFFFFF; // White tint for WMO detail doodads

                instances.Add(new DetailDoodadInstance(
                    worldPos,
                    orientation,
                    scale,
                    colorBgra,
                    chosenDoodadId,
                    doodad?.FileDataId,
                    doodad?.ModelPath,
                    flags));
            }
        }

        return instances;
    }
}
