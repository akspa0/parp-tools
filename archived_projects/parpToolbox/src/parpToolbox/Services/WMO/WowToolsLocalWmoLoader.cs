using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.WMO;
using WoWFormatLib.FileReaders;
using WoWFormatLib.Structs.WMO;

namespace ParpToolbox.Services.WMO
{
    /// <summary>
    /// Concrete implementation of <see cref="IWmoLoader"/> that delegates loading to
    /// the wow.tools.local <see cref="WMOReader"/> and converts the result into
    /// immutable <see cref="WmoGroup"/> domain objects.
    /// </summary>
    internal sealed class WowToolsLocalWmoLoader : IWmoLoader
    {
        public (IReadOnlyList<string> textures, IReadOnlyList<WmoGroup> groups) Load(string path, bool includeFacades = false)
        {
            return LoadWithFilter(path, includeFacades, g => true);
        }

        public (IReadOnlyList<string> textures, IReadOnlyList<WmoGroup> groups) LoadCollisionOnly(string path)
        {
            return LoadWithFilter(path, false, g => IsCollisionGroup(g));
        }

        private (IReadOnlyList<string> textures, IReadOnlyList<WmoGroup> groups) LoadWithFilter(string path, bool includeFacades, Func<MOGP, bool> groupFilter)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));

            if (!File.Exists(path))
                throw new FileNotFoundException("WMO root file not found", path);

            var reader = new WMOReader();
            var wmo = reader.LoadWMO(path);

            var textures = wmo.textures.Select(t => t.filename ?? string.Empty).ToList();

            var groups = new List<WmoGroup>(wmo.group?.Length ?? 0);
            if (wmo.group != null)
            {
                for (var gi = 0; gi < wmo.group.Length; gi++)
                {
                    var grp = wmo.group[gi];
                    var mogp = grp.mogp;

                    // Apply group filter
                    if (!groupFilter(mogp))
                        continue;

                    if (mogp.vertices == null || mogp.indices == null)
                        continue; // skip empty groups

                    var verts = mogp.vertices.Select(v => new Vector3(v.vector.X, v.vector.Y, v.vector.Z)).ToList();

                    var faces = new List<(ushort, ushort, ushort)>();
                    // Pre-compute face ranges marked as no-draw (facades)
                    var noDrawFaces = new HashSet<int>();
                    if (mogp.renderBatches != null)
                    {
                        foreach (var batch in mogp.renderBatches)
                        {
                            if (((batch.flags & 0x04) != 0) && !includeFacades)
                            {
                                int start = (int)batch.firstFace;
                                int end = start + batch.numFaces;
                                for (int f = start; f < end; f++)
                                    noDrawFaces.Add(f);
                            }
                        }
                    }

                    int faceIndex = 0;
                    for (int i = 0; i + 2 < mogp.indices.Length; i += 3, faceIndex++)
                    {
                        if (noDrawFaces.Contains(faceIndex))
                            continue;

                        faces.Add((mogp.indices[i], mogp.indices[i + 1], mogp.indices[i + 2]));
                    }

                    // Gather material IDs for each face if present via renderBatches
                    var faceMaterialIds = new List<byte>();
                    if (mogp.renderBatches != null && mogp.renderBatches.Length > 0)
                    {
                        foreach (var batch in mogp.renderBatches)
                        {
                            if (((batch.flags & 0x04) != 0) && !includeFacades)
                                continue; // skip no-draw batch

                            for (var fi = 0; fi < batch.numFaces; fi++)
                                faceMaterialIds.Add(batch.materialID);
                        }
                    }

                    string groupName;
                    // Try exact index first
                    if (wmo.groupNames != null && gi < wmo.groupNames.Length)
                    {
                        groupName = wmo.groupNames[gi].name;
                    }
                    else
                    {
                        groupName = $"group_{gi}";
                    }

                    // Attempt to refine name via nameOffset lookup (most accurate)
                    if (wmo.groupNames != null)
                    {
                        var offsetMatch = wmo.groupNames.FirstOrDefault(n => n.offset == mogp.nameOffset);
                        if (!string.IsNullOrEmpty(offsetMatch.name))
                            groupName = offsetMatch.name;
                    }

                    groups.Add(new WmoGroup(groupName, verts, faces, (uint)mogp.flags, faceMaterialIds));
                }
            }

            return (textures, groups);
        }

        private bool IsCollisionGroup(MOGP mogp)
        {
            // Check if group has unreachable or lod flags which indicate collision geometry
            return (mogp.flags & (MOGPFlags.mogp_unreachable | MOGPFlags.mogp_lod)) != 0;
        }
    }
}
