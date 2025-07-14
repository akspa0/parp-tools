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
        public (IReadOnlyList<string> textures, IReadOnlyList<WmoGroup> groups) Load(string path)
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

                    if (mogp.vertices == null || mogp.indices == null)
                        continue; // skip empty groups

                    var verts = mogp.vertices.Select(v => new Vector3(v.vector.X, v.vector.Y, v.vector.Z)).ToList();

                    var faces = new List<(ushort, ushort, ushort)>();
                    for (int i = 0; i + 2 < mogp.indices.Length; i += 3)
                    {
                        faces.Add((mogp.indices[i], mogp.indices[i + 1], mogp.indices[i + 2]));
                    }

                    // Gather material IDs for each face if present via renderBatches
                    var faceMaterialIds = new List<byte>(faces.Count);
                    if (mogp.renderBatches != null && mogp.renderBatches.Length > 0)
                    {
                        foreach (var batch in mogp.renderBatches)
                        {
                            var end = batch.firstFace + batch.numFaces;
                            for (var fi = batch.firstFace; fi < end; fi++)
                                faceMaterialIds.Add(batch.materialID);
                        }
                    }

                    var groupName = wmo.groupNames != null && gi < wmo.groupNames.Length
                        ? wmo.groupNames[gi].name
                        : $"group_{gi}";

                    groups.Add(new WmoGroup(groupName, verts, faces, faceMaterialIds));
                }
            }

            return (textures, groups);
        }
    }
}
