using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace WoWToolbox.Core.WMO
{
    /// <summary>
    /// Provides utilities for loading WMO meshes and exporting them to OBJ format.
    /// </summary>
    public static class WmoMeshExporter
    {
        /// <summary>
        /// Loads and merges all group meshes for a WMO root file.
        /// </summary>
        /// <param name="rootWmoPath">Path to the WMO root file.</param>
        /// <returns>The merged WmoGroupMesh, or null if no valid groups were loaded.</returns>
        public static WmoGroupMesh? LoadMergedWmoMesh(string rootWmoPath)
        {
            if (!File.Exists(rootWmoPath))
                throw new FileNotFoundException($"WMO root file not found: {rootWmoPath}");

            string groupsDir = Path.GetDirectoryName(rootWmoPath) ?? ".";
            string rootBaseName = Path.GetFileNameWithoutExtension(rootWmoPath);
            var (groupCount, internalGroupNames) = WmoRootLoader.LoadGroupInfo(rootWmoPath);
            if (groupCount <= 0)
                return null;

            var groupMeshes = new List<WmoGroupMesh>();
            for (int i = 0; i < groupCount; i++)
            {
                string? groupPathToLoad = FindGroupFilePath(i, rootBaseName, groupsDir, internalGroupNames);
                if (groupPathToLoad == null)
                    continue;
                using var groupStream = File.OpenRead(groupPathToLoad);
                WmoGroupMesh mesh = WmoGroupMesh.LoadFromStream(groupStream, groupPathToLoad);
                if (mesh != null && mesh.Vertices.Count > 0 && mesh.Triangles.Count > 0)
                    groupMeshes.Add(mesh);
            }
            if (groupMeshes.Count == 0)
                return null;
            return WmoGroupMesh.MergeMeshes(groupMeshes);
        }

        /// <summary>
        /// Exports a merged WmoGroupMesh to OBJ format.
        /// </summary>
        /// <param name="mesh">The merged WmoGroupMesh to export.</param>
        /// <param name="outputPath">The output OBJ file path.</param>
        public static void SaveMergedWmoToObj(WmoGroupMesh mesh, string outputPath)
        {
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            WmoGroupMesh.SaveToObj(mesh, outputPath);
        }

        /// <summary>
        /// Finds the group file path for a given group index.
        /// </summary>
        /// <param name="groupIndex">The group index.</param>
        /// <param name="rootBaseName">The base name of the root WMO file (without extension).</param>
        /// <param name="groupsDir">The directory containing the group files.</param>
        /// <param name="internalGroupNames">The list of internal group names from the WMO root.</param>
        /// <returns>The path to the group file, or null if not found.</returns>
        public static string? FindGroupFilePath(int groupIndex, string rootBaseName, string groupsDir, List<string> internalGroupNames)
        {
            string? groupPathToLoad = null;
            string internalName = (groupIndex < internalGroupNames.Count) ? internalGroupNames[groupIndex] : null;
            string numberedName = $"{rootBaseName}_{groupIndex:D3}.wmo";

            if (!string.IsNullOrEmpty(internalName))
            {
                string potentialInternalPath = Path.Combine(groupsDir, internalName);
                if (File.Exists(potentialInternalPath))
                    groupPathToLoad = potentialInternalPath;
            }
            if (groupPathToLoad == null)
            {
                string potentialNumberedPath = Path.Combine(groupsDir, numberedName);
                if (File.Exists(potentialNumberedPath))
                    groupPathToLoad = potentialNumberedPath;
            }
            return groupPathToLoad;
        }
    }
} 