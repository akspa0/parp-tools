using System.Numerics;
using WoWToolbox.Core;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using WoWToolbox.Core.Navigation.PM4.Models;

namespace WoWToolbox.PM4Parsing.BuildingExtraction
{
    /// <summary>
    /// Main building extraction service for PM4 navigation files.
    /// Implements the proven "flexible method" that handles both MDSF/MDOS and MSLK-based extraction.
    /// </summary>
    public class PM4BuildingExtractor
    {
        /// <summary>
        /// Extracts complete buildings from a PM4 file using the optimal strategy.
        /// Automatically detects whether to use MDSF/MDOS building IDs or MSLK root node grouping.
        /// </summary>
        /// <param name="pm4File">The loaded PM4 file</param>
        /// <param name="sourceFileName">Source filename for metadata</param>
        /// <returns>List of complete building models ready for export</returns>
        public List<CompleteWMOModel> ExtractBuildings(PM4File pm4File, string sourceFileName)
        {
            if (pm4File.MSLK?.Entries == null)
                throw new ArgumentException("PM4 file missing required MSLK chunk");

            // Determine optimal extraction strategy
            bool hasMdsfMdos = pm4File.MDSF?.Entries?.Count > 0 && pm4File.MDOS?.Entries?.Count > 0;
            
            if (hasMdsfMdos)
            {
                return ExtractUsingMdsfMdosSystem(pm4File, sourceFileName);
            }
            else
            {
                return ExtractUsingMslkRootNodesWithSpatialClustering(pm4File, sourceFileName);
            }
        }

        /// <summary>
        /// Extracts buildings using MDSF/MDOS building ID system.
        /// Groups MSUR surfaces by their associated building IDs for precise building separation.
        /// </summary>
        private List<CompleteWMOModel> ExtractUsingMdsfMdosSystem(PM4File pm4File, string sourceFileName)
        {
            // Group MSUR surfaces by building ID
            var buildingGroups = new Dictionary<uint, List<int>>();
            
            for (int msurIndex = 0; msurIndex < pm4File.MSUR.Entries.Count; msurIndex++)
            {
                var mdsfEntry = pm4File.MDSF.Entries.FirstOrDefault(entry => entry.msur_index == msurIndex);
                
                if (mdsfEntry != null)
                {
                    var mdosIndex = mdsfEntry.mdos_index;
                    if (mdosIndex < pm4File.MDOS.Entries.Count)
                    {
                        var mdosEntry = pm4File.MDOS.Entries[(int)mdosIndex];
                        var buildingId = mdosEntry.m_destructible_building_index;
                        
                        if (!buildingGroups.ContainsKey(buildingId))
                            buildingGroups[buildingId] = new List<int>();
                        buildingGroups[buildingId].Add(msurIndex);
                    }
                }
            }

            // Create buildings from grouped surfaces
            var buildings = new List<CompleteWMOModel>();
            int buildingIndex = 0;
            
            foreach (var buildingGroup in buildingGroups.OrderBy(g => g.Key))
            {
                var buildingId = buildingGroup.Key;
                var surfaceIndices = buildingGroup.Value;
                
                var building = CreateBuildingFromMSURSurfaces(pm4File, surfaceIndices, sourceFileName, buildingIndex);
                building.Metadata["BuildingID"] = $"0x{buildingId:X8}";
                building.Metadata["RenderSurfaces"] = surfaceIndices.Count;
                building.Metadata["ExtractionMethod"] = "MDSF/MDOS";
                
                buildings.Add(building);
                buildingIndex++;
            }
            
            return buildings;
        }

        /// <summary>
        /// Extracts buildings using MSLK root nodes with spatial clustering.
        /// Falls back method when MDSF/MDOS data is not available.
        /// Uses multiple strategies for universal PM4 file compatibility.
        /// </summary>
        private List<CompleteWMOModel> ExtractUsingMslkRootNodesWithSpatialClustering(PM4File pm4File, string sourceFileName)
        {
            // Strategy 1: Find MSLK root nodes (self-referencing entries)
            var rootNodes = new List<(int nodeIndex, MSLKEntry entry)>();
            
            for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
            {
                var entry = pm4File.MSLK.Entries[i];
                if (entry.Unknown_0x04 == i) // Self-referencing = root node
                {
                    rootNodes.Add((i, entry));
                }
            }

            var buildings = new List<CompleteWMOModel>();
            bool hasValidRoots = false;
            
            // Try to extract from root nodes first
            for (int buildingIndex = 0; buildingIndex < rootNodes.Count; buildingIndex++)
            {
                var (rootNodeIndex, rootEntry) = rootNodes[buildingIndex];
                var rootGroupKey = rootEntry.Unknown_0x04;
                
                // Get MSLK structural elements for this building
                var buildingEntries = pm4File.MSLK.Entries
                    .Select((entry, index) => new { entry, index })
                    .Where(x => x.entry.Unknown_0x04 == rootGroupKey && 
                               x.entry.MspiFirstIndex >= 0 && 
                               x.entry.MspiIndexCount > 0)
                    .ToList();
                
                if (buildingEntries.Count == 0) continue;
                
                // Calculate bounding box of structural elements
                var structuralBounds = CalculateStructuralElementsBounds(pm4File, buildingEntries.Cast<dynamic>().ToList());
                if (!structuralBounds.HasValue) continue;
                
                // Find MSUR surfaces within or near this bounding box
                var nearbySurfaces = FindMSURSurfacesNearBounds(pm4File, structuralBounds.Value, tolerance: 50.0f);
                
                // Create hybrid building with both structural and render data
                var building = CreateHybridBuilding(pm4File, buildingEntries.Cast<dynamic>().ToList(), nearbySurfaces, sourceFileName, buildingIndex);
                building.Metadata["RootNodeIndex"] = rootNodeIndex;
                building.Metadata["StructuralElements"] = buildingEntries.Count;
                building.Metadata["RenderSurfaces"] = nearbySurfaces.Count;
                building.Metadata["ExtractionMethod"] = "MSLK Root Nodes";
                
                buildings.Add(building);
                if (building.Vertices.Count > 0) hasValidRoots = true;
            }

            // Strategy 2: Fallback - Group by Unknown_0x04 if no valid roots found
            if (!hasValidRoots && rootNodes.Count > 0)
            {
                Console.WriteLine("Root nodes found but no geometry detected, using enhanced fallback strategy...");
                
                var groupedEntries = pm4File.MSLK.Entries
                    .Select((entry, index) => new { entry, index })
                    .Where(x => x.entry.MspiFirstIndex >= 0 && x.entry.MspiIndexCount > 0)
                    .GroupBy(x => x.entry.Unknown_0x04)
                    .Where(g => g.Count() > 0)
                    .ToList();

                Console.WriteLine($"Found {groupedEntries.Count} geometry groups");
                buildings.Clear(); // Start fresh with fallback strategy

                for (int groupIndex = 0; groupIndex < groupedEntries.Count; groupIndex++)
                {
                    var group = groupedEntries[groupIndex];
                    var buildingEntries = group.ToList();
                    
                    // Calculate bounding box of structural elements
                    var structuralBounds = CalculateStructuralElementsBounds(pm4File, buildingEntries.Cast<dynamic>().ToList());
                    if (!structuralBounds.HasValue) continue;
                    
                    // Find MSUR surfaces within or near this bounding box
                    var nearbySurfaces = FindMSURSurfacesNearBounds(pm4File, structuralBounds.Value, tolerance: 50.0f);
                    
                    // Create hybrid building
                    var building = CreateHybridBuilding(pm4File, buildingEntries.Cast<dynamic>().ToList(), nearbySurfaces, sourceFileName, groupIndex);
                    building.Metadata["GroupKey"] = group.Key;
                    building.Metadata["StructuralElements"] = buildingEntries.Count;
                    building.Metadata["RenderSurfaces"] = nearbySurfaces.Count;
                    building.Metadata["ExtractionMethod"] = "MSLK Fallback Grouping";
                    
                    buildings.Add(building);
                }
            }
            
            return buildings;
        }

        /// <summary>
        /// Creates a building from MSUR render surfaces only.
        /// Used in MDSF/MDOS extraction mode.
        /// </summary>
        private CompleteWMOModel CreateBuildingFromMSURSurfaces(PM4File pm4File, List<int> surfaceIndices, string sourceFileName, int buildingIndex)
        {
            var building = new CompleteWMOModel
            {
                FileName = $"{sourceFileName}_Building_{buildingIndex + 1:D2}",
                Category = "MDSF_Building",
                MaterialName = "Building_Material"
            };
            
            if (pm4File.MSVT?.Vertices == null || pm4File.MSVI?.Indices == null || pm4File.MSUR?.Entries == null)
                return building;
            
            // --- NEW: Only add MSVT vertices actually referenced by this building's surfaces ---
            var usedMsvtIndices = new HashSet<uint>();
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= pm4File.MSUR.Entries.Count) continue;
                var surface = pm4File.MSUR.Entries[surfaceIndex];
                if (surface.IndexCount < 3) continue;
                for (int i = 0; i < surface.IndexCount; i++)
                {
                    if (surface.MsviFirstIndex + i < pm4File.MSVI.Indices.Count)
                    {
                        uint msvtIdx = pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i];
                        if (msvtIdx < pm4File.MSVT.Vertices.Count)
                            usedMsvtIndices.Add(msvtIdx);
                    }
                }
            }
            // Build mapping: global MSVT index → local OBJ vertex index
            var msvtIndexToLocal = new Dictionary<uint, int>();
            foreach (var msvtIdx in usedMsvtIndices.OrderBy(idx => idx))
            {
                var worldCoords = Pm4CoordinateTransforms.FromMsvtVertex(pm4File.MSVT.Vertices[(int)msvtIdx]);
                msvtIndexToLocal[msvtIdx] = building.Vertices.Count;
                building.Vertices.Add(worldCoords);
            }
            // Add faces, remapping indices
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= pm4File.MSUR.Entries.Count) continue;
                var surface = pm4File.MSUR.Entries[surfaceIndex];
                if (surface.IndexCount < 3) continue;
                for (int i = 0; i < surface.IndexCount - 2; i += 3)
                {
                    if (surface.MsviFirstIndex + i + 2 < pm4File.MSVI.Indices.Count)
                    {
                        uint v1Index = pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i];
                        uint v2Index = pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i + 1];
                        uint v3Index = pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i + 2];
                        if (msvtIndexToLocal.ContainsKey(v1Index) && msvtIndexToLocal.ContainsKey(v2Index) && msvtIndexToLocal.ContainsKey(v3Index))
                        {
                            building.TriangleIndices.Add(msvtIndexToLocal[v1Index]);
                            building.TriangleIndices.Add(msvtIndexToLocal[v2Index]);
                            building.TriangleIndices.Add(msvtIndexToLocal[v3Index]);
                        }
                    }
                }
            }
            // Generate normals for the complete model
            CompleteWMOModelUtilities.GenerateNormals(building);
            return building;
        }

        /// <summary>
        /// Creates a hybrid building combining MSLK structural elements and nearby MSUR surfaces.
        /// Used in MSLK root node extraction mode.
        /// </summary>
        private CompleteWMOModel CreateHybridBuilding(PM4File pm4File, List<dynamic> structuralEntries, List<int> nearbySurfaces, string sourceFileName, int buildingIndex)
        {
            var building = new CompleteWMOModel
            {
                FileName = $"{sourceFileName}_Building_{buildingIndex + 1:D2}",
                Category = "Hybrid_Building",
                MaterialName = "Hybrid_Material"
            };
            
            // Add MSPV vertices first (structural)
            if (pm4File.MSPV?.Vertices != null)
            {
                foreach (var vertex in pm4File.MSPV.Vertices)
                {
                    var worldCoords = Pm4CoordinateTransforms.FromMspvVertex(vertex);
                    building.Vertices.Add(worldCoords);
                }
            }
            
            var structuralVertexOffset = building.Vertices.Count;
            
            // Add MSVT vertices (render)
            if (pm4File.MSVT?.Vertices != null)
            {
                foreach (var vertex in pm4File.MSVT.Vertices)
                {
                    var worldCoords = Pm4CoordinateTransforms.FromMsvtVertex(vertex);
                    building.Vertices.Add(worldCoords);
                }
            }
            
            // Add structural elements geometry
            if (pm4File.MSPI?.Indices != null)
            {
                foreach (var entryData in structuralEntries)
                {
                    var entry = entryData.entry;
                    
                    var validIndices = new List<int>();
                    for (int i = entry.MspiFirstIndex; i < entry.MspiFirstIndex + entry.MspiIndexCount && i < pm4File.MSPI.Indices.Count; i++)
                    {
                        uint mspvIndex = pm4File.MSPI.Indices[i];
                        if (mspvIndex < pm4File.MSPV.Vertices.Count)
                        {
                            validIndices.Add((int)mspvIndex);
                        }
                    }
                    
                    // Create triangular faces from structural points
                    for (int i = 0; i < validIndices.Count - 2; i += 3)
                    {
                        building.TriangleIndices.Add(validIndices[i]);
                        building.TriangleIndices.Add(validIndices[i + 1]);
                        building.TriangleIndices.Add(validIndices[i + 2]);
                    }
                }
            }
            
            // Add nearby MSUR surfaces
            if (pm4File.MSUR?.Entries != null && pm4File.MSVI?.Indices != null)
            {
                foreach (int surfaceIndex in nearbySurfaces)
                {
                    if (surfaceIndex >= pm4File.MSUR.Entries.Count) continue;
                    
                    var surface = pm4File.MSUR.Entries[surfaceIndex];
                    if (surface.IndexCount < 3) continue;
                    
                    for (int i = 0; i < surface.IndexCount - 2; i += 3)
                    {
                        if (surface.MsviFirstIndex + i + 2 < pm4File.MSVI.Indices.Count)
                        {
                            uint v1Index = pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i];
                            uint v2Index = pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i + 1];
                            uint v3Index = pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i + 2];
                            
                            if (v1Index < pm4File.MSVT.Vertices.Count && 
                                v2Index < pm4File.MSVT.Vertices.Count && 
                                v3Index < pm4File.MSVT.Vertices.Count)
                            {
                                building.TriangleIndices.Add(structuralVertexOffset + (int)v1Index);
                                building.TriangleIndices.Add(structuralVertexOffset + (int)v2Index);
                                building.TriangleIndices.Add(structuralVertexOffset + (int)v3Index);
                            }
                        }
                    }
                }
            }
            
            // Generate normals for the complete model
            CompleteWMOModelUtilities.GenerateNormals(building);
            
            return building;
        }

        /// <summary>
        /// Calculates bounding box of MSLK structural elements.
        /// </summary>
        private (Vector3 min, Vector3 max)? CalculateStructuralElementsBounds(PM4File pm4File, List<dynamic> buildingEntries)
        {
            if (pm4File.MSPV?.Vertices == null || pm4File.MSPI?.Indices == null) return null;
            
            var allVertices = new List<Vector3>();
            
            foreach (var entryData in buildingEntries)
            {
                var entry = entryData.entry;
                
                for (int i = entry.MspiFirstIndex; i < entry.MspiFirstIndex + entry.MspiIndexCount && i < pm4File.MSPI.Indices.Count; i++)
                {
                    uint mspvIndex = pm4File.MSPI.Indices[i];
                    if (mspvIndex < pm4File.MSPV.Vertices.Count)
                    {
                        var vertex = pm4File.MSPV.Vertices[(int)mspvIndex];
                        var worldCoords = Pm4CoordinateTransforms.FromMspvVertex(vertex);
                        allVertices.Add(worldCoords);
                    }
                }
            }
            
            if (allVertices.Count == 0) return null;
            
            var minX = allVertices.Min(v => v.X);
            var minY = allVertices.Min(v => v.Y);
            var minZ = allVertices.Min(v => v.Z);
            var maxX = allVertices.Max(v => v.X);
            var maxY = allVertices.Max(v => v.Y);
            var maxZ = allVertices.Max(v => v.Z);
            
            return (new Vector3(minX, minY, minZ), new Vector3(maxX, maxY, maxZ));
        }

        /// <summary>
        /// Finds MSUR surfaces that are spatially near the given bounds.
        /// </summary>
        private List<int> FindMSURSurfacesNearBounds(PM4File pm4File, (Vector3 min, Vector3 max) bounds, float tolerance)
        {
            var nearbySurfaces = new List<int>();
            
            if (pm4File.MSUR?.Entries == null || pm4File.MSVT?.Vertices == null || pm4File.MSVI?.Indices == null)
                return nearbySurfaces;
            
            for (int surfaceIndex = 0; surfaceIndex < pm4File.MSUR.Entries.Count; surfaceIndex++)
            {
                var surface = pm4File.MSUR.Entries[surfaceIndex];
                
                // Check if any vertex of this surface is near the bounds
                bool isNearby = false;
                for (int i = (int)surface.MsviFirstIndex; i < surface.MsviFirstIndex + surface.IndexCount && i < pm4File.MSVI.Indices.Count; i++)
                {
                    uint msvtIndex = pm4File.MSVI.Indices[i];
                    if (msvtIndex < pm4File.MSVT.Vertices.Count)
                    {
                        var vertex = pm4File.MSVT.Vertices[(int)msvtIndex];
                        var worldCoords = Pm4CoordinateTransforms.FromMsvtVertex(vertex);
                        
                        // Check if vertex is within expanded bounds
                        if (worldCoords.X >= bounds.min.X - tolerance && worldCoords.X <= bounds.max.X + tolerance &&
                            worldCoords.Y >= bounds.min.Y - tolerance && worldCoords.Y <= bounds.max.Y + tolerance &&
                            worldCoords.Z >= bounds.min.Z - tolerance && worldCoords.Z <= bounds.max.Z + tolerance)
                        {
                            isNearby = true;
                            break;
                        }
                    }
                }
                
                if (isNearby)
                {
                    nearbySurfaces.Add(surfaceIndex);
                }
            }
            
            return nearbySurfaces;
        }
    }
} 