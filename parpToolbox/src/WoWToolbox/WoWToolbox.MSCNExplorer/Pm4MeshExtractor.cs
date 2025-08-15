using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core; // For PM4File, chunk classes, vectors etc.
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using WoWToolbox.Core.Models; // For MeshData
using WoWToolbox.Core.Vectors; // For C3Vector if needed, though likely not

namespace WoWToolbox.MSCNExplorer
{
    /// <summary>
    /// Responsible for extracting renderable mesh geometry from PM4/PD4 files.
    /// Logic originally adapted from WoWToolbox.Tests.Navigation.PM4.PM4FileTests.
    /// Kept separate from Core temporarily per user request.
    /// </summary>
    public class Pm4MeshExtractor
    {
        // Constants for transformation (potentially move to Core or config later)
        private const float CoordinateOffset = 17066.666f;
        // ScaleFactor seems unused in the relevant vertex transform, omitting for now
        // private const float ScaleFactor = 36.0f;

        // Transforms MSVT vertices based on PM4FileTests logic for render_mesh_transformed.obj
        // Y, X, Z mapping first, then offset-X, offset-Y, Z.
        private static Vector3 MsvtToWorld_PM4(MsvtVertex v)
        {
            // Apply Y, X, Z mapping first (as done before writing original vertex)
            float originalX = v.Y;
            float originalY = v.X;
            float originalZ = v.Z;

            // Apply final offset transformation (as done before writing transformed vertex)
            float transformedX = CoordinateOffset - originalY; // Offset applied to original X (v.X)
            float transformedY = CoordinateOffset - originalX; // Offset applied to original Y (v.Y)
            float transformedZ = originalZ; // Z remains unchanged

            return new Vector3(transformedX, transformedY, transformedZ);
        }

        /// <summary>
        /// Extracts vertices and triangles from the MSVT, MSVI, and MSUR chunks of a PM4 file.
        /// Can optionally filter the results to include only geometry linked to a specific WMO file via MDBH.
        /// </summary>
        /// <param name="pm4File">The loaded PM4File object.</param>
        /// <param name="targetWmoFilename">Optional. The filename of the WMO to filter by (e.g., "path\\to\\wmo.wmo"). If null or empty, extracts all state 0 geometry.</param>
        /// <returns>A MeshData object containing the extracted geometry, or null if essential chunks are missing.</returns>
        public MeshData? ExtractMesh(PM4File pm4File, string? targetWmoFilename = null)
        {
            // Basic null checks for essential chunks
            if (pm4File?.MSVT?.Vertices == null || pm4File.MSVI?.Indices == null || pm4File.MSUR?.Entries == null)
            {
                Console.WriteLine("[ERR][Pm4MeshExtractor] Missing essential chunks (MSVT, MSVI, or MSUR) for mesh extraction.");
                return null;
            }

            bool isFiltering = !string.IsNullOrEmpty(targetWmoFilename);
            uint targetMdbhIndex = uint.MaxValue;
            HashSet<uint>? allowedMdosIndices = null; // HashSet for efficient lookup of *indices within MDOS chunk*

            Console.WriteLine(isFiltering
                ? $"[DEBUG][Pm4MeshExtractor] Starting extraction, filtering for WMO: '{targetWmoFilename}'"
                : "[DEBUG][Pm4MeshExtractor] Starting extraction (unfiltered)...");

            // --- Pre-filtering Steps (if filtering is enabled) ---
            if (isFiltering)
            {
                // Check for chunks required for filtering
                if (pm4File.MDBH?.Entries == null || pm4File.MDOS?.Entries == null)
                {
                    Console.WriteLine($"[WARN][Pm4MeshExtractor] Cannot filter by WMO '{targetWmoFilename}'. Required MDBH or MDOS chunk is missing or empty in this PM4 file. Returning empty mesh.");
                    return new MeshData(); // Return empty, not null
                }
                if (pm4File.MDBH.Entries.Count == 0 || pm4File.MDOS.Entries.Count == 0)
                {
                     Console.WriteLine($"[WARN][Pm4MeshExtractor] Cannot filter by WMO '{targetWmoFilename}'. MDBH or MDOS chunk contains no entries. Returning empty mesh.");
                    return new MeshData(); // Return empty, not null
                }

                // 1. Find target Index in MDBH based on filename
                var mdbhEntry = pm4File.MDBH.Entries.FirstOrDefault(e => string.Equals(e.Filename, targetWmoFilename, StringComparison.OrdinalIgnoreCase));
                if (mdbhEntry == null)
                {
                    Console.WriteLine($"[WARN][Pm4MeshExtractor] Target WMO '{targetWmoFilename}' not found in MDBH chunk. Returning empty mesh.");
                    return new MeshData(); // Return empty, not null
                }
                targetMdbhIndex = mdbhEntry.Index;
                Console.WriteLine($"[DEBUG][Pm4MeshExtractor] Found target WMO in MDBH at Index: {targetMdbhIndex}");

                // 2. Find MDOS entries matching the MDBH index and collect their *list indices*
                allowedMdosIndices = new HashSet<uint>();
                for (int mdosListIndex = 0; mdosListIndex < pm4File.MDOS.Entries.Count; mdosListIndex++)
                {
                    if (pm4File.MDOS.Entries[mdosListIndex].m_destructible_building_index == targetMdbhIndex)
                    {
                        allowedMdosIndices.Add((uint)mdosListIndex); // Add the index *of* the entry in the MDOS list
                    }
                }

                if (allowedMdosIndices.Count == 0)
                {
                    Console.WriteLine($"[WARN][Pm4MeshExtractor] No MDOS entries found linked to MDBH Index {targetMdbhIndex} for WMO '{targetWmoFilename}'. Returning empty mesh.");
                    return new MeshData(); // Return empty, not null
                }
                Console.WriteLine($"[DEBUG][Pm4MeshExtractor] Found {allowedMdosIndices.Count} MDOS entries linked to MDBH Index {targetMdbhIndex}.");
            }
            // End pre-filtering steps

            // Add checks for MDOS/MDSF even when not filtering, as they are used for state checking
            if (pm4File.MDOS?.Entries == null || pm4File.MDSF?.Entries == null)
            {
                 Console.WriteLine("[WARN][Pm4MeshExtractor] Missing MDOS or MDSF chunk. Will only extract unlinked MSUR faces (assumed state 0).");
                 // Allow proceeding, but only unlinked faces will be extracted.
            }

            var meshData = new MeshData();
            int msvtVertexCount = pm4File.MSVT.Vertices.Count;
            int msviIndexCount = pm4File.MSVI.Indices.Count;
            int msurEntryCount = pm4File.MSUR.Entries.Count;
            int mdosEntryCount = pm4File.MDOS?.Entries?.Count ?? 0;

            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MSVT Vertices: {msvtVertexCount}");
            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MSVI Indices: {msviIndexCount}");
            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MSUR Entries: {msurEntryCount}");
            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MDOS Entries: {mdosEntryCount}");
            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MDSF Entries: {pm4File.MDSF?.Entries?.Count ?? 0}");
            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MDBH Entries: {pm4File.MDBH?.Entries?.Count ?? 0}");

            if (msvtVertexCount == 0)
            {
                Console.WriteLine("[WARN][Pm4MeshExtractor] No MSVT vertices found. Cannot extract mesh.");
                return meshData; // Return empty mesh data
            }

            // --- 1. Process Vertices (MSVT) ---
            // We process all vertices initially, as filtering happens based on faces/MSUR links.
            // Unused vertices won't be included in the final OBJ if no faces reference them.
            for (int i = 0; i < msvtVertexCount; i++)
            {
                var msvtVertex = pm4File.MSVT.Vertices[i];
                Vector3 worldPos = MsvtToWorld_PM4(msvtVertex); // Apply transformation
                meshData.Vertices.Add(worldPos);
            }
            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] Processed {meshData.Vertices.Count} total vertices initially.");

            // --- 2. Process Faces (MSUR -> MSVI -> MSVT index) ---
            // Pre-build MDSF lookup if MDSF is available
            var mdsfLookup = pm4File.MDSF?.Entries?.ToDictionary(e => e.msur_index, e => e.mdos_index)
                             ?? new Dictionary<uint, uint>();
            
            int processedMsurCount = 0;

            for (uint msurIndex = 0; msurIndex < msurEntryCount; msurIndex++)
            {
                var msurEntry = pm4File.MSUR.Entries[(int)msurIndex];

                // --- Apply WMO Filter (if active) ---
                if (isFiltering)
                {
                    // MsurEntry.MdosIndex points to an *entry* in the MDOS list.
                    // We need to check if *that entry's index* is in our allowed set.
                    if (allowedMdosIndices == null || !allowedMdosIndices.Contains(msurEntry.MdosIndex))
                    {
                        // This MSUR entry does not belong to the target WMO.
                        continue; // Skip to the next MSUR entry
                    }
                     // If we reach here, the MSUR entry passed the WMO filter.
                }
                // --- End WMO Filter ---

                int firstMsvi = (int)msurEntry.MsviFirstIndex;
                int indexCount = msurEntry.IndexCount;

                // Validate MSVI range
                if (firstMsvi < 0 || firstMsvi + indexCount > msviIndexCount)
                {
                    Console.WriteLine($"[WARN][Pm4MeshExtractor] MSUR Entry {msurIndex} defines invalid MSVI range [First:{firstMsvi}, Count:{indexCount}] (Max MSVI Index: {msviIndexCount - 1}). Skipping entry.");
                    continue;
                }

                if (indexCount < 3 || indexCount % 3 != 0)
                {
                    // Still log this warning, even if filtering
                    Console.WriteLine($"[WARN][Pm4MeshExtractor] MSUR Entry {msurIndex} index count ({indexCount}) is less than 3 or not divisible by 3. Skipping entry.");
                    continue;
                }

                // Check link via MDSF -> MDOS for state (only state 0 is considered render geometry)
                bool isStateZero = false;
                if (!mdsfLookup.TryGetValue(msurIndex, out uint linkedMdosIndexFromMdsf))
                {
                    // Case 1: No MDSF Link Found - Assume default state (0)
                    isStateZero = true;
                    // Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MSUR Entry {msurIndex}: No MDSF link found. Assuming state 0.");
                }
                else
                {
                    // Case 2: MDSF Link Found
                    if (pm4File.MDOS?.Entries == null || linkedMdosIndexFromMdsf >= mdosEntryCount)
                    {
                        Console.WriteLine($"[WARN][Pm4MeshExtractor] MSUR Entry {msurIndex}: MDSF links to invalid MDOS index {linkedMdosIndexFromMdsf} (Max: {mdosEntryCount - 1}) or MDOS missing. Skipping face.");
                        continue;
                    }
                    var linkedMdosEntry = pm4File.MDOS!.Entries[(int)linkedMdosIndexFromMdsf];
                    isStateZero = linkedMdosEntry.destruction_state == 0;
                    // Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MSUR Entry {msurIndex}: Linked via MDSF to MDOS {linkedMdosIndexFromMdsf}, State: {linkedMdosEntry.destruction_state}");
                }

                // Only process faces if they are considered state 0
                if (!isStateZero)
                {
                    // Console.WriteLine($"[DEBUG][Pm4MeshExtractor] MSUR Entry {msurIndex}: Skipping face (State != 0).");
                    continue;
                }
                
                processedMsurCount++; // Count MSUR entries we are actually processing faces for

                // Process triangles for this MSUR entry
                for (int i = 0; i < indexCount; i += 3)
                {
                    try
                    {
                        // Get the indices into MSVI
                        int msviIdx0 = firstMsvi + i + 0;
                        int msviIdx1 = firstMsvi + i + 1;
                        int msviIdx2 = firstMsvi + i + 2;

                        // Get the indices into MSVT from MSVI - Use int as MSVI can exceed ushort.MaxValue
                        int msvtIdx0 = (int)pm4File.MSVI.Indices[msviIdx0];
                        int msvtIdx1 = (int)pm4File.MSVI.Indices[msviIdx1];
                        int msvtIdx2 = (int)pm4File.MSVI.Indices[msviIdx2];

                        // Validate MSVT indices against the *actual* number of vertices processed
                        if (msvtIdx0 >= meshData.Vertices.Count || msvtIdx1 >= meshData.Vertices.Count || msvtIdx2 >= meshData.Vertices.Count)
                        {
                            Console.WriteLine($"[WARN][Pm4MeshExtractor] MSUR {msurIndex}: Triangle vertex index out of bounds ({msvtIdx0}, {msvtIdx1}, {msvtIdx2} vs {meshData.Vertices.Count}). Skipping triangle.");
                            continue; // Skip this triangle only
                        }

                        meshData.Indices.Add(msvtIdx0);
                        meshData.Indices.Add(msvtIdx1);
                        meshData.Indices.Add(msvtIdx2);
                    }
                    catch (Exception ex) // Catch potential IndexOutOfRange or other errors defensively
                    {
                        Console.WriteLine($"[ERR][Pm4MeshExtractor] Error processing triangle for MSUR entry {msurIndex}, MSVI index {firstMsvi + i}: {ex.Message}. Skipping triangle.");
                    }
                }
            } // End MSUR loop

            Console.WriteLine(isFiltering
                ? $"[DEBUG][Pm4MeshExtractor] Processed {processedMsurCount} MSUR entries matching filter criteria."
                : $"[DEBUG][Pm4MeshExtractor] Processed {processedMsurCount} MSUR entries (State 0 / Unlinked).");
            Console.WriteLine($"[DEBUG][Pm4MeshExtractor] Extracted {meshData.Indices.Count / 3} total triangles.");

            if (meshData.Vertices.Count > 0 && meshData.Indices.Count == 0)
            {
                 Console.WriteLine(isFiltering
                     ? $"[WARN][Pm4MeshExtractor] Filtered mesh for '{targetWmoFilename}' has vertices but 0 triangles."
                     : "[WARN][Pm4MeshExtractor] Extracted mesh has vertices but 0 triangles (State 0 / Unlinked).");
            }
            else if (meshData.Vertices.Count == 0 && meshData.Indices.Count > 0)
            {
                 // This shouldn't happen if validation works, but good to note
                 Console.WriteLine("[WARN][Pm4MeshExtractor] Extracted mesh has 0 vertices but > 0 triangles. Data inconsistency likely.");
                 meshData.Indices.Clear(); // Clear inconsistent indices
            }

            return meshData;
        }
    }
} 