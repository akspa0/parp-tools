using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.PM4Module
{
    /// <summary>
    /// Patches WoWMuseum LK ADTs with PM4-derived WMO placements while preserving existing chunk data.
    /// Uses AdtPatcher to parse/write LK ADTs via the manual binary path (no Warcraft.NET serialization).
    /// </summary>
    public sealed class MuseumAdtPatcher
    {
        private readonly AdtPatcher _adtPatcher = new AdtPatcher();

        public void PatchWmoPlacements(
            string inputAdtPath,
            string outputAdtPath,
            IReadOnlyList<string> pm4WmoNames,
            IReadOnlyList<AdtPatcher.ModfEntry> newModfEntries,
            ref uint nextGlobalUniqueId)
        {
            if (string.IsNullOrWhiteSpace(inputAdtPath))
                throw new ArgumentException("Input ADT path is required", nameof(inputAdtPath));
            if (string.IsNullOrWhiteSpace(outputAdtPath))
                throw new ArgumentException("Output ADT path is required", nameof(outputAdtPath));
            if (pm4WmoNames is null)
                throw new ArgumentNullException(nameof(pm4WmoNames));
            if (newModfEntries is null)
                throw new ArgumentNullException(nameof(newModfEntries));

            // If no new placements for this tile, copy the source ADT unchanged.
            if (newModfEntries.Count == 0)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(outputAdtPath)!);
                File.Copy(inputAdtPath, outputAdtPath, overwrite: true);
                return;
            }

            var bytes = File.ReadAllBytes(inputAdtPath);
            var parsed = _adtPatcher.ParseAdt(bytes);

            // Ensure placement-related chunks exist in canonical order.
            EnsureChunkExists(parsed, "MWMO", "MMID");
            EnsureChunkExists(parsed, "MWID", "MWMO");
            EnsureChunkExists(parsed, "MODF", "MDDF");

            var mwmoChunk = parsed.FindChunk("MWMO");
            var mwidChunk = parsed.FindChunk("MWID");
            var modfChunk = parsed.FindChunk("MODF");

            if (mwmoChunk == null || mwidChunk == null || modfChunk == null)
                throw new InvalidOperationException("Failed to ensure required placement chunks (MWMO/MWID/MODF).");

            // Parse existing MWMO names from the chunk payload (null-terminated strings).
            // Dictionary to map WMO path -> NameId (index in MWMO)
            var nameToIndex = new Dictionary<string, uint>(StringComparer.OrdinalIgnoreCase);
            var mergedNames = new List<string>();

            // Parse existing MWMO names
            var existingNames = ParseNullTerminatedStrings(mwmoChunk.Data);

            // 1. Index existing names
            for (int i = 0; i < existingNames.Count; i++)
            {
                var name = existingNames[i];
                if (!nameToIndex.ContainsKey(name)) // Handle potential existing duplicates gracefully
                {
                    nameToIndex[name] = (uint)i;
                    mergedNames.Add(name);
                }
                else
                {
                    // Existing duplicate found in source ADT - pointing to first instance
                    mergedNames.Add(name); 
                }
            }
            // Note: We keep 'mergedNames' as 'existingNames' logic for minimal disruption, 
            // but for new entries we strictly check 'nameToIndex'.
            
            // 2. Add pm4WmoNames if they don't exist
            // We need a map from 'index in pm4WmoNames' -> 'final NameId'
            var pm4IndexToFinalId = new uint[pm4WmoNames.Count];

            for (int i = 0; i < pm4WmoNames.Count; i++)
            {
                var rawName = pm4WmoNames[i];
                if (string.IsNullOrWhiteSpace(rawName))
                {
                    pm4IndexToFinalId[i] = 0; // Point to empty string or 0 index? fallback
                    continue;
                }

                var normName = NormalizePath(rawName);
                
                if (nameToIndex.TryGetValue(normName, out uint existingId))
                {
                    pm4IndexToFinalId[i] = existingId;
                }
                else
                {
                    // New name
                    uint newId = (uint)mergedNames.Count;
                    mergedNames.Add(normName);
                    nameToIndex[normName] = newId;
                    pm4IndexToFinalId[i] = newId;
                }
            }

            // Rebuild MWMO / MWID using AdtPatcher helper methods.
            mwmoChunk.Data = _adtPatcher.BuildMwmoData(mergedNames);
            mwidChunk.Data = _adtPatcher.BuildMwidData(mergedNames);

            // Adjust NameId for new MODF entries using the mapping
            // CRITICAL: Also check for UniqueId collisions with existing MODF entries!
            var existingModfData = modfChunk.Data ?? Array.Empty<byte>();
            
            // Parse existing MODF entries to collect their UniqueIds
            var existingUniqueIds = new HashSet<uint>();
            if (existingModfData.Length >= 64)
            {
                int entryCount = existingModfData.Length / 64;
                for (int i = 0; i < entryCount; i++)
                {
                    int offset = i * 64 + 4; // UniqueId is at byte 4 in each 64-byte entry
                    if (offset + 4 <= existingModfData.Length)
                    {
                        uint existingId = BitConverter.ToUInt32(existingModfData, offset);
                        existingUniqueIds.Add(existingId);
                    }
                }
            }
            
            // Prepare new entries with adjusted NameId and conflict-free UniqueIds
            var adjustedEntries = new List<AdtPatcher.ModfEntry>(newModfEntries.Count);
            int reassignedCount = 0;
            
            foreach (var entry in newModfEntries)
            {
                var adjusted = entry;
                
                // Map the input NameId (which was index into pm4WmoNames) to final merged ID
                if (entry.NameId < pm4IndexToFinalId.Length)
                {
                    adjusted.NameId = pm4IndexToFinalId[entry.NameId];
                }
                else
                {
                    // Fallback should not happen if logic is correct
                    adjusted.NameId = 0; 
                }

                // Check for UniqueId collision with existing entries
                if (existingUniqueIds.Contains(adjusted.UniqueId))
                {
                    // Find next available ID using GLOBAL counter
                    while (existingUniqueIds.Contains(nextGlobalUniqueId))
                        nextGlobalUniqueId++;
                    adjusted.UniqueId = nextGlobalUniqueId++;
                    reassignedCount++;
                }
                existingUniqueIds.Add(adjusted.UniqueId);
                adjustedEntries.Add(adjusted);
            }
            
            if (reassignedCount > 0)
                Console.WriteLine($"[INFO] Reassigned {reassignedCount} UniqueIds to avoid conflicts with existing MODF entries");

            var newModfData = _adtPatcher.BuildModfData(adjustedEntries);

            var combined = new byte[existingModfData.Length + newModfData.Length];
            Buffer.BlockCopy(existingModfData, 0, combined, 0, existingModfData.Length);
            Buffer.BlockCopy(newModfData, 0, combined, existingModfData.Length, newModfData.Length);
            modfChunk.Data = combined;

            // Serialize ADT with recalculated MHDR/MCIN offsets.
            var outputBytes = _adtPatcher.WriteAdt(parsed);

            Directory.CreateDirectory(Path.GetDirectoryName(outputAdtPath)!);
            File.WriteAllBytes(outputAdtPath, outputBytes);
        }

        private static void EnsureChunkExists(AdtPatcher.ParsedAdt adt, string sig, string afterSig)
        {
            if (adt.FindChunk(sig) != null)
                return;

            adt.InsertChunkAfter(afterSig, new AdtPatcher.AdtChunk
            {
                Signature = sig,
                Data = Array.Empty<byte>()
            });
        }

        private static List<string> ParseNullTerminatedStrings(byte[] data)
        {
            var result = new List<string>();
            if (data == null || data.Length == 0)
                return result;

            int start = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] != 0)
                    continue;

                if (i > start)
                {
                    var s = Encoding.ASCII.GetString(data, start, i - start);
                    result.Add(s);
                }

                start = i + 1;
            }

            return result;
        }

        private static string NormalizePath(string path)
        {
            // WoW expects UPPERCASE paths with backslashes
            // WoW paths use forward slashes and lowercase
            var normalized = path.Replace('\\', '/');
            return normalized.ToLowerInvariant();
        }

        /// <summary>
        /// Patches doodad (M2) placements into an ADT's MDDF chunk.
        /// Similar to PatchWmoPlacements but for MMDX/MMID/MDDF instead of MWMO/MWID/MODF.
        /// </summary>
        public void PatchDoodadPlacements(
            string inputAdtPath,
            string outputAdtPath,
            IReadOnlyList<string> m2Names,
            IReadOnlyList<AdtPatcher.MddfEntry> newMddfEntries,
            ref uint nextGlobalUniqueId)
        {
            if (string.IsNullOrWhiteSpace(inputAdtPath))
                throw new ArgumentException("Input ADT path is required", nameof(inputAdtPath));
            if (string.IsNullOrWhiteSpace(outputAdtPath))
                throw new ArgumentException("Output ADT path is required", nameof(outputAdtPath));
            if (m2Names is null)
                throw new ArgumentNullException(nameof(m2Names));
            if (newMddfEntries is null)
                throw new ArgumentNullException(nameof(newMddfEntries));

            // If no new placements for this tile, don't modify (or copy unchanged).
            if (newMddfEntries.Count == 0)
            {
                // Don't copy - let caller decide (MODF patch may have already created output)
                return;
            }

            var bytes = File.ReadAllBytes(inputAdtPath);
            var parsed = _adtPatcher.ParseAdt(bytes);

            // Ensure doodad-related chunks exist in canonical order.
            // Order: MVER, MHDR, MCIN, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF...
            EnsureChunkExists(parsed, "MMDX", "MTEX");
            EnsureChunkExists(parsed, "MMID", "MMDX");
            EnsureChunkExists(parsed, "MDDF", "MWID");

            var mmdxChunk = parsed.FindChunk("MMDX");
            var mmidChunk = parsed.FindChunk("MMID");
            var mddfChunk = parsed.FindChunk("MDDF");

            if (mmdxChunk == null || mmidChunk == null || mddfChunk == null)
                throw new InvalidOperationException("Failed to ensure required doodad chunks (MMDX/MMID/MDDF).");

            // Parse existing MMDX names from the chunk payload (null-terminated strings).
            // Dictionary to map M2 path -> NameId (index in MMDX)
            var nameToIndex = new Dictionary<string, uint>(StringComparer.OrdinalIgnoreCase);
            var mergedNames = new List<string>();

            // Parse existing MMDX names
            var existingNames = ParseNullTerminatedStrings(mmdxChunk.Data);

            // 1. Index existing names
            for (int i = 0; i < existingNames.Count; i++)
            {
                var name = existingNames[i];
                if (!nameToIndex.ContainsKey(name))
                {
                    nameToIndex[name] = (uint)i;
                    mergedNames.Add(name);
                }
                else
                {
                    mergedNames.Add(name);
                }
            }

            // 2. Add new names if they don't exist
            var m2IndexToFinalId = new uint[m2Names.Count];

            foreach (var wrapper in m2Names.Select((name, index) => new { name, index }))
            {
                if (string.IsNullOrWhiteSpace(wrapper.name))
                {
                    m2IndexToFinalId[wrapper.index] = 0;
                    continue;
                }

                var normName = NormalizePath(wrapper.name);

                if (nameToIndex.TryGetValue(normName, out uint existingId))
                {
                    m2IndexToFinalId[wrapper.index] = existingId;
                }
                else
                {
                    uint newId = (uint)mergedNames.Count;
                    mergedNames.Add(normName);
                    nameToIndex[normName] = newId;
                    m2IndexToFinalId[wrapper.index] = newId;
                }
            }

            // Rebuild MMDX / MMID using AdtPatcher helper methods.
            mmdxChunk.Data = _adtPatcher.BuildMmdxData(mergedNames);
            mmidChunk.Data = _adtPatcher.BuildMmidData(mergedNames);

            // Parse existing MDDF entries to collect their UniqueIds
            var existingMddfData = mddfChunk.Data ?? Array.Empty<byte>();
            var existingUniqueIds = new HashSet<uint>();
            if (existingMddfData.Length >= 36)
            {
                int entryCount = existingMddfData.Length / 36;
                for (int i = 0; i < entryCount; i++)
                {
                    int offset = i * 36 + 4; // UniqueId is at byte 4 in each 36-byte entry
                    if (offset + 4 <= existingMddfData.Length)
                    {
                        uint existingId = BitConverter.ToUInt32(existingMddfData, offset);
                        existingUniqueIds.Add(existingId);
                    }
                }
            }

            // Prepare new entries with adjusted NameId and conflict-free UniqueIds
            var adjustedEntries = new List<AdtPatcher.MddfEntry>(newMddfEntries.Count);
            int reassignedCount = 0;

            foreach (var entry in newMddfEntries)
            {
                var adjusted = entry;
                
                // Map the input NameId to final merged ID
                if (entry.NameId < m2IndexToFinalId.Length)
                {
                    adjusted.NameId = m2IndexToFinalId[entry.NameId];
                }
                else
                {
                    adjusted.NameId = 0;
                }

                // Check for UniqueId collision with existing entries
                if (existingUniqueIds.Contains(adjusted.UniqueId))
                {
                    // Find next available ID using GLOBAL counter
                    while (existingUniqueIds.Contains(nextGlobalUniqueId))
                        nextGlobalUniqueId++;
                    adjusted.UniqueId = nextGlobalUniqueId++;
                    reassignedCount++;
                }
                existingUniqueIds.Add(adjusted.UniqueId);
                adjustedEntries.Add(adjusted);
            }

            if (reassignedCount > 0)
                Console.WriteLine($"[INFO] Reassigned {reassignedCount} MDDF UniqueIds to avoid conflicts");

            var newMddfData = _adtPatcher.BuildMddfData(adjustedEntries);

            var combined = new byte[existingMddfData.Length + newMddfData.Length];
            Buffer.BlockCopy(existingMddfData, 0, combined, 0, existingMddfData.Length);
            Buffer.BlockCopy(newMddfData, 0, combined, existingMddfData.Length, newMddfData.Length);
            mddfChunk.Data = combined;

            // Serialize ADT with recalculated MHDR/MCIN offsets.
            var outputBytes = _adtPatcher.WriteAdt(parsed);

            Directory.CreateDirectory(Path.GetDirectoryName(outputAdtPath)!);
            File.WriteAllBytes(outputAdtPath, outputBytes);
        }
    }
}
