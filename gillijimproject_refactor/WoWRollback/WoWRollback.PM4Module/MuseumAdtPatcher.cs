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
            IReadOnlyList<AdtPatcher.ModfEntry> newModfEntries)
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
            var existingNames = ParseNullTerminatedStrings(mwmoChunk.Data);
            int existingCount = existingNames.Count;

            // Build full WMO name list: existing names + PM4-derived names (normalized).
            var allNames = new List<string>(existingNames.Count + pm4WmoNames.Count);
            allNames.AddRange(existingNames);

            foreach (var name in pm4WmoNames)
            {
                if (string.IsNullOrWhiteSpace(name))
                {
                    allNames.Add(string.Empty);
                    continue;
                }

                allNames.Add(NormalizePath(name));
            }

            // Rebuild MWMO / MWID using AdtPatcher helper methods.
            mwmoChunk.Data = _adtPatcher.BuildMwmoData(allNames);
            mwidChunk.Data = _adtPatcher.BuildMwidData(allNames);

            // Adjust NameId for new MODF entries and append them to any existing MODF data.
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
            uint nextAvailableId = 200_000_000; // High base to avoid conflicts
            int reassignedCount = 0;
            
            foreach (var entry in newModfEntries)
            {
                var adjusted = entry;
                adjusted.NameId = (uint)(existingCount + entry.NameId);
                
                // Check for UniqueId collision with existing entries
                if (existingUniqueIds.Contains(adjusted.UniqueId))
                {
                    // Find next available ID
                    while (existingUniqueIds.Contains(nextAvailableId))
                        nextAvailableId++;
                    adjusted.UniqueId = nextAvailableId++;
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
            var normalized = path.Replace('/', '\\');
            return normalized.ToUpperInvariant();
        }
    }
}
