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
            var adjustedEntries = new List<AdtPatcher.ModfEntry>(newModfEntries.Count);
            foreach (var entry in newModfEntries)
            {
                var adjusted = entry;
                adjusted.NameId = (uint)(existingCount + entry.NameId);
                adjustedEntries.Add(adjusted);
            }

            var existingModfData = modfChunk.Data ?? Array.Empty<byte>();
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
            var normalized = path.Replace('\\', '/');
            return normalized.ToLowerInvariant();
        }
    }
}
