using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWRollback.PM4Module;

/// <summary>
/// Post-processes patched ADTs to ensure all UniqueIDs are globally unique across all tiles.
/// </summary>
public class GlobalUniqueIdFixer
{
    /// <summary>
    /// Reassign all MODF and MDDF UniqueIDs to be globally unique across all ADT files.
    /// </summary>
    public int FixDirectory(string adtDirectory, uint startingId = 1)
    {
        var adtFiles = Directory.GetFiles(adtDirectory, "*.adt")
            .Where(f => !Path.GetFileName(f).Contains("_obj") && !Path.GetFileName(f).Contains("_tex"))
            .OrderBy(f => f)
            .ToList();
        
        Console.WriteLine($"[INFO] Processing {adtFiles.Count} ADT files for UniqueID reassignment...");
        
        uint nextId = startingId;
        int totalModf = 0;
        int totalMddf = 0;
        int filesModified = 0;
        
        foreach (var adtPath in adtFiles)
        {
            try
            {
                var bytes = File.ReadAllBytes(adtPath);
                bool modified = false;
                
                // Find and fix MODF chunk
                int modfModified = FixChunkUniqueIds(bytes, "FDOM", 64, 4, ref nextId, ref modified);
                totalModf += modfModified;
                
                // Find and fix MDDF chunk  
                int mddfModified = FixChunkUniqueIds(bytes, "FDDM", 36, 4, ref nextId, ref modified);
                totalMddf += mddfModified;
                
                if (modified)
                {
                    File.WriteAllBytes(adtPath, bytes);
                    filesModified++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to process {Path.GetFileName(adtPath)}: {ex.Message}");
            }
        }
        
        Console.WriteLine($"[INFO] Reassigned {totalModf} MODF + {totalMddf} MDDF entries across {filesModified} files");
        Console.WriteLine($"[INFO] UniqueID range: {startingId} - {nextId - 1}");
        
        return totalModf + totalMddf;
    }
    
    private int FixChunkUniqueIds(byte[] bytes, string reversedSig, int entrySize, int uidOffset, ref uint nextId, ref bool modified)
    {
        var str = Encoding.ASCII.GetString(bytes);
        int chunkIdx = str.IndexOf(reversedSig);
        
        if (chunkIdx < 0)
            return 0;
        
        int chunkSize = BitConverter.ToInt32(bytes, chunkIdx + 4);
        int entryCount = chunkSize / entrySize;
        int entriesStart = chunkIdx + 8;
        
        for (int i = 0; i < entryCount; i++)
        {
            int entryOffset = entriesStart + i * entrySize + uidOffset;
            if (entryOffset + 4 > bytes.Length)
                break;
            
            // Write new UniqueId
            BitConverter.GetBytes(nextId).CopyTo(bytes, entryOffset);
            nextId++;
            modified = true;
        }
        
        return entryCount;
    }
}
