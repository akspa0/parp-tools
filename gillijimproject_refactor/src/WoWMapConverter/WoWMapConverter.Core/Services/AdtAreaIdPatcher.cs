namespace WoWMapConverter.Core.Services;

/// <summary>
/// Post-processes LK ADT files to patch AreaID values in MCNK headers.
/// Used when AreaID crosswalk mapping is available.
/// </summary>
public static class AdtAreaIdPatcher
{
    private const int McnkHeaderSize = 0x80;
    private const int AreaIdOffsetInHeader = 0x34; // Offset of AreaId field in MCNK header
    
    /// <summary>
    /// Patch all MCNK AreaIDs in an LK ADT file using the provided mapping function.
    /// </summary>
    /// <param name="adtPath">Path to the LK ADT file</param>
    /// <param name="mapName">Map name for crosswalk lookup</param>
    /// <param name="areaIdMapper">Function that maps (mapName, alphaAreaId) -> lkAreaId</param>
    /// <returns>Number of chunks patched</returns>
    public static int PatchAreaIds(string adtPath, string mapName, Func<string, int, int> areaIdMapper)
    {
        if (!File.Exists(adtPath))
            return 0;

        var data = File.ReadAllBytes(adtPath);
        int patched = 0;

        // Find all MCNK chunks and patch their AreaID fields
        int offset = 0;
        while (offset < data.Length - 8)
        {
            // Look for MCNK magic (reversed: "KNCM")
            if (data[offset] == 'K' && data[offset + 1] == 'N' && 
                data[offset + 2] == 'C' && data[offset + 3] == 'M')
            {
                int chunkSize = BitConverter.ToInt32(data, offset + 4);
                int headerStart = offset + 8; // After FourCC + size
                
                if (headerStart + McnkHeaderSize <= data.Length)
                {
                    // Read current AreaID (at offset 0x34 in header)
                    int areaIdOffset = headerStart + AreaIdOffsetInHeader;
                    int currentAreaId = BitConverter.ToInt32(data, areaIdOffset);
                    
                    // Map to LK AreaID
                    int mappedAreaId = areaIdMapper(mapName, currentAreaId);
                    
                    if (mappedAreaId != 0 && mappedAreaId != currentAreaId)
                    {
                        // Write new AreaID
                        var newAreaIdBytes = BitConverter.GetBytes(mappedAreaId);
                        Array.Copy(newAreaIdBytes, 0, data, areaIdOffset, 4);
                        patched++;
                    }
                }
                
                // Move to next chunk
                offset += 8 + chunkSize;
            }
            else
            {
                offset++;
            }
        }

        if (patched > 0)
        {
            File.WriteAllBytes(adtPath, data);
        }

        return patched;
    }

    /// <summary>
    /// Patch AreaIDs in all ADT files in a directory.
    /// </summary>
    public static (int filesPatched, int chunksPatched) PatchDirectory(
        string directory, 
        string mapName, 
        Func<string, int, int> areaIdMapper,
        bool verbose = false)
    {
        int filesPatched = 0;
        int totalChunks = 0;

        foreach (var adtFile in Directory.GetFiles(directory, "*.adt"))
        {
            int chunks = PatchAreaIds(adtFile, mapName, areaIdMapper);
            if (chunks > 0)
            {
                filesPatched++;
                totalChunks += chunks;
                if (verbose)
                    Console.WriteLine($"  Patched {chunks} chunks in {Path.GetFileName(adtFile)}");
            }
        }

        return (filesPatched, totalChunks);
    }
}
