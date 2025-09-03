using System;using System.IO;
using System.Collections.Generic;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port skeleton of Mcrf (see lib/gillijimproject/wowfiles/Mcrf.h)
/// </summary>
public class Mcrf : Chunk
{
    public Mcrf(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Mcrf(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Mcrf(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }
    
    /// <summary>
    /// [PORT] Default parameterless constructor
    /// </summary>
    public Mcrf() : base("MCRF", 0, Array.Empty<byte>()) { }
    
    /// <summary>
    /// [PORT] Gets indices for doodads from the MCRF chunk
    /// </summary>
    /// <param name="doodadsNumber">Number of doodad indices to extract</param>
    /// <returns>List of doodad indices</returns>
    public List<int> GetDoodadsIndices(int doodadsNumber)
    {
        List<int> mcrfIndices = new List<int>(Data.Length / 4);
        for (int i = 0; i < Data.Length; i += 4)
        {
            mcrfIndices.Add(BitConverter.ToInt32(Data, i));
        }
        
        return mcrfIndices.GetRange(0, doodadsNumber);
    }
    
    /// <summary>
    /// [PORT] Gets indices for WMOs from the MCRF chunk
    /// </summary>
    /// <param name="wmosNumber">Number of WMO indices to extract</param>
    /// <returns>List of WMO indices</returns>
    public List<int> GetWmosIndices(int wmosNumber)
    {
        List<int> mcrfIndices = new List<int>(Data.Length / 4);
        for (int i = 0; i < Data.Length; i += 4)
        {
            mcrfIndices.Add(BitConverter.ToInt32(Data, i));
        }
        
        return mcrfIndices.GetRange(mcrfIndices.Count - wmosNumber, wmosNumber);
    }
    
    /// <summary>
    /// [PORT] Updates indices for Lich King format
    /// </summary>
    /// <param name="alphaM2Indices">Mapping of Alpha M2 indices to LK M2 indices.</param>
    /// <param name="m2Count">Number of M2s in the parent MCNK.</param>
    /// <param name="alphaWmoIndices">Mapping of Alpha WMO indices to LK WMO indices.</param>
    /// <param name="wmoCount">Number of WMOs in the parent MCNK.</param>
    /// <returns>A new Mcrf object with updated indices for LK.</returns>
    public Mcrf UpdateIndicesForLk(Dictionary<int, int> alphaM2Indices, int m2Count, Dictionary<int, int> alphaWmoIndices, int wmoCount)
    {
        // [PORT] Remap indices by parsing MCRF payload; first M2 refs, then WMO refs.
        var oldDoodadRefs = GetDoodadsIndices(m2Count);
        var oldWmoRefs = GetWmosIndices(wmoCount);

        var newMcrfData = new List<byte>((oldDoodadRefs.Count + oldWmoRefs.Count) * 4);

        // Add M2 indices (doodads)
        foreach (var old in oldDoodadRefs)
        {
            int mapped = alphaM2Indices.TryGetValue(old, out var v) ? v : old;
            newMcrfData.AddRange(BitConverter.GetBytes(mapped));
        }

        // Add WMO indices
        foreach (var old in oldWmoRefs)
        {
            int mapped = alphaWmoIndices.TryGetValue(old, out var v) ? v : old;
            newMcrfData.AddRange(BitConverter.GetBytes(mapped));
        }

        // [PORT] Data is immutable in Chunk; return a new Mcrf with remapped payload
        return new Mcrf("MCRF", newMcrfData.Count, newMcrfData.ToArray());
    }
}
