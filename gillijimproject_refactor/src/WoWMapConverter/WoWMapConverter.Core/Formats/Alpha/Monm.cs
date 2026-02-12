using System.Text;
using WoWMapConverter.Core.Formats.Shared;
using WoWMapConverter.Core.Utilities;

namespace WoWMapConverter.Core.Formats.Alpha;

/// <summary>
/// Alpha WDT MONM chunk - WMO name table.
/// </summary>
public class Monm : Chunk
{
    public Monm() : base("MONM", 0, Array.Empty<byte>()) { }
    public Monm(FileStream file, int offsetInFile) : base(file, offsetInFile) { }
    public Monm(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }
    public Monm(string letters, int givenSize, byte[] data) : base("MONM", givenSize, data) { }

    /// <summary>
    /// Parse the NUL-separated filename table.
    /// </summary>
    public List<string> GetFileNames() => FileUtils.GetFileNames(Data);

    /// <summary>
    /// Convert to LK MWMO chunk.
    /// </summary>
    public Chunk ToMwmo() => new Chunk("MWMO", GivenSize, Data);
    
    /// <summary>
    /// Convert to LK MWMO chunk with path remapping.
    /// Substitutes WMO paths according to the mapping dictionary.
    /// </summary>
    public Chunk ToMwmoWithRemapping(Dictionary<string, string> pathMapping)
    {
        if (pathMapping == null || pathMapping.Count == 0)
            return ToMwmo();
        
        var originalNames = GetFileNames();
        var remappedNames = new List<string>();
        
        foreach (var name in originalNames)
        {
            // Normalize the path for lookup
            var normalized = name.Replace('/', '\\').TrimStart('\\');
            
            // Check if we have a remapping for this WMO
            if (pathMapping.TryGetValue(name, out var newPath) ||
                pathMapping.TryGetValue(normalized, out newPath))
            {
                remappedNames.Add(newPath);
            }
            else
            {
                remappedNames.Add(name);
            }
        }
        
        // Rebuild the NUL-separated string table
        var result = new List<byte>();
        foreach (var name in remappedNames)
        {
            result.AddRange(Encoding.ASCII.GetBytes(name));
            result.Add(0); // NUL terminator
        }
        
        return new Chunk("MWMO", result.Count, result.ToArray());
    }
}
