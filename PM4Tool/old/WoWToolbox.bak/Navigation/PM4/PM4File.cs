using WoWToolbox.MPQ;

namespace WoWToolbox.Navigation.PM4;

/// <summary>
/// Represents the MSRN chunk containing Mesh Surface Referenced Normals.
/// </summary>
public class MSRNChunk
{
    public List<C3Vectori> Normals { get; } = new();

    public MSRNChunk(BinaryReader reader, int chunkSize)
    {
        if (chunkSize % C3Vectori.Size != 0)
        {
            Console.WriteLine($"Warning: MSRN chunk size {chunkSize} is not a multiple of C3Vectori size ({C3Vectori.Size}).");
            // Decide how to handle: throw, log, or attempt partial read?
            // For now, log and continue reading as many full vectors as possible.
        }

        int count = chunkSize / C3Vectori.Size;
        for (int i = 0; i < count; i++)
        {
            Normals.Add(new C3Vectori(reader));
        }

        // Log if we didn't read the whole chunk due to size mismatch
        int bytesRead = count * C3Vectori.Size;
        if (bytesRead < chunkSize)
        {
            Console.WriteLine($"Warning: MSRN chunk size mismatch. Read {bytesRead} bytes, expected {chunkSize}.");
            reader.BaseStream.Seek(chunkSize - bytesRead, SeekOrigin.Current); // Skip remaining bytes
        }
    }

    public static readonly uint Magic = BlizzChunk.GetMagicInt("MSRN"); // NRSM
}


/// <summary>
/// Represents the PM4 file format.
/// </summary>
public class PM4File
{
    public MDOSChunk MDOS { get; private set; }
    public MDSFChunk MDSF { get; private set; }
    public MSRNChunk MSRN { get; private set; }

    public void Load(string filePath)
    {
        // Implementation of Load method
    }
} 