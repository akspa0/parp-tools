using System;
using System.IO;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of McnrAlpha (see lib/gillijimproject/wowfiles/alpha/McnrAlpha.h)
/// Handles Alpha format MCNR chunks (terrain normals)
/// </summary>
public class McnrAlpha : Chunk
{
    /// <summary>
    /// Default constructor
    /// </summary>
    public McnrAlpha() : base("RNCM", 0, Array.Empty<byte>()) { }

    /// <summary>
    /// Constructs a McnrAlpha from file at the given offset
    /// </summary>
    /// <param name="adtFile">The file stream to read from</param>
    /// <param name="offsetInFile">Offset in the file where the chunk starts</param>
    public McnrAlpha(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }

    /// <summary>
    /// Constructs a McnrAlpha from a byte array at the given offset
    /// </summary>
    /// <param name="wholeFile">The byte array containing the full file</param>
    /// <param name="offsetInFile">Offset in the array where the chunk starts</param>
    public McnrAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    /// <summary>
    /// Constructs a McnrAlpha from provided data
    /// </summary>
    /// <param name="letters">The FourCC code</param>
    /// <param name="givenSize">The size of the chunk</param>
    /// <param name="chunkData">The chunk data</param>
    public McnrAlpha(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    /// <summary>
    /// Converts the Alpha format MCNR to LichKing format by reordering normals
    /// </summary>
    /// <returns>A McnrLk object with the reordered normal data</returns>
    public McnrLk ToMcnrLk()
    {
        /* Note: Alpha normals are NOT interleaved... Which means there are all outer normals 
         * first (all 81), then all inner normals (all 64) in MCNR (and not 9-8-9-8 etc. of each).
         * So here we re-order them for post-alpha format.
         */

        byte[] cMcnrData = new byte[0];
        
        const int outerNormalsSequence = 9 * 3; // 9 normals * 3 bytes per normal
        const int innerNormalsSequence = 8 * 3; // 8 normals * 3 bytes per normal
        
        const int innerDataStart = outerNormalsSequence * 9; // Where inner normals data begins
        
        // Calculate output size (including the 13 unknown bytes padding at the end)
        const int unknownBytes = 13;
        int outputSize = Data.Length;
        
        // Create a new byte array with the correct size
        cMcnrData = new byte[outputSize];
        int destPos = 0;
        
        for (int i = 0; i < 9; i++)
        {
            // Copy outer normals sequence
            Array.Copy(Data, i * outerNormalsSequence, cMcnrData, destPos, outerNormalsSequence);
            destPos += outerNormalsSequence;
            
            if (i == 8)
                break;
                
            // Copy inner normals sequence
            Array.Copy(Data, innerDataStart + (i * innerNormalsSequence), cMcnrData, destPos, innerNormalsSequence);
            destPos += innerNormalsSequence;
        }
        
        // Add unknown bytes (13 bytes of padding/unknown data at the end)
        byte[] unknownData = new byte[unknownBytes];
        Array.Copy(unknownData, 0, cMcnrData, destPos, unknownBytes);
        
        // Create the new LK format MCNR chunk
        McnrLk mcnrLk = new McnrLk("RNCM", GivenSize - unknownBytes, cMcnrData);
        
        return mcnrLk;
    }
}
