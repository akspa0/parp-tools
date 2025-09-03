using System;
using System.IO;
using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of McvtAlpha (see lib/gillijimproject/wowfiles/alpha/McvtAlpha.h)
/// Handles Alpha format MCVT chunks (terrain vertices)
/// </summary>
public class McvtAlpha : Chunk
{
    /// <summary>
    /// Default constructor
    /// </summary>
    public McvtAlpha() : base("MCVT", 0, Array.Empty<byte>()) { }

    /// <summary>
    /// Constructs a McvtAlpha from file at the given offset
    /// </summary>
    /// <param name="adtFile">The file stream to read from</param>
    /// <param name="offsetInFile">Offset in the file where the chunk starts</param>
    public McvtAlpha(FileStream adtFile, int offsetInFile) : base(adtFile, offsetInFile) { }

    /// <summary>
    /// Constructs a McvtAlpha from a byte array at the given offset
    /// </summary>
    /// <param name="wholeFile">The byte array containing the full file</param>
    /// <param name="offsetInFile">Offset in the array where the chunk starts</param>
    public McvtAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    /// <summary>
    /// Constructs a McvtAlpha from provided data
    /// </summary>
    /// <param name="letters">The FourCC code</param>
    /// <param name="givenSize">The size of the chunk</param>
    /// <param name="chunkData">The chunk data</param>
    public McvtAlpha(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    /// <summary>
    /// Converts the Alpha format MCVT to LichKing format by reordering vertices
    /// </summary>
    /// <returns>A Mcvt containing the reordered vertex data in LK format</returns>
    public Mcvt ToMcvt()
    {
        /* Note: Alpha vertices are NOT interleaved... Which means there are all outer vertices 
         * first (all 81), then all inner vertices (all 64) in MCVT (and not 9-8-9-8 etc. of each).
         * So here we re-order them for post-alpha format.
         */

        byte[] cMcvtData = new byte[0];
        
        const int outerVerticesSequence = 9 * 4; // 9 vertices * 4 bytes per float
        const int innerVerticesSequence = 8 * 4; // 8 vertices * 4 bytes per float
        
        const int innerDataStart = outerVerticesSequence * 9; // Where inner vertices data begins
        
        // Create a new byte array with the correct size
        cMcvtData = new byte[Data.Length];
        int destPos = 0;
        
        for (int i = 0; i < 9; i++)
        {
            // Copy outer vertices sequence
            Array.Copy(Data, i * outerVerticesSequence, cMcvtData, destPos, outerVerticesSequence);
            destPos += outerVerticesSequence;
            
            if (i == 8)
                break;
                
            // Copy inner vertices sequence
            Array.Copy(Data, innerDataStart + (i * innerVerticesSequence), cMcvtData, destPos, innerVerticesSequence);
            destPos += innerVerticesSequence;
        }
        
        // Create the new LK format MCVT chunk
        var mcvtLk = new Mcvt("MCVT", cMcvtData.Length, cMcvtData);

        return mcvtLk;
    }
}
