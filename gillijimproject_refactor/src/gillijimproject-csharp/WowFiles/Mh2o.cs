using System;
using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] C# port of Mh2o (see lib/gillijimproject/wowfiles/Mh2o.h)
/// Handles water information for ADT files
/// </summary>
public class Mh2o : Chunk
{
    /// <summary>
    /// Default constructor
    /// </summary>
    public Mh2o() : base("MH2O", 0, Array.Empty<byte>()) { }

    /// <summary>
    /// Constructs an Mh2o from a file stream at the given offset
    /// </summary>
    /// <param name="file">The file stream to read from</param>
    /// <param name="offsetInFile">Offset in the file where the chunk starts</param>
    public Mh2o(FileStream file, int offsetInFile) : base(file, offsetInFile) { }

    /// <summary>
    /// Constructs an Mh2o from a byte array at the given offset
    /// </summary>
    /// <param name="wholeFile">The byte array containing the full file</param>
    /// <param name="offsetInFile">Offset in the array where the chunk starts</param>
    public Mh2o(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) { }

    /// <summary>
    /// Constructs an Mh2o from provided data
    /// </summary>
    /// <param name="letters">The FourCC code</param>
    /// <param name="givenSize">The size of the chunk</param>
    /// <param name="chunkData">The chunk data</param>
    public Mh2o(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) { }

    /// <summary>
    /// Checks if the chunk is empty (has no data)
    /// </summary>
    /// <returns>True if the chunk is empty, false otherwise</returns>
    public new bool IsEmpty()
    {
        return GetRealSize() == 0;
    }
    
    /// <summary>
    /// [PORT] C# equivalent of Mh2o::toFile method
    /// Writes the chunk data to a file
    /// </summary>
    /// <param name="fileName">The path to the output file</param>
    public void ToFile(string fileName)
    {
        // [PORT] Original C++ used std::ofstream, C# uses FileStream
        using (FileStream outputFile = new FileStream(fileName, FileMode.Create, FileAccess.Write))
        {
            if (!IsEmpty())
            {
                byte[] wholeChunk = GetWholeChunk();
                outputFile.Write(wholeChunk, 0, wholeChunk.Length);
            }
        }
    }
}
