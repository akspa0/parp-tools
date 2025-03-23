using System.IO;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.M2;

/// <summary>
/// Handles parsing and writing of M2 (Model) files
/// </summary>
public class M2File
{
    private readonly ChunkParser _parser;

    public M2File()
    {
        _parser = new ChunkParser();
        RegisterChunkHandlers();
    }

    private void RegisterChunkHandlers()
    {
        // Register all M2 chunk handlers
        // TODO: Register M2 chunks as they are implemented
    }

    public void Parse(string filePath)
    {
        foreach (var chunk in _parser.ReadFile(filePath))
        {
            // Process each chunk based on its type
            // TODO: Handle M2 chunk types
        }
    }

    public string CreateReport(string filePath)
    {
        return _parser.CreateReadableReport(filePath);
    }
} 