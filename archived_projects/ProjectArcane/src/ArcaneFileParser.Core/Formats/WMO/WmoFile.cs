using System.IO;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.WMO;

/// <summary>
/// Handles parsing and writing of WMO (World Map Object) files
/// </summary>
public class WmoFile
{
    private readonly ChunkParser _parser;

    public WmoFile()
    {
        _parser = new ChunkParser();
        RegisterChunkHandlers();
    }

    private void RegisterChunkHandlers()
    {
        // Register all WMO chunk handlers
        // TODO: Register WMO chunks as they are implemented
    }

    public void Parse(string filePath)
    {
        foreach (var chunk in _parser.ReadFile(filePath))
        {
            // Process each chunk based on its type
            // TODO: Handle WMO chunk types
        }
    }

    public string CreateReport(string filePath)
    {
        return _parser.CreateReadableReport(filePath);
    }
} 