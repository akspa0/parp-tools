using System.IO;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.WDT;

/// <summary>
/// Handles parsing and writing of WDT (World Definition Table) files
/// </summary>
public class WdtFile
{
    private readonly ChunkParser _parser;

    public WdtFile()
    {
        _parser = new ChunkParser();
        RegisterChunkHandlers();
    }

    private void RegisterChunkHandlers()
    {
        // Register all WDT chunk handlers
        // TODO: Register WDT chunks as they are implemented
    }

    public void Parse(string filePath)
    {
        foreach (var chunk in _parser.ReadFile(filePath))
        {
            // Process each chunk based on its type
            // TODO: Handle WDT chunk types
        }
    }

    public string CreateReport(string filePath)
    {
        return _parser.CreateReadableReport(filePath);
    }
} 