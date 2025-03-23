using System;
using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats;

/// <summary>
/// Base class for all WoW file format handlers.
/// </summary>
public abstract class FileFormatBase
{
    /// <summary>
    /// Gets the list of chunks in this file.
    /// </summary>
    protected List<IChunk> Chunks { get; } = new();

    /// <summary>
    /// Gets whether the file was successfully parsed.
    /// </summary>
    public bool IsValid { get; protected set; }

    /// <summary>
    /// Gets the file format version.
    /// </summary>
    public uint Version { get; protected set; }

    /// <summary>
    /// Gets the file path that was parsed.
    /// </summary>
    public string FilePath { get; }

    protected FileFormatBase(string filePath)
    {
        FilePath = filePath;
        IsValid = false;
    }

    /// <summary>
    /// Parses the file format from the given file path.
    /// </summary>
    public virtual void Parse()
    {
        using var stream = File.OpenRead(FilePath);
        using var reader = new BinaryReader(stream);
        ParseInternal(reader);
    }

    /// <summary>
    /// Internal method to parse the file format from a binary reader.
    /// </summary>
    protected abstract void ParseInternal(BinaryReader reader);

    /// <summary>
    /// Gets a chunk of the specified type.
    /// </summary>
    protected T? GetChunk<T>() where T : class, IChunk
    {
        foreach (var chunk in Chunks)
        {
            if (chunk is T typedChunk && chunk.IsValid)
                return typedChunk;
        }
        return null;
    }

    /// <summary>
    /// Gets all chunks of the specified type.
    /// </summary>
    protected IEnumerable<T> GetChunks<T>() where T : class, IChunk
    {
        foreach (var chunk in Chunks)
        {
            if (chunk is T typedChunk && chunk.IsValid)
                yield return typedChunk;
        }
    }

    /// <summary>
    /// Creates a string representation of the file format for debugging.
    /// </summary>
    public override string ToString()
    {
        return $"{GetType().Name} [Path: {FilePath}, Valid: {IsValid}, Version: {Version}, Chunks: {Chunks.Count}]";
    }

    /// <summary>
    /// Gets a detailed report of the file format and its chunks.
    /// </summary>
    public virtual string GetReport()
    {
        var report = new System.Text.StringBuilder();
        report.AppendLine($"File Format: {GetType().Name}");
        report.AppendLine($"File Path: {FilePath}");
        report.AppendLine($"Valid: {IsValid}");
        report.AppendLine($"Version: {Version}");
        report.AppendLine($"Chunk Count: {Chunks.Count}");
        report.AppendLine();

        report.AppendLine("Chunks:");
        foreach (var chunk in Chunks)
        {
            report.AppendLine($"  {chunk}");
        }

        return report.ToString();
    }
} 