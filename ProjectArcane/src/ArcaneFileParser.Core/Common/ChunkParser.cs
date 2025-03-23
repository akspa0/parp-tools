using System;
using System.Collections.Generic;
using System.IO;

namespace ArcaneFileParser.Core.Common;

/// <summary>
/// Generic chunk parser that can handle any chunk-based file format
/// </summary>
public class ChunkParser
{
    private readonly Dictionary<string, Type> _chunkHandlers;

    public ChunkParser()
    {
        _chunkHandlers = new Dictionary<string, Type>();
    }

    /// <summary>
    /// Registers a chunk handler type for a specific chunk ID (in documentation format)
    /// </summary>
    public void RegisterHandler<T>(string chunkId) where T : IChunk, new()
    {
        if (string.IsNullOrEmpty(chunkId) || chunkId.Length != 4)
        {
            throw new ArgumentException("Chunk ID must be exactly 4 characters", nameof(chunkId));
        }

        _chunkHandlers[chunkId] = typeof(T);
    }

    /// <summary>
    /// Reads the next chunk from the stream
    /// </summary>
    public IChunk ReadNextChunk(BinaryReader reader)
    {
        // Read chunk header
        var header = ChunkHeader.Read(reader);

        // Find handler for this chunk type
        if (!_chunkHandlers.TryGetValue(header.Id, out Type? handlerType))
        {
            throw new InvalidOperationException($"No handler registered for chunk type: {header.Id}");
        }

        // Create and initialize handler
        var handler = (IChunk)Activator.CreateInstance(handlerType)!;
        handler.Parse(reader, header.Size);

        return handler;
    }

    /// <summary>
    /// Reads all chunks from a stream until the end
    /// </summary>
    public IEnumerable<IChunk> ReadAllChunks(BinaryReader reader)
    {
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            yield return ReadNextChunk(reader);
        }
    }

    /// <summary>
    /// Reads all chunks from a file
    /// </summary>
    public IEnumerable<IChunk> ReadFile(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);
        
        foreach (var chunk in ReadAllChunks(reader))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Creates a human-readable report of all chunks in a file
    /// </summary>
    public string CreateReadableReport(string filePath)
    {
        using var writer = new StringWriter();
        
        writer.WriteLine($"File: {Path.GetFileName(filePath)}");
        writer.WriteLine("Chunks:");
        writer.WriteLine();

        foreach (var chunk in ReadFile(filePath))
        {
            writer.WriteLine($"=== {chunk.ChunkId} ===");
            writer.WriteLine(chunk.ToHumanReadable());
            writer.WriteLine();
        }

        return writer.ToString();
    }
} 