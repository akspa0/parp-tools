using System.Text.Json;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// VLM ADT Decoder - reconstructs ADT files from VLM JSON output.
/// Enables round-trip: ADT → VLM JSON → ADT.
/// </summary>
public class VlmAdtDecoder
{
    private const float TILE_SIZE = 533.33333f;
    private const float CHUNK_SIZE = TILE_SIZE / 16f;
    private const float UNIT_SIZE = CHUNK_SIZE / 8f;

    /// <summary>
    /// Decode VLM JSON to ADT file.
    /// </summary>
    public async Task<bool> DecodeAsync(string jsonPath, string outputAdtPath, string? basePath = null)
    {
        var json = await File.ReadAllTextAsync(jsonPath);
        var sample = JsonSerializer.Deserialize<VlmTrainingSample>(json);
        
        if (sample?.TerrainData == null)
            return false;

        basePath ??= Path.GetDirectoryName(jsonPath) ?? ".";
        
        return await WriteAdtAsync(sample.TerrainData, outputAdtPath, basePath);
    }

    /// <summary>
    /// Write ADT file from terrain data.
    /// </summary>
    private async Task<bool> WriteAdtAsync(VlmTerrainData data, string outputPath, string basePath)
    {
        using var fs = new FileStream(outputPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // MVER
        WriteChunk(bw, "MVER", writer =>
        {
            writer.Write(18);  // v18 format
        });

        // MHDR - placeholder, will update offsets later
        long mhdrPos = fs.Position;
        WriteChunk(bw, "MHDR", writer =>
        {
            for (int i = 0; i < 16; i++)
                writer.Write(0);  // 64 bytes of zeros
        });

        // MCIN - chunk info (will update after writing MCNKs)
        long mcinPos = fs.Position;
        var mcnkOffsets = new (long offset, int size)[256];
        WriteChunk(bw, "MCIN", writer =>
        {
            for (int i = 0; i < 256; i++)
            {
                writer.Write(0);  // offset
                writer.Write(0);  // size
                writer.Write(0);  // flags
                writer.Write(0);  // asyncID
            }
        });

        // MTEX - textures
        long mtexPos = fs.Position;
        WriteChunk(bw, "MTEX", writer =>
        {
            foreach (var tex in data.Textures)
            {
                var bytes = System.Text.Encoding.ASCII.GetBytes(tex);
                writer.Write(bytes);
                writer.Write((byte)0);  // null terminator
            }
        });

        // MCNKs - one per chunk
        for (int i = 0; i < 256; i++)
        {
            mcnkOffsets[i].offset = fs.Position;
            
            var chunkHeights = data.Heights?.FirstOrDefault(h => h.ChunkIndex == i);
            var chunkLayers = data.ChunkLayers?.FirstOrDefault(l => l.ChunkIndex == i);
            
            WriteMcnk(bw, i, data, chunkHeights, chunkLayers, basePath);
            
            mcnkOffsets[i].size = (int)(fs.Position - mcnkOffsets[i].offset);
        }

        // Update MCIN with actual offsets
        long endPos = fs.Position;
        fs.Position = mcinPos + 8;  // After chunk header
        for (int i = 0; i < 256; i++)
        {
            bw.Write((int)mcnkOffsets[i].offset);
            bw.Write(mcnkOffsets[i].size);
            bw.Write(0);  // flags
            bw.Write(0);  // asyncID
        }

        // Update MHDR offsets
        fs.Position = mhdrPos + 8;  // After chunk header
        bw.Write(0);  // flags
        bw.Write((int)(mcinPos - 20));  // MCIN offset (relative to end of MVER)
        bw.Write((int)(mtexPos - 20));  // MTEX offset

        fs.Position = endPos;
        
        await Task.CompletedTask;
        return true;
    }

    private void WriteMcnk(BinaryWriter bw, int chunkIndex, VlmTerrainData data,
                          VlmChunkHeights? heights, VlmChunkLayers? layers, string basePath)
    {
        int ix = chunkIndex % 16;
        int iy = chunkIndex / 16;

        // Calculate chunk position
        float posX = data.ChunkPositions != null && chunkIndex * 3 + 2 < data.ChunkPositions.Length
            ? data.ChunkPositions[chunkIndex * 3]
            : 0;
        float posY = data.ChunkPositions != null && chunkIndex * 3 + 2 < data.ChunkPositions.Length
            ? data.ChunkPositions[chunkIndex * 3 + 1]
            : 0;
        float posZ = data.ChunkPositions != null && chunkIndex * 3 + 2 < data.ChunkPositions.Length
            ? data.ChunkPositions[chunkIndex * 3 + 2]
            : 0;

        int holes = data.Holes != null && chunkIndex < data.Holes.Length
            ? data.Holes[chunkIndex] : 0;

        // Build MCNK sub-chunks in memory
        using var mcnkMs = new MemoryStream();
        using var mcnkBw = new BinaryWriter(mcnkMs);

        // MCNK header (128 bytes for v18)
        long headerStart = mcnkMs.Position;
        for (int i = 0; i < 32; i++)  // 128 bytes / 4
            mcnkBw.Write(0);

        // MCVT - heights
        long mcvtOffset = mcnkMs.Position - headerStart;
        if (heights?.Heights != null && heights.Heights.Length == 145)
        {
            WriteSubChunk(mcnkBw, "MCVT", w =>
            {
                foreach (var h in heights.Heights)
                    w.Write(h);
            });
        }

        // MCNR - normals (placeholder)
        long mcnrOffset = mcnkMs.Position - headerStart;
        WriteSubChunk(mcnkBw, "MCNR", w =>
        {
            for (int i = 0; i < 145; i++)
            {
                w.Write((sbyte)0);   // X
                w.Write((sbyte)127); // Z (up)
                w.Write((sbyte)0);   // Y
            }
            // Padding to 448 bytes
            for (int i = 0; i < 13; i++)
                w.Write((byte)0);
        });

        // Update MCNK header
        mcnkMs.Position = headerStart;
        mcnkBw.Write(0);           // flags
        mcnkBw.Write(ix);          // ix
        mcnkBw.Write(iy);          // iy
        mcnkBw.Write(0);           // nLayers
        mcnkBw.Write(0);           // nDoodadRefs
        mcnkBw.Write((int)mcvtOffset);   // ofsHeight
        mcnkBw.Write((int)mcnrOffset);   // ofsNormal
        // ... rest of header would go here

        mcnkMs.Position = 104;     // Position offsets in header
        mcnkBw.Write(posZ);
        mcnkBw.Write(posX);
        mcnkBw.Write(posY);

        mcnkMs.Position = 68;      // holes offset in header
        mcnkBw.Write(holes);

        // Write MCNK chunk
        mcnkMs.Position = mcnkMs.Length;
        var mcnkData = mcnkMs.ToArray();
        
        WriteChunkRaw(bw, "MCNK", mcnkData);
    }

    private void WriteChunk(BinaryWriter bw, string tag, Action<BinaryWriter> writeContent)
    {
        using var ms = new MemoryStream();
        using var contentBw = new BinaryWriter(ms);
        writeContent(contentBw);
        var data = ms.ToArray();
        
        WriteChunkRaw(bw, tag, data);
    }

    private void WriteSubChunk(BinaryWriter bw, string tag, Action<BinaryWriter> writeContent)
    {
        WriteChunk(bw, tag, writeContent);
    }

    private void WriteChunkRaw(BinaryWriter bw, string tag, byte[] data)
    {
        // Reverse tag for little-endian
        var tagBytes = System.Text.Encoding.ASCII.GetBytes(tag);
        Array.Reverse(tagBytes);
        bw.Write(tagBytes);
        bw.Write(data.Length);
        bw.Write(data);
    }
}
