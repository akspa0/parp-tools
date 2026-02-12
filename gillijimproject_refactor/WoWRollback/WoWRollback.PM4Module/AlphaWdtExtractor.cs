using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.PM4Module;

/// <summary>
/// Extracts data from Alpha 0.5.3 WDT files (monolithic format with embedded ADT chunks).
/// Ported from WoWRollback.LkToAlphaModule.AlphaWdtReader.
/// </summary>
public sealed class AlphaWdtExtractor
{
    private readonly byte[] _bytes;
    private readonly string _path;

    public AlphaWdtExtractor(string path)
    {
        _path = path;
        _bytes = File.ReadAllBytes(path);
    }
    
    public AlphaWdtExtractor(byte[] bytes, string path = "")
    {
        _bytes = bytes;
        _path = path;
    }

    /// <summary>
    /// Read an Alpha WDT file.
    /// </summary>
    public static AlphaWdtData Read(string path)
    {
        var extractor = new AlphaWdtExtractor(path);
        return extractor.Extract();
    }
    
    /// <summary>
    /// Extract data from the WDT.
    /// </summary>
    public AlphaWdtData Extract()
    {
        var result = new AlphaWdtData { Path = _path };
        
        // Top-level chunk scan
        var topChunks = ScanChunks(_bytes, 0, _bytes.Length);
        
        // Read MPHD (header with doodad/mapobj name offsets)
        if (topChunks.TryGetValue("MPHD", out var mphd))
        {
            int mphdData = mphd.offset + 8;
            if (mphdData + 16 <= _bytes.Length)
            {
                result.NDoodadNames = BitConverter.ToInt32(_bytes, mphdData + 0);
                result.OffsDoodadNames = BitConverter.ToInt32(_bytes, mphdData + 4);
                result.NMapObjNames = BitConverter.ToInt32(_bytes, mphdData + 8);
                result.OffsMapObjNames = BitConverter.ToInt32(_bytes, mphdData + 12);
            }
        }
        
        // MAIN grid (64x64 tile pointers)
        if (!topChunks.TryGetValue("MAIN", out var main))
            return result;
            
        int mainData = main.offset + 8;
        for (int i = 0; i < 4096; i++)
        {
            int pos = mainData + i * 16;
            if (pos + 16 > _bytes.Length) break;
            
            int mhdrOffset = BitConverter.ToInt32(_bytes, pos);
            if (mhdrOffset <= 0) continue;
            
            int tileX = i % 64;
            int tileY = i / 64;
            
            var tile = new AlphaTileData
            {
                Index = i,
                TileX = tileX,
                TileY = tileY,
                MhdrOffset = mhdrOffset
            };
            
            // Read MHDR at offset
            int mhdrData = mhdrOffset + 8;
            if (mhdrData + 28 <= _bytes.Length)
            {
                tile.OffsInfo = BitConverter.ToInt32(_bytes, mhdrData + 0);
                tile.OffsTex = BitConverter.ToInt32(_bytes, mhdrData + 4);
                tile.SizeTex = BitConverter.ToInt32(_bytes, mhdrData + 8);
                tile.OffsDoo = BitConverter.ToInt32(_bytes, mhdrData + 12);
                tile.SizeDoo = BitConverter.ToInt32(_bytes, mhdrData + 16);
                tile.OffsMob = BitConverter.ToInt32(_bytes, mhdrData + 20);
                tile.SizeMob = BitConverter.ToInt32(_bytes, mhdrData + 24);
            }
            
            // Calculate first MCNK position
            int mcInEnd = mhdrData + tile.OffsInfo + 8 + 4096;
            int mtExEnd = mhdrData + tile.OffsTex + 8 + tile.SizeTex;
            int mdDfEnd = mhdrData + tile.OffsDoo + 8 + tile.SizeDoo;
            int moDfEnd = mhdrData + tile.OffsMob + 8 + tile.SizeMob;
            int firstMcnk = Math.Max(Math.Max(mcInEnd, mtExEnd), Math.Max(mdDfEnd, moDfEnd));
            
            if (firstMcnk + 8 + 128 <= _bytes.Length)
            {
                tile.FirstMcnkOffset = firstMcnk;
                
                // Read MCNK header info at firstMcnk + 8
                int mcnkData = firstMcnk + 8;
                if (mcnkData + 128 <= _bytes.Length)
                {
                    tile.Flags = BitConverter.ToInt32(_bytes, mcnkData + 0);
                    tile.ChunkX = BitConverter.ToInt32(_bytes, mcnkData + 4);
                    tile.ChunkY = BitConverter.ToInt32(_bytes, mcnkData + 8);
                    tile.NLayers = BitConverter.ToInt32(_bytes, mcnkData + 16);
                    tile.AreaId = BitConverter.ToInt32(_bytes, mcnkData + 56);
                }
            }
            
            result.Tiles.Add(tile);
        }
        
        return result;
    }
    
    /// <summary>
    /// Extract terrain data for a specific tile embedded in the WDT.
    /// </summary>
    public TileTerrainData? ExtractTileTerrain(int tileIndex)
    {
        var wdtData = Extract();
        var tile = wdtData.Tiles.Find(t => t.Index == tileIndex);
        if (tile == null || tile.FirstMcnkOffset <= 0)
            return null;
            
        // For Alpha format, we need to navigate to MCNK chunks
        // This requires parsing the embedded ADT structure
        var result = new TileTerrainData
        {
            Map = Path.GetFileNameWithoutExtension(_path),
            TileX = tile.TileX,
            TileY = tile.TileY,
            Textures = new List<string>(),
            Chunks = new List<ChunkTerrainData>()
        };
        
        // TODO: Full terrain extraction from embedded MCNKs
        // For now, return basic structure - can be enhanced later
        
        return result;
    }
    
    private Dictionary<string, (int offset, int size)> ScanChunks(byte[] data, int start, int maxLen)
    {
        var dict = new Dictionary<string, (int offset, int size)>(StringComparer.Ordinal);
        int pos = start;
        int end = Math.Min(start + maxLen, data.Length);
        
        while (pos + 8 <= end)
        {
            // Read FourCC as on-disk (reversed)
            string sig = Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);
            
            if (size < 0 || pos + 8 + size > end) break;
            
            // Reverse to get readable name
            string readable = new string(sig.Reverse().ToArray());
            dict[readable] = (pos, size);
            
            int next = pos + 8 + size;
            if ((next & 1) == 1) next++; // Align
            if (next <= pos) break;
            pos = next;
        }
        
        return dict;
    }
}

#region Alpha WDT Data Models

/// <summary>Data extracted from an Alpha WDT file.</summary>
public class AlphaWdtData
{
    public string Path { get; set; } = "";
    public int NDoodadNames { get; set; }
    public int OffsDoodadNames { get; set; }
    public int NMapObjNames { get; set; }
    public int OffsMapObjNames { get; set; }
    public List<AlphaTileData> Tiles { get; set; } = new();
}

/// <summary>Data for a single tile in an Alpha WDT.</summary>
public class AlphaTileData
{
    public int Index { get; set; }
    public int TileX { get; set; }
    public int TileY { get; set; }
    public int MhdrOffset { get; set; }
    
    // MHDR fields
    public int OffsInfo { get; set; }
    public int OffsTex { get; set; }
    public int SizeTex { get; set; }
    public int OffsDoo { get; set; }
    public int SizeDoo { get; set; }
    public int OffsMob { get; set; }
    public int SizeMob { get; set; }
    
    // First MCNK info
    public int FirstMcnkOffset { get; set; }
    public int Flags { get; set; }
    public int ChunkX { get; set; }
    public int ChunkY { get; set; }
    public int NLayers { get; set; }
    public int AreaId { get; set; }
}

#endregion
