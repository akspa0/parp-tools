using System;
using System.IO;
using System.Text;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using Warcraft.NET.Files.ADT.Chunks;
using Warcraft.NET.Files.ADT.Terrain.MCNK;
using Warcraft.NET.Files.ADT.Terrain.MCNK.SubChunks;
using Warcraft.NET.Files.ADT.TerrainTexture.MapChunk.SubChunks;
using Warcraft.NET.Files.ADT.TerrainTexture.MapChunk.Entries;
using WotlkMCNK = Warcraft.NET.Files.ADT.Terrain.Wotlk.MCNK;

namespace WoWRollback.PM4Module;

/// <summary>
/// Merges split ADT files (root + _obj0 + _tex0) into monolithic 3.3.5 ADT
/// using Warcraft.NET library for proper parsing and serialization.
/// </summary>
public sealed class SplitAdtMerger
{
    /// <summary>
    /// Result of a merge operation.
    /// </summary>
    public class MergeResult
    {
        public bool Success { get; set; }
        public string? Error { get; set; }
        public byte[]? Data { get; set; }
        public int RootMcnkCount { get; set; }
        public int Tex0McnkCount { get; set; }
        public int Obj0McnkCount { get; set; }
        public int TextureCount { get; set; }
        public int ModelCount { get; set; }
        public int WmoCount { get; set; }
    }

    /// <summary>
    /// Merge split ADT files into a monolithic 3.3.5 ADT.
    /// </summary>
    /// <param name="rootPath">Path to root ADT file (e.g., development_1_1.adt)</param>
    /// <param name="obj0Path">Optional path to _obj0.adt file</param>
    /// <param name="tex0Path">Optional path to _tex0.adt file</param>
    /// <returns>MergeResult containing the merged ADT bytes or error</returns>
    public MergeResult Merge(string rootPath, string? obj0Path = null, string? tex0Path = null)
    {
        var result = new MergeResult();

        try
        {
            // Load root ADT
            if (!File.Exists(rootPath))
            {
                result.Error = $"Root ADT not found: {rootPath}";
                return result;
            }

            var rootBytes = File.ReadAllBytes(rootPath);
            
            // Check if this is Cataclysm split format (no MCIN chunk)
            bool isCataSplit = !HasChunk(rootBytes, "MCIN");
            
            Terrain wotlkAdt;
            
            if (isCataSplit)
            {
                Console.WriteLine("[INFO] Detected Cataclysm split format - building WotLK structure");
                wotlkAdt = BuildWotlkFromCataSplit(rootBytes, obj0Path, tex0Path, result);
            }
            else
            {
                // Standard WotLK monolithic format
                try
                {
                    wotlkAdt = new Terrain(rootBytes);
                }
                catch (Exception ex)
                {
                    result.Error = $"Failed to parse root ADT: {ex.Message}";
                    return result;
                }
                
                // Merge split files if available
                MergeSplitFiles(wotlkAdt, obj0Path, tex0Path, result);
            }

            result.RootMcnkCount = wotlkAdt.Chunks?.Length ?? 0;
            result.TextureCount = wotlkAdt.Textures?.Filenames?.Count ?? 0;
            result.ModelCount = wotlkAdt.Models?.Filenames?.Count ?? 0;
            result.WmoCount = wotlkAdt.WorldModelObjects?.Filenames?.Count ?? 0;

            Console.WriteLine($"[INFO] Final ADT: {result.RootMcnkCount} MCNKs, {result.TextureCount} textures, {result.ModelCount} models, {result.WmoCount} WMOs");

            // Serialize the merged ADT
            result.Data = wotlkAdt.Serialize();
            result.Success = true;

            Console.WriteLine($"[INFO] Merged ADT size: {result.Data.Length} bytes");

            return result;
        }
        catch (Exception ex)
        {
            result.Error = $"Merge failed: {ex.Message}\n{ex.StackTrace}";
            return result;
        }
    }

    /// <summary>
    /// Check if a chunk exists in the ADT data.
    /// </summary>
    private bool HasChunk(byte[] data, string chunkName)
    {
        string reversed = new string(chunkName.Reverse().ToArray());
        byte[] pattern = Encoding.ASCII.GetBytes(reversed);
        
        for (int i = 0; i <= data.Length - 4; i++)
        {
            if (data[i] == pattern[0] && data[i+1] == pattern[1] && 
                data[i+2] == pattern[2] && data[i+3] == pattern[3])
                return true;
        }
        return false;
    }

    /// <summary>
    /// Build a WotLK Terrain from Cataclysm split format files.
    /// </summary>
    private Terrain BuildWotlkFromCataSplit(byte[] rootBytes, string? obj0Path, string? tex0Path, MergeResult result)
    {
        var wotlkAdt = new Terrain();
        
        // Parse root file chunks manually
        var rootChunks = ParseChunks(rootBytes);
        
        // Set version
        wotlkAdt.Version = new MVER(18);
        
        // Parse MHDR
        if (rootChunks.TryGetValue("MHDR", out var mhdrData))
        {
            wotlkAdt.Header = new MHDR(mhdrData);
        }
        else
        {
            wotlkAdt.Header = new MHDR();
        }
        
        // Parse MH2O (water)
        if (rootChunks.TryGetValue("MH2O", out var mh2oData))
        {
            wotlkAdt.Water = new MH2O(mh2oData);
        }
        
        // Parse MCNKs from root (terrain data)
        var mcnkList = new List<WotlkMCNK>();
        foreach (var chunk in rootChunks.Where(c => c.Key == "MCNK"))
        {
            // Cata MCNKs have same structure as WotLK for terrain
            var mcnk = new WotlkMCNK(chunk.Value);
            mcnkList.Add(mcnk);
        }
        
        // Actually, we need to collect all MCNKs properly
        mcnkList.Clear();
        int pos = 0;
        while (pos < rootBytes.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(rootBytes, pos, 4);
            int size = BitConverter.ToInt32(rootBytes, pos + 4);
            
            if (size < 0 || pos + 8 + size > rootBytes.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            if (readable == "MCNK")
            {
                var chunkData = new byte[size];
                Buffer.BlockCopy(rootBytes, pos + 8, chunkData, 0, size);
                var mcnk = new WotlkMCNK(chunkData);
                mcnkList.Add(mcnk);
            }
            
            pos += 8 + size;
        }
        
        Console.WriteLine($"[INFO] Parsed {mcnkList.Count} MCNKs from root");
        result.RootMcnkCount = mcnkList.Count;
        
        // Load tex0 for texture data
        if (!string.IsNullOrEmpty(tex0Path) && File.Exists(tex0Path))
        {
            try
            {
                var tex0Bytes = File.ReadAllBytes(tex0Path);
                var tex0Chunks = ParseChunks(tex0Bytes);
                
                // Get MTEX
                if (tex0Chunks.TryGetValue("MTEX", out var mtexData))
                {
                    wotlkAdt.Textures = new MTEX(mtexData);
                    Console.WriteLine($"[INFO] Loaded MTEX: {wotlkAdt.Textures.Filenames.Count} textures");
                }
                
                // Parse tex0 MCNKs for MCLY/MCAL/MCSH
                var tex0Mcnks = new List<Warcraft.NET.Files.ADT.TerrainTexture.MCNK>();
                pos = 0;
                while (pos < tex0Bytes.Length - 8)
                {
                    string sig = Encoding.ASCII.GetString(tex0Bytes, pos, 4);
                    int size = BitConverter.ToInt32(tex0Bytes, pos + 4);
                    
                    if (size < 0 || pos + 8 + size > tex0Bytes.Length) break;
                    
                    string readable = new string(sig.Reverse().ToArray());
                    
                    if (readable == "MCNK")
                    {
                        var chunkData = new byte[size];
                        Buffer.BlockCopy(tex0Bytes, pos + 8, chunkData, 0, size);
                        var mcnk = new Warcraft.NET.Files.ADT.TerrainTexture.MCNK(chunkData);
                        tex0Mcnks.Add(mcnk);
                    }
                    
                    pos += 8 + size;
                }
                
                Console.WriteLine($"[INFO] Parsed {tex0Mcnks.Count} MCNKs from tex0");
                result.Tex0McnkCount = tex0Mcnks.Count;
                
                // Merge texture data into root MCNKs
                for (int i = 0; i < Math.Min(mcnkList.Count, tex0Mcnks.Count); i++)
                {
                    if (tex0Mcnks[i].TextureLayers != null)
                        mcnkList[i].TextureLayers = tex0Mcnks[i].TextureLayers;
                    if (tex0Mcnks[i].AlphaMaps != null)
                        mcnkList[i].AlphaMaps = tex0Mcnks[i].AlphaMaps;
                    if (tex0Mcnks[i].BakedShadows != null)
                        mcnkList[i].BakedShadows = tex0Mcnks[i].BakedShadows;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to parse tex0: {ex.Message}");
            }
        }
        
        // Load obj0 for object data
        if (!string.IsNullOrEmpty(obj0Path) && File.Exists(obj0Path))
        {
            try
            {
                var obj0Bytes = File.ReadAllBytes(obj0Path);
                var obj0Chunks = ParseChunks(obj0Bytes);
                
                // Get model/WMO data
                if (obj0Chunks.TryGetValue("MMDX", out var mmdxData))
                {
                    wotlkAdt.Models = new MMDX(mmdxData);
                    Console.WriteLine($"[INFO] Loaded MMDX: {wotlkAdt.Models.Filenames.Count} models");
                }
                
                if (obj0Chunks.TryGetValue("MMID", out var mmidData))
                    wotlkAdt.ModelIndices = new MMID(mmidData);
                
                if (obj0Chunks.TryGetValue("MWMO", out var mwmoData))
                {
                    wotlkAdt.WorldModelObjects = new MWMO(mwmoData);
                    Console.WriteLine($"[INFO] Loaded MWMO: {wotlkAdt.WorldModelObjects.Filenames.Count} WMOs");
                }
                
                if (obj0Chunks.TryGetValue("MWID", out var mwidData))
                    wotlkAdt.WorldModelObjectIndices = new MWID(mwidData);
                
                if (obj0Chunks.TryGetValue("MDDF", out var mddfData))
                {
                    wotlkAdt.ModelPlacementInfo = new MDDF(mddfData);
                    Console.WriteLine($"[INFO] Loaded MDDF: {wotlkAdt.ModelPlacementInfo.MDDFEntries.Count} placements");
                }
                
                if (obj0Chunks.TryGetValue("MODF", out var modfData))
                {
                    wotlkAdt.WorldModelObjectPlacementInfo = new MODF(modfData);
                    Console.WriteLine($"[INFO] Loaded MODF: {wotlkAdt.WorldModelObjectPlacementInfo.MODFEntries.Count} placements");
                }
                
                result.Obj0McnkCount = 0; // obj0 MCNKs contain MCRD/MCRW, not full MCNKs
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to parse obj0: {ex.Message}");
            }
        }
        
        // Ensure we have 256 MCNKs
        while (mcnkList.Count < 256)
        {
            mcnkList.Add(new WotlkMCNK());
        }
        
        wotlkAdt.Chunks = mcnkList.ToArray();
        
        return wotlkAdt;
    }

    /// <summary>
    /// Parse chunks from ADT data into a dictionary.
    /// </summary>
    private Dictionary<string, byte[]> ParseChunks(byte[] data)
    {
        var result = new Dictionary<string, byte[]>();
        int pos = 0;
        
        while (pos < data.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);
            
            if (size < 0 || pos + 8 + size > data.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            var chunkData = new byte[size];
            Buffer.BlockCopy(data, pos + 8, chunkData, 0, size);
            
            // Only store first occurrence (except MCNK which we handle separately)
            if (!result.ContainsKey(readable) && readable != "MCNK")
            {
                result[readable] = chunkData;
            }
            
            pos += 8 + size;
        }
        
        return result;
    }

    /// <summary>
    /// Merge split files into an existing WotLK Terrain.
    /// </summary>
    private void MergeSplitFiles(Terrain wotlkAdt, string? obj0Path, string? tex0Path, MergeResult result)
    {
        // Load and merge tex0 if available
        if (!string.IsNullOrEmpty(tex0Path) && File.Exists(tex0Path))
        {
            try
            {
                var tex0Bytes = File.ReadAllBytes(tex0Path);
                var tex0 = new Warcraft.NET.Files.ADT.TerrainTexture.Legion.TerrainTexture(tex0Bytes);
                
                result.Tex0McnkCount = tex0.Chunks?.Length ?? 0;
                Console.WriteLine($"[INFO] Tex0 parsed: {result.Tex0McnkCount} MCNKs");

                MergeTextureData(wotlkAdt, tex0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to parse tex0: {ex.Message}");
            }
        }

        // Load and merge obj0 if available
        if (!string.IsNullOrEmpty(obj0Path) && File.Exists(obj0Path))
        {
            try
            {
                var obj0Bytes = File.ReadAllBytes(obj0Path);
                var obj0 = new Warcraft.NET.Files.ADT.TerrainObject.Zero.TerrainObjectZero(obj0Bytes);
                
                result.Obj0McnkCount = obj0.Chunks?.Length ?? 0;
                Console.WriteLine($"[INFO] Obj0 parsed: {result.Obj0McnkCount} MCNKs");

                MergeObjectData(wotlkAdt, obj0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to parse obj0: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Merge texture data from tex0 into the root ADT.
    /// tex0 contains: MTEX (texture paths), per-MCNK MCLY/MCAL/MCSH
    /// </summary>
    private void MergeTextureData(Terrain root, Warcraft.NET.Files.ADT.TerrainTexture.Legion.TerrainTexture tex0)
    {
        // Merge MTEX (texture paths)
        if (tex0.Textures?.Filenames != null && tex0.Textures.Filenames.Count > 0)
        {
            root.Textures = tex0.Textures;
            Console.WriteLine($"  Merged MTEX: {tex0.Textures.Filenames.Count} textures");
        }

        // Merge per-chunk texture data
        if (tex0.Chunks != null && root.Chunks != null)
        {
            int mergedCount = 0;
            for (int i = 0; i < Math.Min(tex0.Chunks.Length, root.Chunks.Length); i++)
            {
                var tex0Chunk = tex0.Chunks[i];
                var rootChunk = root.Chunks[i];

                if (tex0Chunk == null || rootChunk == null) continue;

                // Merge MCLY (texture layers)
                if (tex0Chunk.TextureLayers != null)
                {
                    rootChunk.TextureLayers = tex0Chunk.TextureLayers;
                }

                // Merge MCAL (alpha maps)
                if (tex0Chunk.AlphaMaps != null)
                {
                    rootChunk.AlphaMaps = tex0Chunk.AlphaMaps;
                }

                // Merge MCSH (baked shadows)
                if (tex0Chunk.BakedShadows != null)
                {
                    rootChunk.BakedShadows = tex0Chunk.BakedShadows;
                }

                mergedCount++;
            }
            Console.WriteLine($"  Merged texture data for {mergedCount} chunks");
        }
    }

    /// <summary>
    /// Merge object data from obj0 into the root ADT.
    /// obj0 contains: MMDX/MMID (M2 models), MWMO/MWID (WMOs), MDDF/MODF (placements), per-MCNK MCRD/MCRW
    /// </summary>
    private void MergeObjectData(Terrain root, Warcraft.NET.Files.ADT.TerrainObject.Zero.TerrainObjectZero obj0)
    {
        // Merge M2 model data
        if (obj0.Models?.Filenames != null && obj0.Models.Filenames.Count > 0)
        {
            root.Models = obj0.Models;
            Console.WriteLine($"  Merged MMDX: {obj0.Models.Filenames.Count} models");
        }

        if (obj0.ModelIndices != null)
        {
            root.ModelIndices = obj0.ModelIndices;
        }

        if (obj0.ModelPlacementInfo?.MDDFEntries != null && obj0.ModelPlacementInfo.MDDFEntries.Count > 0)
        {
            root.ModelPlacementInfo = obj0.ModelPlacementInfo;
            Console.WriteLine($"  Merged MDDF: {obj0.ModelPlacementInfo.MDDFEntries.Count} placements");
        }

        // Merge WMO data
        if (obj0.WorldModelObjects?.Filenames != null && obj0.WorldModelObjects.Filenames.Count > 0)
        {
            root.WorldModelObjects = obj0.WorldModelObjects;
            Console.WriteLine($"  Merged MWMO: {obj0.WorldModelObjects.Filenames.Count} WMOs");
        }

        if (obj0.WorldModelObjectIndices != null)
        {
            root.WorldModelObjectIndices = obj0.WorldModelObjectIndices;
        }

        if (obj0.WorldModelObjectPlacementInfo?.MODFEntries != null && obj0.WorldModelObjectPlacementInfo.MODFEntries.Count > 0)
        {
            root.WorldModelObjectPlacementInfo = obj0.WorldModelObjectPlacementInfo;
            Console.WriteLine($"  Merged MODF: {obj0.WorldModelObjectPlacementInfo.MODFEntries.Count} placements");
        }

        // Merge per-chunk object references
        if (obj0.Chunks != null && root.Chunks != null)
        {
            int mergedCount = 0;
            for (int i = 0; i < Math.Min(obj0.Chunks.Length, root.Chunks.Length); i++)
            {
                var obj0Chunk = obj0.Chunks[i];
                var rootChunk = root.Chunks[i];

                if (obj0Chunk == null || rootChunk == null) continue;

                // Merge MCRF (model references) - obj0 uses MCRD/MCRW but we need to combine into MCRF
                if (obj0Chunk.ModelReferences != null || obj0Chunk.WorldObjectReferences != null)
                {
                    // Create combined MCRF from MCRD + MCRW
                    var mcrf = new MCRF();
                    
                    if (obj0Chunk.ModelReferences?.ModelReferences != null)
                    {
                        mcrf.ModelReferences = new List<uint>(obj0Chunk.ModelReferences.ModelReferences);
                    }
                    else
                    {
                        mcrf.ModelReferences = new List<uint>();
                    }

                    if (obj0Chunk.WorldObjectReferences?.WorldObjectReferences != null)
                    {
                        mcrf.WorldObjectReferences = new List<uint>(obj0Chunk.WorldObjectReferences.WorldObjectReferences);
                    }
                    else
                    {
                        mcrf.WorldObjectReferences = new List<uint>();
                    }

                    rootChunk.ModelReferences = mcrf;
                    mergedCount++;
                }
            }
            Console.WriteLine($"  Merged object refs for {mergedCount} chunks");
        }
    }

    /// <summary>
    /// Batch merge all split ADTs in a directory.
    /// </summary>
    /// <param name="inputDir">Directory containing split ADT files</param>
    /// <param name="outputDir">Directory to write merged ADTs</param>
    /// <param name="mapName">Map name prefix (e.g., "development")</param>
    /// <returns>Number of successfully merged tiles</returns>
    public int BatchMerge(string inputDir, string outputDir, string mapName)
    {
        Directory.CreateDirectory(outputDir);

        // Find all root ADT files (not _obj0 or _tex0)
        var rootFiles = Directory.GetFiles(inputDir, $"{mapName}_*.adt")
            .Where(f => !f.Contains("_obj0") && !f.Contains("_tex0") && !f.Contains("_lod"))
            .OrderBy(f => f)
            .ToList();

        Console.WriteLine($"[INFO] Found {rootFiles.Count} root ADT files to merge");

        int successCount = 0;
        int failCount = 0;

        foreach (var rootPath in rootFiles)
        {
            var fileName = Path.GetFileName(rootPath);
            var baseName = Path.GetFileNameWithoutExtension(rootPath);
            
            // Construct paths for split files
            var obj0Path = Path.Combine(inputDir, $"{baseName}_obj0.adt");
            var tex0Path = Path.Combine(inputDir, $"{baseName}_tex0.adt");

            Console.WriteLine($"\n[MERGE] {fileName}");

            var result = Merge(rootPath, obj0Path, tex0Path);

            if (result.Success && result.Data != null)
            {
                var outputPath = Path.Combine(outputDir, fileName);
                File.WriteAllBytes(outputPath, result.Data);
                Console.WriteLine($"  -> {outputPath} ({result.Data.Length:N0} bytes)");
                successCount++;
            }
            else
            {
                Console.WriteLine($"  [FAIL] {result.Error}");
                failCount++;
            }
        }

        Console.WriteLine($"\n[SUMMARY] Merged: {successCount}, Failed: {failCount}");
        return successCount;
    }

    /// <summary>
    /// Compare merged ADT against a reference file.
    /// </summary>
    public void CompareWithReference(string mergedPath, string referencePath)
    {
        if (!File.Exists(mergedPath) || !File.Exists(referencePath))
        {
            Console.WriteLine("[ERROR] One or both files not found");
            return;
        }

        var mergedBytes = File.ReadAllBytes(mergedPath);
        var refBytes = File.ReadAllBytes(referencePath);

        Console.WriteLine($"\n=== Comparison: {Path.GetFileName(mergedPath)} ===");
        Console.WriteLine($"Merged size:    {mergedBytes.Length:N0} bytes");
        Console.WriteLine($"Reference size: {refBytes.Length:N0} bytes");
        Console.WriteLine($"Difference:     {mergedBytes.Length - refBytes.Length:+#;-#;0} bytes");

        try
        {
            var merged = new Terrain(mergedBytes);
            var reference = new Terrain(refBytes);

            Console.WriteLine("\n--- Chunk Comparison ---");
            CompareChunk("MTEX", 
                merged.Textures?.Filenames?.Count ?? 0,
                reference.Textures?.Filenames?.Count ?? 0);
            CompareChunk("MMDX", 
                merged.Models?.Filenames?.Count ?? 0,
                reference.Models?.Filenames?.Count ?? 0);
            CompareChunk("MWMO", 
                merged.WorldModelObjects?.Filenames?.Count ?? 0,
                reference.WorldModelObjects?.Filenames?.Count ?? 0);
            CompareChunk("MDDF", 
                merged.ModelPlacementInfo?.MDDFEntries?.Count ?? 0,
                reference.ModelPlacementInfo?.MDDFEntries?.Count ?? 0);
            CompareChunk("MODF", 
                merged.WorldModelObjectPlacementInfo?.MODFEntries?.Count ?? 0,
                reference.WorldModelObjectPlacementInfo?.MODFEntries?.Count ?? 0);
            CompareChunk("MCNK", 
                merged.Chunks?.Length ?? 0,
                reference.Chunks?.Length ?? 0);

            // Compare MCNK subchunks
            if (merged.Chunks != null && reference.Chunks != null)
            {
                int mclyCount = 0, mcalCount = 0, mcrfCount = 0;
                int refMclyCount = 0, refMcalCount = 0, refMcrfCount = 0;

                foreach (var chunk in merged.Chunks)
                {
                    if (chunk?.TextureLayers?.Layers?.Count > 0) mclyCount++;
                    if (chunk?.AlphaMaps != null) mcalCount++;
                    if (chunk?.ModelReferences != null) mcrfCount++;
                }

                foreach (var chunk in reference.Chunks)
                {
                    if (chunk?.TextureLayers?.Layers?.Count > 0) refMclyCount++;
                    if (chunk?.AlphaMaps != null) refMcalCount++;
                    if (chunk?.ModelReferences != null) refMcrfCount++;
                }

                Console.WriteLine("\n--- MCNK Subchunk Presence ---");
                CompareChunk("MCLY (chunks with layers)", mclyCount, refMclyCount);
                CompareChunk("MCAL (chunks with alpha)", mcalCount, refMcalCount);
                CompareChunk("MCRF (chunks with refs)", mcrfCount, refMcrfCount);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to parse for comparison: {ex.Message}");
        }
    }

    private void CompareChunk(string name, int merged, int reference)
    {
        var status = merged == reference ? "✓" : "✗";
        var diff = merged - reference;
        Console.WriteLine($"  {status} {name,-20} Merged: {merged,6}  Ref: {reference,6}  Diff: {diff:+#;-#;0}");
    }
}
