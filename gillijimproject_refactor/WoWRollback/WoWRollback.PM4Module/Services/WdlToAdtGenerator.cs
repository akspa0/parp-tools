using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.PM4Module.Services;

/// <summary>
/// Generates 3.3.5 (WotLK) ADT terrain files from WDL low-resolution height data.
/// WDL has 17x17 outer + 16x16 inner heights per tile (coarse grid).
/// ADT MCNK has 145 vertices (9x9 + 8x8 interleaved) per chunk, 16x16 chunks per tile.
/// This interpolates WDL heights to ADT resolution.
/// </summary>
public static class WdlToAdtGenerator
{
    private const float TileSize = 533.3333f;
    private const float ChunkSize = TileSize / 16f;

    /// <summary>
    /// Simple WDL tile data structure.
    /// </summary>
    public class WdlTileData
    {
        public short[,] Height17 { get; } = new short[17, 17];
        public short[,] Height16 { get; } = new short[16, 16];
        public ushort[] HoleMask16 { get; } = new ushort[16];
    }

    /// <summary>
    /// MPRL terrain intersection point (transformed to MSVT coordinate space).
    /// </summary>
    public class MprlPoint
    {
        public float X { get; set; }  // MSVT X coordinate
        public float Y { get; set; }  // MSVT Y coordinate  
        public float Z { get; set; }  // Height value
        public int Floor { get; set; } // Floor level (optional)
    }

    /// <summary>
    /// Apply MPRL terrain intersection heights to refine museum texture heightmap data.
    /// MPRL contains precise Z heights where WMO/M2 objects touch the terrain.
    /// </summary>
    /// <param name="museumTexture">Museum texture data with McvtPerChunk to be modified.</param>
    /// <param name="mprlPoints">List of MPRL points in MSVT coordinate space.</param>
    /// <param name="tileMinX">Minimum X coordinate of the tile.</param>
    /// <param name="tileMinY">Minimum Y coordinate of the tile.</param>
    /// <returns>Number of vertices modified.</returns>
    public static int ApplyMprlHeights(MuseumTextureExtractor.MuseumTextureData museumTexture, 
        List<MprlPoint> mprlPoints, float tileMinX, float tileMinY)
    {
        if (museumTexture?.McvtPerChunk == null) return 0;
        
        int modifiedCount = 0;
        const float blendFactor = 0.5f;  // Weight for MPRL vs existing height
        const float spikeThreshold = 50.0f;  // Ignore changes larger than this
        
        foreach (var pt in mprlPoints)
        {
            // Calculate which chunk this point belongs to
            float relX = pt.X - tileMinX;
            float relY = pt.Y - tileMinY;
            
            // Skip points outside tile bounds
            if (relX < 0 || relX >= TileSize || relY < 0 || relY >= TileSize)
                continue;
            
            int chunkX = (int)(relX / ChunkSize);
            int chunkY = (int)(relY / ChunkSize);
            chunkX = Math.Clamp(chunkX, 0, 15);
            chunkY = Math.Clamp(chunkY, 0, 15);
            
            int chunkIdx = chunkY * 16 + chunkX;
            var mcvtData = museumTexture.McvtPerChunk[chunkIdx];
            
            if (mcvtData == null || mcvtData.Length != 145 * 4)
                continue;
            
            // Calculate vertex position within chunk
            float chunkMinX = tileMinX + chunkX * ChunkSize;
            float chunkMinY = tileMinY + chunkY * ChunkSize;
            
            float vertRelX = (pt.X - chunkMinX) / ChunkSize * 8;
            float vertRelY = (pt.Y - chunkMinY) / ChunkSize * 8;
            
            int vertCol = (int)Math.Round(vertRelX);
            int vertRow = (int)Math.Round(vertRelY * 2);  // *2 because alternating rows
            vertCol = Math.Clamp(vertCol, 0, 8);
            vertRow = Math.Clamp(vertRow, 0, 16);
            
            // Calculate vertex index in 145-vertex array
            // Layout: alternating rows of 9 outer vertices and 8 inner vertices
            int vertIdx;
            bool isInnerRow = (vertRow % 2) == 1;
            int actualRow = vertRow / 2;
            
            if (isInnerRow)
            {
                // Inner rows have 8 vertices (indices 0-7)
                int innerIdx = Math.Min(vertCol, 7);
                // Count vertices: each full pair of rows = 9 + 8 = 17 vertices
                vertIdx = actualRow * 17 + 9 + innerIdx;
            }
            else
            {
                // Outer rows have 9 vertices (indices 0-8)
                vertIdx = actualRow * 17 + vertCol;
            }
            
            if (vertIdx < 0 || vertIdx >= 145)
                continue;
            
            // Read current height from MCVT
            int byteOffset = vertIdx * 4;
            float currentHeight = BitConverter.ToSingle(mcvtData, byteOffset);
            
            // Apply blended height (MPRL Z is absolute, MCVT stores relative)
            // Need to estimate base Z from first vertex
            float baseZ = BitConverter.ToSingle(mcvtData, 0);
            float mprlRelativeHeight = pt.Z - baseZ;
            float heightDiff = mprlRelativeHeight - currentHeight;
            
            // Skip if change would cause a spike
            if (Math.Abs(heightDiff) > spikeThreshold)
                continue;
            
            // Blend toward MPRL height
            float newHeight = currentHeight + heightDiff * blendFactor;
            byte[] newHeightBytes = BitConverter.GetBytes(newHeight);
            Array.Copy(newHeightBytes, 0, mcvtData, byteOffset, 4);
            
            modifiedCount++;
        }
        
        return modifiedCount;
    }

    /// <summary>
    /// Extract terrain intersection points from PM4 MPRL data.
    /// Transforms MPRL coordinates to MSVT space for terrain analysis.
    /// </summary>
    /// <param name="pm4">PM4 file with MPRL data</param>
    /// <returns>List of MprlPoints in MSVT coordinate space</returns>
    public static List<MprlPoint> ExtractMprlPoints(PM4File pm4)
    {
        var points = new List<MprlPoint>();
        
        if (pm4.PositionRefs == null || pm4.PositionRefs.Count == 0)
            return points;
        
        // Extract non-terminator entries (Unk16 == 0 indicates normal entry)
        foreach (var entry in pm4.PositionRefs)
        {
            if (entry.Unk16 != 0)  // Skip terminators
                continue;
            
            // MPRL to MSVT coordinate mapping (discovered December 2025):
            // MPRL.PositionZ -> MSVT X
            // MPRL.PositionX -> MSVT Y
            // MPRL.PositionY -> MSVT Z (height)
            points.Add(new MprlPoint
            {
                X = entry.PositionZ,   // MPRL Z -> MSVT X
                Y = entry.PositionX,   // MPRL X -> MSVT Y
                Z = entry.PositionY,   // MPRL Y -> MSVT Z (terrain height)
                Floor = entry.Unk14    // Floor level
            });
        }
        
        return points;
    }

    /// <summary>
    /// Extract MPRL points and transform to WoW World coordinates.
    /// MPRL appears to be in a coordinate system related to placement space.
    /// </summary>
    public static List<MprlPoint> ExtractMprlPointsAsWorld(PM4File pm4, int tileX, int tileY)
    {
        var points = new List<MprlPoint>();
        
        if (pm4.PositionRefs == null || pm4.PositionRefs.Count == 0)
            return points;
        
        const float HalfMap = 17066.66656f;
        
        // Based on debug output:
        // MPRL after axis swap: X=11749-12263, Y=9604-10129
        // Expected tile world: X=4800-5333, Y=6933-7466
        // 
        // If MPRL is in placement space (like MODF writes), then:
        // worldX = 17066 - placementZ, worldY = 17066 - placementX
        //
        // Let's try: After mapping PositionZ->X, PositionX->Y,
        // they may already be in placement order
        
        foreach (var entry in pm4.PositionRefs)
        {
            if (entry.Unk16 != 0)  // Skip terminators
                continue;
            
            // Raw MPRL mapping (from spec):
            // PositionZ -> Our X
            // PositionX -> Our Y
            // PositionY -> Height
            float mprlX = entry.PositionZ;
            float mprlY = entry.PositionX;
            float height = entry.PositionY;
            
            // Try: MPRL might be using placement coordinates
            // Placement: x = 17066 - worldY, z = 17066 - worldX
            // Reverse: worldX = 17066 - placementZ, worldY = 17066 - placementX
            float worldX = HalfMap - mprlY;  // Try reversing the Y component
            float worldY = HalfMap - mprlX;  // Try reversing the X component
            
            points.Add(new MprlPoint
            {
                X = worldX,
                Y = worldY,
                Z = height,
                Floor = entry.Unk14
            });
        }
        
        return points;
    }

    /// <summary>
    /// Records a single height modification made during terrain patching.
    /// </summary>
    public record HeightDiff(int ChunkX, int ChunkY, int VertexIdx, float OriginalHeight, float NewHeight, float MprlHeight);

    /// <summary>
    /// Patch MCVT heights directly in existing ADT bytes (in-place modification).
    /// This preserves chunk continuity and all original positioning.
    /// </summary>
    public static int PatchAdtMcvtInPlace(byte[] adtBytes, List<MprlPoint> mprlPoints, int tileX, int tileY)
    {
        return PatchAdtMcvtInPlace(adtBytes, mprlPoints, tileX, tileY, out _);
    }

    /// <summary>
    /// Patch MCVT heights with diff output for verification.
    /// </summary>
    public static int PatchAdtMcvtInPlace(byte[] adtBytes, List<MprlPoint> mprlPoints, int tileX, int tileY, out List<HeightDiff> diffs)
    {
        diffs = new List<HeightDiff>();
        
        if (mprlPoints == null || mprlPoints.Count == 0)
            return 0;
        
        int modifiedCount = 0;
        const float blendFactor = 0.5f;  // Interpolate 50% with old terrain to prevent "holes"/tearing
        const float spikeThreshold = 500.0f;  // Increased from 50 for large terrain differences
        bool debugHeightPrinted = false;
        
        // Calculate tile world bounds (WoW world coordinate space)
        // ADT tiles are 533.33 yards each, with tile 32,32 at world origin
        float tileWorldX = (32 - tileX) * TileSize;
        float tileWorldY = (32 - tileY) * TileSize;
        
        // Parse ADT to find MCNK chunks and their MCVT subchunks
        int pos = 0;
        int mcnkIndex = 0;
        int mcvtFoundCount = 0;
        
        while (pos + 8 <= adtBytes.Length)
        {
            // Read chunk signature (reversed)
            string sig = System.Text.Encoding.ASCII.GetString(adtBytes, pos, 4);
            string readable = new string(sig.Reverse().ToArray());
            uint size = BitConverter.ToUInt32(adtBytes, pos + 4);
            
            if (pos + 8 + size > adtBytes.Length)
                break;
            
            if (readable == "MCNK")
            {
                // Parse MCNK header to get MCVT offset
                int mcnkDataStart = pos + 8;
                
                if (mcnkDataStart + 128 > adtBytes.Length)
                    break;
                
                // MCNK header: flags @ 0, IndexX @ 4, IndexY @ 8, ofsHeight @ 0x14
                uint chunkX = BitConverter.ToUInt32(adtBytes, mcnkDataStart + 4);
                uint chunkY = BitConverter.ToUInt32(adtBytes, mcnkDataStart + 8);
                uint ofsHeight = BitConverter.ToUInt32(adtBytes, mcnkDataStart + 0x14);
                
                // baseZ is at position offset 0x70 (position.z in C3Vector at 0x68)
                // C3Vector: x @ 0x68, y @ 0x6C, z @ 0x70
                float baseZ = BitConverter.ToSingle(adtBytes, mcnkDataStart + 0x70);
                
                if (ofsHeight > 0)
                {
                    // MCVT offset is relative to MCNK chunk start (including 8-byte header)
                    int mcvtPos = pos + (int)ofsHeight;
                    
                    if (mcvtPos + 8 <= adtBytes.Length)
                    {
                        // Verify it's MCVT
                        string mcvtSig = System.Text.Encoding.ASCII.GetString(adtBytes, mcvtPos, 4);
                        string mcvtReadable = new string(mcvtSig.Reverse().ToArray());
                        

                        
                        if (mcvtReadable == "MCVT")
                        {
                            mcvtFoundCount++;
                            uint mcvtSize = BitConverter.ToUInt32(adtBytes, mcvtPos + 4);
                            int mcvtDataStart = mcvtPos + 8;
                            
                            if (mcvtSize >= 145 * 4 && mcvtDataStart + 145 * 4 <= adtBytes.Length)
                            {
                                // Calculate chunk world position
                                // Tile corner is at (tileWorldX, tileWorldY), tile spans DOWNWARD
                                // Chunk (0,0) is at tile corner, Chunk (15,15) is at tile_corner - 15*ChunkSize
                                // Each chunk covers ChunkSize (33.33 units) from its position going DOWN
                                float chunkMinX = tileWorldX - (chunkX + 1) * ChunkSize;  // Lower bound
                                float chunkMaxX = tileWorldX - chunkX * ChunkSize;        // Upper bound
                                float chunkMinY = tileWorldY - (chunkY + 1) * ChunkSize;
                                float chunkMaxY = tileWorldY - chunkY * ChunkSize;
                                

                                

                                

                                

                                
                                // Apply each MPRL point that falls within or near this chunk
                                foreach (var pt in mprlPoints)
                                {
                                    // Check if point is within this chunk (with margin)
                                    float margin = ChunkSize * 0.1f;
                                    if (pt.X < chunkMinX - margin || pt.X > chunkMaxX + margin)
                                        continue;
                                    if (pt.Y < chunkMinY - margin || pt.Y > chunkMaxY + margin)
                                        continue;
                                    
                                    // Calculate vertex position within chunk
                                    // Chunk spans from chunkMinX to chunkMaxX
                                    // relX = 0 at chunkMaxX, 8 at chunkMinX (8 outer vertices per row)
                                    float relX = (chunkMaxX - pt.X) / ChunkSize * 8;
                                    float relY = (chunkMaxY - pt.Y) / ChunkSize * 8;
                                    
                                    int vertCol = (int)Math.Round(relX);
                                    int vertRow = (int)Math.Round(relY * 2);
                                    
                                    if (vertCol < 0 || vertCol > 8 || vertRow < 0 || vertRow > 16)
                                        continue;
                                    
                                    // Calculate vertex index in 145-vertex array
                                    bool isInnerRow = (vertRow % 2) == 1;
                                    int actualRow = vertRow / 2;
                                    
                                    int vertIdx;
                                    if (isInnerRow)
                                    {
                                        int innerIdx = Math.Min(vertCol, 7);
                                        vertIdx = actualRow * 17 + 9 + innerIdx;
                                    }
                                    else
                                    {
                                        vertIdx = actualRow * 17 + vertCol;
                                    }
                                    
                                    if (vertIdx < 0 || vertIdx >= 145)
                                        continue;
                                    
                                    // Read current height (relative to baseZ)
                                    int byteOffset = mcvtDataStart + vertIdx * 4;
                                    float currentRelHeight = BitConverter.ToSingle(adtBytes, byteOffset);
                                    float currentAbsHeight = baseZ + currentRelHeight;
                                    
                                    // Calculate height difference
                                    float heightDiff = pt.Z - currentAbsHeight;
                                    

                                    
                                    // Skip if change would cause a spike
                                    if (Math.Abs(heightDiff) > spikeThreshold)
                                        continue;
                                    
                                    // Blend toward MPRL height (modify relative height)
                                    float newRelHeight = currentRelHeight + heightDiff * blendFactor;
                                    
                                    // CRITICAL: Ensure we don't write invalid floats (NaN/Infinity) which crash Noggit
                                    if (float.IsNaN(newRelHeight) || float.IsInfinity(newRelHeight))
                                    {
                                        continue; 
                                    }
                                    
                                    byte[] newHeightBytes = BitConverter.GetBytes(newRelHeight);
                                    Array.Copy(newHeightBytes, 0, adtBytes, byteOffset, 4);
                                    
                                    // Record the diff for verification
                                    diffs.Add(new HeightDiff(
                                        (int)chunkX, (int)chunkY, vertIdx,
                                        currentAbsHeight, baseZ + newRelHeight, pt.Z));
                                    
                                    modifiedCount++;
                                }
                            }
                        }
                    }
                }
                
                mcnkIndex++;
            }
            
            pos += 8 + (int)size;
        }
        
        return modifiedCount;
    }

    /// <summary>
    /// Generate a complete 3.3.5 monolithic ADT from WDL tile heights.
    /// </summary>
    public static byte[] GenerateAdt(WdlTileData wdlTile, int tileX, int tileY)
        => GenerateAdt(wdlTile, tileX, tileY, null);

    /// <summary>
    /// Generate a complete 3.3.5 monolithic ADT from WDL tile heights with optional minimap MCCV.
    /// </summary>
    /// <param name="mccvData">Optional array of 256 MCCV byte arrays (one per MCNK), or null for neutral gray.</param>
    public static byte[] GenerateAdt(WdlTileData wdlTile, int tileX, int tileY, byte[][]? mccvData)
        => GenerateAdt(wdlTile, tileX, tileY, mccvData, null);

    /// <summary>
    /// Generate a complete 3.3.5 monolithic ADT from WDL tile heights with museum texturing.
    /// </summary>
    /// <param name="mccvData">Optional array of 256 MCCV byte arrays, or null for neutral gray.</param>
    /// <param name="museumTexture">Museum texture data (MTEX, MCLY, MCAL, MCCV), or null for no textures.</param>
    public static byte[] GenerateAdt(WdlTileData wdlTile, int tileX, int tileY, byte[][]? mccvData, 
        MuseumTextureExtractor.MuseumTextureData? museumTexture)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // MVER - version 18 for 3.3.5
        WriteChunk(bw, "MVER", BitConverter.GetBytes(18u));

        // MHDR - header (64 bytes, will be updated later)
        long mhdrPos = ms.Position;
        WriteChunk(bw, "MHDR", new byte[64]);

        // Track chunk positions for MHDR offsets
        long mcinPos = ms.Position;
        WriteChunk(bw, "MCIN", new byte[256 * 16]);

        // MTEX - use museum textures if available
        long mtexPos = ms.Position;
        if (museumTexture != null && museumTexture.MtexData.Length > 0)
        {
            WriteChunk(bw, "MTEX", museumTexture.MtexData);
        }
        else
        {
            WriteChunk(bw, "MTEX", Array.Empty<byte>());
        }

        long mmdxPos = ms.Position;
        WriteChunk(bw, "MMDX", Array.Empty<byte>());

        long mmidPos = ms.Position;
        WriteChunk(bw, "MMID", Array.Empty<byte>());

        long mwmoPos = ms.Position;
        WriteChunk(bw, "MWMO", Array.Empty<byte>());

        long mwidPos = ms.Position;
        WriteChunk(bw, "MWID", Array.Empty<byte>());

        long mddfPos = ms.Position;
        WriteChunk(bw, "MDDF", Array.Empty<byte>());

        long modfPos = ms.Position;
        WriteChunk(bw, "MODF", Array.Empty<byte>());

        // Generate 256 MCNK chunks (16x16 grid)
        var mcnkOffsets = new uint[256];
        var mcnkSizes = new uint[256];

        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                int idx = cy * 16 + cx;
                mcnkOffsets[idx] = (uint)ms.Position;
                
                // Get per-chunk data from museum (prefer museum data over WDL)
                var mcnkMccv = museumTexture?.MccvPerChunk[idx] ?? mccvData?[idx];
                var mcnkMcly = museumTexture?.MclyPerChunk[idx];
                var mcnkMcal = museumTexture?.McalPerChunk[idx];
                var mcnkMcvt = museumTexture?.McvtPerChunk[idx]; // Museum heights
                int nLayers = museumTexture?.NLayersPerChunk[idx] ?? 0;
                
                var mcnkData = GenerateMcnk(wdlTile, tileX, tileY, cx, cy, mcnkMccv, mcnkMcly, mcnkMcal, mcnkMcvt, nLayers);
                WriteChunk(bw, "MCNK", mcnkData);
                
                mcnkSizes[idx] = (uint)mcnkData.Length;
            }
        }

        // MFBO - flight bounds (optional, 36 bytes)
        long mfboPos = ms.Position;
        var mfboData = new byte[36];
        short maxHeight = 500;
        short minHeight = -500;
        for (int i = 0; i < 9; i++)
        {
            BitConverter.GetBytes(maxHeight).CopyTo(mfboData, i * 2);
            BitConverter.GetBytes(minHeight).CopyTo(mfboData, 18 + i * 2);
        }
        WriteChunk(bw, "MFBO", mfboData);

        // Update MCIN with chunk offsets
        var result = ms.ToArray();
        for (int i = 0; i < 256; i++)
        {
            int mcinEntryPos = (int)mcinPos + 8 + i * 16;
            BitConverter.GetBytes(mcnkOffsets[i]).CopyTo(result, mcinEntryPos);
            BitConverter.GetBytes(mcnkSizes[i] + 8).CopyTo(result, mcinEntryPos + 4);
        }

        // Update MHDR offsets (relative to MHDR data start)
        int mhdrDataStart = (int)mhdrPos + 8;
        
        uint mhdrFlags = 0x01; // has MFBO
        BitConverter.GetBytes(mhdrFlags).CopyTo(result, mhdrDataStart + 0x00);
        BitConverter.GetBytes((uint)(mcinPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x04);
        BitConverter.GetBytes((uint)(mtexPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x08);
        BitConverter.GetBytes((uint)(mmdxPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x0C);
        BitConverter.GetBytes((uint)(mmidPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x10);
        BitConverter.GetBytes((uint)(mwmoPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x14);
        BitConverter.GetBytes((uint)(mwidPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x18);
        BitConverter.GetBytes((uint)(mddfPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x1C);
        BitConverter.GetBytes((uint)(modfPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x20);
        BitConverter.GetBytes((uint)(mfboPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x24);

        return result;
    }

    private static byte[] GenerateMcnk(WdlTileData wdlTile, int tileX, int tileY, int chunkX, int chunkY, 
        byte[]? mccvData = null, byte[]? mclyData = null, byte[]? mcalData = null, byte[]? mcvtData = null, int nLayers = 0)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        float baseX = (32 - tileX) * TileSize - chunkX * ChunkSize;
        float baseY = (32 - tileY) * TileSize - chunkY * ChunkSize;
        
        // Use museum MCVT heights if available, otherwise interpolate from WDL
        float baseZ;
        byte[]? mcvtBytes = null;
        
        if (mcvtData != null && mcvtData.Length == 145 * 4)
        {
            // Use museum heights directly
            mcvtBytes = mcvtData;
            // Get baseZ from first height value in museum MCVT
            baseZ = BitConverter.ToSingle(mcvtData, 0);
        }
        else
        {
            // Fall back to WDL interpolation
            float[] heights = InterpolateChunkHeights(wdlTile, chunkX, chunkY);
            baseZ = heights[0];
            
            // Convert heights to bytes
            using var heightMs = new MemoryStream();
            using var heightBw = new BinaryWriter(heightMs);
            for (int i = 0; i < 145; i++)
                heightBw.Write(heights[i] - baseZ);
            mcvtBytes = heightMs.ToArray();
        }

        // MCNK header (128 bytes)
        uint flags = 0x40u; // has_mccv flag
        bw.Write(flags);
        bw.Write((uint)chunkX);
        bw.Write((uint)chunkY);
        bw.Write((uint)nLayers); // nLayers from museum
        bw.Write(0u); // nDoodadRefs
        bw.Write(0u); // ofsHeight
        bw.Write(0u); // ofsNormal
        bw.Write(0u); // ofsLayer
        bw.Write(0u); // ofsRefs
        bw.Write(0u); // ofsAlpha
        bw.Write(0u); // sizeAlpha
        bw.Write(0u); // ofsShadow
        bw.Write(0u); // sizeShadow
        bw.Write(0u); // areaid
        bw.Write(0u); // nMapObjRefs
        bw.Write(0u); // holes
        for (int i = 0; i < 8; i++) bw.Write((ushort)0); // doodadMapping
        for (int i = 0; i < 8; i++) bw.Write((byte)0);   // doodadStencil
        bw.Write(0u); // ofsSndEmitters
        bw.Write(0u); // nSndEmitters
        bw.Write(0u); // ofsLiquid
        bw.Write(0u); // sizeLiquid
        bw.Write(baseX); // zpos (WoW X)
        bw.Write(baseY); // xpos (WoW Z)
        bw.Write(baseZ); // ypos (WoW Y)
        bw.Write(0u); // ofsMCCV
        bw.Write(0u); // unused1
        bw.Write(0u); // unused2

        // MCVT - height map
        // Note: offsets are relative to MCNK chunk start (including 8-byte chunk header)
        // WriteChunk adds the 8-byte header, so we add 8 to our data positions
        uint mcvtOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("TVCM"));
        bw.Write(145 * 4);
        bw.Write(mcvtBytes!);

        // MCCV - vertex colors
        uint mccvOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("VCCM"));
        bw.Write(145 * 4);
        if (mccvData != null && mccvData.Length == 145 * 4)
        {
            bw.Write(mccvData);
        }
        else
        {
            for (int i = 0; i < 145; i++)
            {
                bw.Write((byte)0x7F);
                bw.Write((byte)0x7F);
                bw.Write((byte)0x7F);
                bw.Write((byte)0x00);
            }
        }

        // MCNR - normals
        uint mcnrOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("RNCM"));
        bw.Write(448);
        for (int i = 0; i < 145; i++)
        {
            bw.Write((sbyte)0);
            bw.Write((sbyte)127);
            bw.Write((sbyte)0);
        }
        for (int i = 0; i < 13; i++) bw.Write((byte)0);

        // MCLY - texture layers (use museum data if available)
        uint mclyOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("YLCM"));
        if (mclyData != null && mclyData.Length > 0)
        {
            bw.Write(mclyData.Length);
            bw.Write(mclyData);
        }
        else
        {
            bw.Write(0);
        }
        
        uint mcrfOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("FRCM")); bw.Write(0);
        
        // MCAL - alpha maps (use museum data if available)
        uint mcalOffset = (uint)ms.Position + 8;
        int mcalSize = 0;
        bw.Write(Encoding.ASCII.GetBytes("LACM"));
        if (mcalData != null && mcalData.Length > 0)
        {
            mcalSize = mcalData.Length;
            bw.Write(mcalSize);
            bw.Write(mcalData);
        }
        else
        {
            bw.Write(0);
        }
        
        uint mcshOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("HSCM")); bw.Write(0);
        
        uint mcseOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("ESCM")); bw.Write(0);

        var result = ms.ToArray();

        // Update header offsets in MCNK
        BitConverter.GetBytes(mcvtOffset).CopyTo(result, 0x14);
        BitConverter.GetBytes(mcnrOffset).CopyTo(result, 0x18);
        BitConverter.GetBytes(mclyOffset).CopyTo(result, 0x1C);
        BitConverter.GetBytes(mcrfOffset).CopyTo(result, 0x20);
        BitConverter.GetBytes(mcalOffset).CopyTo(result, 0x24);
        BitConverter.GetBytes((uint)mcalSize).CopyTo(result, 0x28); // sizeAlpha
        BitConverter.GetBytes(mcshOffset).CopyTo(result, 0x2C);
        BitConverter.GetBytes(mcseOffset).CopyTo(result, 0x58);
        BitConverter.GetBytes(mccvOffset).CopyTo(result, 0x74);

        return result;
    }

    private static float[] InterpolateChunkHeights(WdlTileData wdlTile, int chunkX, int chunkY)
    {
        var heights = new float[145];
        
        // Get the 4 corner heights for this chunk from WDL 17x17 grid
        float h00 = wdlTile.Height17[chunkY, chunkX];
        float h10 = wdlTile.Height17[chunkY, Math.Min(chunkX + 1, 16)];
        float h01 = wdlTile.Height17[Math.Min(chunkY + 1, 16), chunkX];
        float h11 = wdlTile.Height17[Math.Min(chunkY + 1, 16), Math.Min(chunkX + 1, 16)];

        // Use Height16 center if available for better accuracy
        float hCenter = (chunkX < 16 && chunkY < 16) 
            ? wdlTile.Height16[chunkY, chunkX] 
            : (h00 + h10 + h01 + h11) / 4f;

        int idx = 0;
        for (int row = 0; row < 17; row++)
        {
            bool isInnerRow = (row % 2) == 1;
            int colCount = isInnerRow ? 8 : 9;

            for (int col = 0; col < colCount; col++)
            {
                float u, v;
                if (isInnerRow)
                {
                    u = (col + 0.5f) / 8f;
                    v = row / 16f;
                }
                else
                {
                    u = col / 8f;
                    v = row / 16f;
                }

                // Simple bilinear interpolation - stable and doesn't cause distortion
                float height = BilinearInterpolate(h00, h10, h01, h11, u, v);
                
                // Blend with center height for better accuracy
                float centerWeight = 1f - 2f * Math.Max(Math.Abs(u - 0.5f), Math.Abs(v - 0.5f));
                centerWeight = Math.Max(0f, centerWeight);
                height = height * (1f - centerWeight * 0.3f) + hCenter * (centerWeight * 0.3f);

                heights[idx++] = height;
            }
        }

        return heights;
    }

    private static float BilinearInterpolate(float h00, float h10, float h01, float h11, float u, float v)
    {
        float top = h00 * (1f - u) + h10 * u;
        float bottom = h01 * (1f - u) + h11 * u;
        return top * (1f - v) + bottom * v;
    }

    private static void WriteChunk(BinaryWriter bw, string sig, byte[] data)
    {
        var sigBytes = Encoding.ASCII.GetBytes(sig);
        Array.Reverse(sigBytes);
        bw.Write(sigBytes);
        bw.Write(data.Length);
        bw.Write(data);
    }
}
