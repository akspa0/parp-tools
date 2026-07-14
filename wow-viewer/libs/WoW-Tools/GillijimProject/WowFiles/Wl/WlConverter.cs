using System;
using System.Collections.Generic;
using System.Numerics;
using GillijimProject.Utilities;
using GillijimProject.WowFiles.Alpha;
using GillijimProject.WowFiles.LichKing;

namespace GillijimProject.WowFiles.Wl;

public static class WlConverter
{
    /// <summary>
    /// Converts a WlwBlock (4x4 vertices) to an Alpha MCLQ chunk (9x9 vertices).
    /// Upscales grid using bicubic or linear interpolation.
    /// </summary>
    public static byte[] ToMclq(WlBlock block, int liquidType)
    {
        // 1. Create 9x9 Height Grid from 4x4 using Interpolation
        // Grid size: Wlw is 4x4 vertices?
        // Wait, 16 vertices could be 4x4.
        // If it covers a whole ADT chunk (533.333 units), then 4x4 is rough.
        // If it covers an ADT (16 chunks), then 4x4 is extremely rough.
        // Assuming block corresponds to a CHUNK (MCNK).
        
        float[] heights = Upscale4x4To9x9(block.Vertices);
        
        // 2. Create Tiles (8x8)
        // Derive flags/mask from Data? Or valid height range?
        // For now, assume full visibility if block exists.
        byte[,] tiles = new byte[8, 8];
        byte liqByte = (byte)(liquidType & 0x0F); 
        
        for(int y=0; y<8; y++)
            for(int x=0; x<8; x++)
                tiles[y,x] = liqByte; // Full liquid
                
        // 3. Serialize to MCLQ (standard Alpha format)
        // Similar to MclqBackporter logic.
        // ... (reuse serialization logic or refactor MclqBackporter to be public/shared)
        
        return SerializeMclq(heights, tiles, liquidType);
    }
    
    private static float[] Upscale4x4To9x9(Vector3[] source)
    {
        // Source: 4x4 row-major.
        // Target: 9x9 row-major.
        // Source range: 0..3 (indices). Target range: 0..8 (indices).
        // Scale factor: x' = x * (3.0 / 8.0) ? No.
        // We want to map [0..8] to [0..3]. 
        // t = index / 8.0. Source coord = t * 3.0.
        
        float[] result = new float[81];
        
        for(int y=0; y<9; y++)
        {
            float v = (y / 8.0f) * 3.0f;
            for(int x=0; x<9; x++)
            {
                float u = (x / 8.0f) * 3.0f;
                result[y*9 + x] = BiLinearSample(source, u, v);
            }
        }
        return result;
    }
    
    private static float BiLinearSample(Vector3[] grid, float u, float v)
    {
        // Grid is 4x4. u,v in [0,3].
        int x0 = (int)Math.Floor(u);
        int y0 = (int)Math.Floor(v);
        int x1 = Math.Min(x0 + 1, 3);
        int y1 = Math.Min(y0 + 1, 3);
        
        float tx = u - x0;
        float ty = v - y0;
        
        float h00 = grid[y0 * 4 + x0].Z; // Z is height? "z-up" in WLW doc? 
        // Doc says "vertices ( z-up )". Yes.
        float h10 = grid[y0 * 4 + x1].Z;
        float h01 = grid[y1 * 4 + x0].Z;
        float h11 = grid[y1 * 4 + x1].Z;
        
        float lerpX0 = Lerp(h00, h10, tx);
        float lerpX1 = Lerp(h01, h11, tx);
        return Lerp(lerpX0, lerpX1, ty);
    }
    
    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    private static byte[] SerializeMclq(float[] heights, byte[,] tiles, int type)
    {
         // TODO: Refactor MclqBackporter to expose this or duplicate for now.
         // Duplicating for speed in POC.
         using var ms = new MemoryStream();
         using var bw = new BinaryWriter(ms);
         // Min/Max
         float min = float.MaxValue, max = float.MinValue;
         foreach(var h in heights) { if(h<min) min=h; if(h>max) max=h; }
         bw.Write(min); bw.Write(max);
         
         // 9x9 Vertices. If Magma/Water, store height.
         // Alpha format: 81 structs.
         // Assuming Water (Type 1) for generic.
         // If Type 1 (Water): [d, f0, f1, fill, h]
         if (type == 2 || type == 6) // Magma (WLW type 6 = Magma, Alpha type 3 = Magma)
         {
             // Magma: [s, t, h] (u16, u16, f32)
             for(int i=0; i<81; i++) { bw.Write((ushort)0); bw.Write((ushort)0); bw.Write(heights[i]); }
         }
         else
         {
             // Water: [d, f, f, f, h]
             for(int i=0; i<81; i++) { bw.Write((byte)128); bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); bw.Write(heights[i]); }
         }
         
         // 8x8 Tiles
         for(int y=0; y<8; y++)
            for(int x=0; x<8; x++)
                bw.Write(tiles[y,x]);
                
         return ms.ToArray();
    }
}
