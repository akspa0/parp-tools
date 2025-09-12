// docs/AlphaWDTReader/snippets/mh2o_emit_helpers.cs
// Purpose: Assemble MH2O data for a tile from per-chunk layer specs.
// Uses MaskPacking (row-major 1-bit masks) and OffsetBuilder (4-byte alignment) snippets.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Snippets
{
    // Minimal per-layer spec used by the converter once Alpha water has been grouped
    public sealed class Mh2oLayerSpec
    {
        public ushort MinX, MinY, MaxX, MaxY; // inclusive bbox in cell coords
        public ushort Flags;                  // liquid type/flags (conservative mapping)
        public float MinHeight;               // stats/min
        public float MaxHeight;               // stats/max
        public bool[,] Mask;                  // occupancy within bbox (width = MaxX-MinX+1)
        public float[,] Height;               // per-cell height (same dims as Mask)
    }

    // Per-MCNK entry wrapper
    public sealed class Mh2oChunkEntry
    {
        public int ChunkIndex;                // 0..255 (row*16+col)
        public List<Mh2oLayerSpec> Layers = new();
    }

    public static class Mh2oEmit
    {
        // Emit a single MH2O block for the entire ADT from per-chunk entries.
        // Returns the MH2O payload bytes. Caller sets MHDR.OfsMH2O and size.
        public static byte[] BuildMh2o(IReadOnlyList<Mh2oChunkEntry> chunks)
        {
            // Layout: [FourCC 'MH2O'][u32 size][per-chunk headers table + variable data]
            var ob = new OffsetBuilder();

            // Reserve space for 256 per-chunk headers (each: LayerCount, OfsLayers, OfsAttribs)
            const int PerChunkHeaderSize = 12; // three u32
            int hdrStart = ob.Position;
            for (int i = 0; i < 256; i++) ob.WriteZeros(PerChunkHeaderSize);

            // For each present chunk, write its layers and patch its header
            for (int i = 0; i < chunks.Count; i++)
            {
                var ce = chunks[i];
                int headerOffset = hdrStart + ce.ChunkIndex * PerChunkHeaderSize;

                if (ce.Layers == null || ce.Layers.Count == 0)
                {
                    // leave zeroed header (no layers)
                    continue;
                }

                int layersOffset = ob.Align4();

                // Reserve layer headers (we use a compact header compatible with our reader/writer)
                // struct Layer { u16 minx,miny,maxx,maxy; u16 flags; u16 pad; u32 ofsHeight; u32 ofsMask; f32 minH; f32 maxH; }
                const int LayerHeaderSize = 2+2+2+2 + 2+2 + 4+4 + 4+4; // 24 bytes
                int layersHeaderStart = ob.Position;
                ob.WriteZeros(LayerHeaderSize * ce.Layers.Count);

                // Write per-layer payloads and patch their headers
                for (int li = 0; li < ce.Layers.Count; li++)
                {
                    var L = ce.Layers[li];
                    int w = L.MaxX - L.MinX + 1;
                    int h = L.MaxY - L.MinY + 1;

                    // Mask
                    int ofsMask = ob.Align4();
                    byte[] packedMask = MaskPacking.PackRowMajor(L.Mask);
                    ob.WriteBytes(packedMask);

                    // Heights
                    int ofsHeights = ob.Align4();
                    using (var ms = new MemoryStream())
                    using (var bw = new BinaryWriter(ms))
                    {
                        for (int yy = 0; yy < h; yy++)
                            for (int xx = 0; xx < w; xx++)
                                bw.Write(L.Height[xx, yy]);
                        ob.WriteBytes(ms.ToArray());
                    }

                    // Patch layer header
                    int lh = layersHeaderStart + li * LayerHeaderSize;
                    ob.PatchU16(lh + 0, L.MinX);
                    ob.PatchU16(lh + 2, L.MinY);
                    ob.PatchU16(lh + 4, L.MaxX);
                    ob.PatchU16(lh + 6, L.MaxY);
                    ob.PatchU16(lh + 8, L.Flags);
                    ob.PatchU16(lh +10, 0); // pad
                    ob.PatchU32(lh +12, (uint)ofsHeights);
                    ob.PatchU32(lh +16, (uint)ofsMask);
                    ob.PatchF32(lh +20, L.MinHeight);
                    ob.PatchF32(lh +24, L.MaxHeight);
                }

                // Patch per-chunk header
                ob.PatchU32(headerOffset + 0, (uint)ce.Layers.Count);
                ob.PatchU32(headerOffset + 4, (uint)layersOffset);
                ob.PatchU32(headerOffset + 8, 0); // OfsAttribs unused
            }

            // Wrap with chunk header FourCC+size
            using var outer = new MemoryStream();
            using var bw = new BinaryWriter(outer);
            FourCC.WriteFourCC(bw, FourCC.MH2O);
            bw.Write((uint)ob.Length);
            bw.Write(ob.ToArray());
            return outer.ToArray();
        }
    }
}
