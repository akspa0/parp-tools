// docs/AlphaWDTReader/snippets/csv_emitter.cs
// Purpose: Minimal CSV writer for per-chunk summary rows.

using System;
using System.IO;
using System.Text;

namespace Snippets
{
    public static class CsvEmitter
    {
        public static void WriteHeader(TextWriter w)
        {
            w.WriteLine("tileX,tileY,chunkX,chunkY,mcvtMin,mcvtMax,hasWater,waterCells,waterMin,waterMax,waterMean");
        }

        public static void WriteRow(TextWriter w, int tileX,int tileY,int chunkX,int chunkY,
            float mcvtMin,float mcvtMax,bool hasWater,int waterCells,float waterMin,float waterMax,float waterMean)
        {
            var sb = new StringBuilder();
            sb.Append(tileX).Append(',').Append(tileY).Append(',').Append(chunkX).Append(',').Append(chunkY).Append(',')
              .Append(mcvtMin).Append(',').Append(mcvtMax).Append(',')
              .Append(hasWater ? 1 : 0).Append(',').Append(waterCells).Append(',')
              .Append(waterMin).Append(',').Append(waterMax).Append(',').Append(waterMean);
            w.WriteLine(sb.ToString());
        }
    }
}
