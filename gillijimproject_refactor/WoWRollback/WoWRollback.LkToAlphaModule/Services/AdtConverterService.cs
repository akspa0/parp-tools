using System;
using System.Threading.Tasks;

namespace WoWRollback.LkToAlphaModule.Services;

public sealed class AdtConverterService
{
    public Task<LkToAlphaConversionResult> ConvertTileAsync(
        string lkRootAdtPath,
        string lkObjAdtPath,
        string lkTexAdtPath,
        string outFile,
        string mapName,
        LkToAlphaOptions opts)
    {
        if (opts is null) throw new ArgumentNullException(nameof(opts));
        
        // 1. Validate Inputs
        if (!System.IO.File.Exists(lkRootAdtPath)) throw new System.IO.FileNotFoundException("Root ADT not found", lkRootAdtPath);
        
        int tilesProcessed = 1;
        bool success = true;
        List<string> logs = new List<string>();

        try 
        {
            // 2. Parse Root ADT for MH2O
            var rootBytes = System.IO.File.ReadAllBytes(lkRootAdtPath);
            int mh2oOffset = -1;
            int mh2oSize = 0;

            int i = 0;
            while(i + 8 <= rootBytes.Length)
            {
                string fcc = System.Text.Encoding.ASCII.GetString(rootBytes, i, 4);
                int size = BitConverter.ToInt32(rootBytes, i + 4);
                if (fcc == "O2HM") // MH2O reversed
                {
                    mh2oOffset = i + 8;
                    mh2oSize = size;
                    break;
                }
                int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
                if (next <= i) break; // prevent infinite loop
                i = next;
            }

            if (mh2oOffset > 0)
            {
                logs.Add($"Found MH2O chunk at {mh2oOffset}, size {mh2oSize}");
                byte[] mh2oData = new byte[mh2oSize];
                Array.Copy(rootBytes, mh2oOffset, mh2oData, 0, mh2oSize);

                // 3. Convert Liquid
                var mclqChunks = Converters.LiquidConverter.ConvertMh2oToMclq(mh2oData);
                logs.Add($"Converted Liquid: {mclqChunks.Length} chunks processed.");
                
                // 4. Write Debug Output (MCLQ blobs) because full ADT writer is missing
                string debugFile = outFile + ".mclq_debug";
                using (var fs = System.IO.File.Create(debugFile))
                using (var bw = new System.IO.BinaryWriter(fs))
                {
                    int mclqCount = 0;
                    foreach(var mclq in mclqChunks)
                    {
                        if (mclq != null && mclq.Length > 0)
                        {
                            bw.Write(mclq);
                            mclqCount++;
                        }
                    }
                    logs.Add($"Wrote {mclqCount} non-null MCLQ chunks to {debugFile}");
                }
            }
            else
            {
                logs.Add("No MH2O chunk found in Root ADT.");
            }
        }
        catch (Exception ex)
        {
            success = false;
            logs.Add($"Error: {ex.Message}");
        }
        
        return Task.FromResult(new LkToAlphaConversionResult(
            AlphaOutputDirectory: System.IO.Path.GetDirectoryName(outFile) ?? string.Empty,
            TilesProcessed: tilesProcessed,
            Success: success,
            ErrorMessage: string.Join("; ", logs)));
    }

    public Task<LkToAlphaConversionResult> ConvertMapAsync(
        string lkDir,
        string outDir,
        string mapName,
        LkToAlphaOptions opts)
    {
        if (opts is null) throw new ArgumentNullException(nameof(opts));
        // TODO: Enumerate tiles and call ConvertTileAsync
        return Task.FromResult(new LkToAlphaConversionResult(
            AlphaOutputDirectory: outDir,
            TilesProcessed: 0,
            Success: false,
            ErrorMessage: "Not implemented"));
    }
}
