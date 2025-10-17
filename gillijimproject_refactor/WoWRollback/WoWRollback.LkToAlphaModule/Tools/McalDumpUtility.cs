using System;
using System.IO;

namespace WoWRollback.LkToAlphaModule.Tools;

public static class McalDumpUtility
{
    private const int LayerSize = 64 * 64;

    public static void Dump(string root, int indexX, int indexY, string stage, bool ascii, int? asciiLayer)
    {
        if (string.IsNullOrWhiteSpace(root)) root = "debug_mcal";
        if (string.IsNullOrWhiteSpace(stage)) stage = "alpha";

        string folder = Path.Combine(root, $"{indexY:D2}_{indexX:D2}");
        string filePath = Path.Combine(folder, $"{stage}_mcal.bin");

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"MCAL dump not found: {filePath}");
            return;
        }

        byte[] data = File.ReadAllBytes(filePath);
        if (data.Length == 0)
        {
            Console.WriteLine("MCAL dump is empty.");
            return;
        }

        if (data.Length % LayerSize != 0)
        {
            Console.WriteLine($"Unexpected MCAL length {data.Length}; must be multiple of {LayerSize}.");
            return;
        }

        int layerCount = data.Length / LayerSize;
        Console.WriteLine($"MCAL summary for {stage} tile ({indexX},{indexY})");
        Console.WriteLine($"Total bytes: {data.Length}, layers (excluding base): {layerCount}");

        for (int layer = 0; layer < layerCount; layer++)
        {
            Span<byte> slice = new Span<byte>(data, layer * LayerSize, LayerSize);
            int min = 255;
            int max = 0;
            int nonZero = 0;
            long sum = 0;

            for (int i = 0; i < slice.Length; i++)
            {
                byte value = slice[i];
                if (value < min) min = value;
                if (value > max) max = value;
                if (value != 0) nonZero++;
                sum += value;
            }

            double avg = sum / (double)slice.Length;
            Console.WriteLine($"Layer {layer}: min={min}, max={max}, nonZero={nonZero}, avg={avg:F2}");

            if (ascii && (asciiLayer == null || asciiLayer.Value == layer))
            {
                PrintAsciiLayer(slice, layer);
            }
        }
    }

    private static void PrintAsciiLayer(Span<byte> layerData, int layerIndex)
    {
        const string ramp = " .:-=+*#%@";
        Console.WriteLine($"ASCII preview for layer {layerIndex}:");

        for (int y = 0; y < 64; y++)
        {
            Span<char> row = stackalloc char[64];
            for (int x = 0; x < 64; x++)
            {
                byte value = layerData[y * 64 + x];
                int rampIndex = value * (ramp.Length - 1) / 255;
                row[x] = ramp[rampIndex];
            }
            Console.WriteLine(new string(row));
        }
    }
}
