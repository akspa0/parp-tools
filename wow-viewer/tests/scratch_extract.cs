using System;
using System.IO;
using System.Text;
using WoWViewer.DataSources;
using WowViewer.Core.IO.Converters;

class Extractor {
    static void Main() {
        var ds = new MpqDataSource(@"H:\CLIENTS\TBC\2.X_Retail_Windows_enUS_2.4.3.8606\World of Warcraft");
        string wmoPath = @"WORLD\ENVIRONMENT\DOODAD\NAGRAND\ROCKSFLOATING\NAGRAND_ROCKFLOATING_01.WMO";
        var rootBytes = ds.ReadFile(wmoPath);
        if (rootBytes == null) {
            Console.WriteLine("Could not read root WMO: " + wmoPath);
            return;
        }
        Console.WriteLine($"Root WMO size: {rootBytes.Length} bytes");
        var groups = new System.Collections.Generic.List<byte[]>();
        for (int gi = 0; ; gi++) {
            string gPath = $@"WORLD\ENVIRONMENT\DOODAD\NAGRAND\ROCKSFLOATING\NAGRAND_ROCKFLOATING_01_{gi:D3}.WMO";
            var gBytes = ds.ReadFile(gPath);
            if (gBytes == null) break;
            Console.WriteLine($"Found group {gi}: {gBytes.Length} bytes");
            groups.Add(gBytes);
        }

        var v17Parser = new WmoV17ToV14Converter();
        var wmo = v17Parser.ParseV17ToModel(rootBytes, groups);
        Console.WriteLine($"Parsed WMO: Materials={wmo.Materials.Count}, Groups={wmo.Groups.Count}");
        for (int i = 0; i < wmo.Materials.Count; i++) {
            var m = wmo.Materials[i];
            Console.WriteLine($"Mat {i}: Shader={m.Shader}, Blend={m.BlendMode}, Flags=0x{m.Flags:X}, Tex1='{m.Texture1Name}', Tex2='{m.Texture2Name}', Tex3='{m.Texture3Name}'");
        }
        for (int gi = 0; gi < wmo.Groups.Count; gi++) {
            var g = wmo.Groups[gi];
            Console.WriteLine($"Group {gi} '{g.Name}': Flags=0x{g.Flags:X}, Verts={g.Vertices.Count}, UVs={g.UVs.Count}, Colors={g.VertexColors.Count}, Batches={g.Batches.Count}");
            for (int bi = 0; bi < g.Batches.Count; bi++) {
                var b = g.Batches[bi];
                Console.WriteLine($"  Batch {bi}: FirstIdx={b.FirstIndex}, Count={b.IndexCount}, MatId={b.MaterialId}, MinIdx={b.MinIndex}, MaxIdx={b.MaxIndex}");
            }
            // Let's sample min/max UVs and colors
            float minU = float.MaxValue, maxU = float.MinValue, minV = float.MaxValue, maxV = float.MinValue;
            foreach (var uv in g.UVs) {
                if (uv.X < minU) minU = uv.X;
                if (uv.X > maxU) maxU = uv.X;
                if (uv.Y < minV) minV = uv.Y;
                if (uv.Y > maxV) maxV = uv.Y;
            }
            Console.WriteLine($"  Group {gi} UV range: U=[{minU:F3}, {maxU:F3}], V=[{minV:F3}, {maxV:F3}]");
            if (g.VertexColors.Count > 0) {
                uint minC = uint.MaxValue, maxC = uint.MinValue;
                foreach (var c in g.VertexColors) {
                    if (c < minC) minC = c;
                    if (c > maxC) maxC = c;
                }
                Console.WriteLine($"  Group {gi} Color range: 0x{minC:X8} to 0x{maxC:X8}");
            }
        }
    }
}
