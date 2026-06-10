using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using PM4NextExporter.Model;
using ParpToolbox.Services.PM4;

namespace PM4NextExporter.Assembly
{
    internal sealed class MsurCompositeKeyAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var output = new List<AssembledObject>();
            if (scene == null) return output;

            var pm4 = LoadPm4Scene(scene.SourcePath, options);
            if (pm4 == null)
                return output;

            var msurObjects = ParpToolbox.Services.PM4.Pm4MsurObjectAssembler.AssembleObjectsByCompositeKey(pm4);
            foreach (var mo in msurObjects)
            {
                // Parse composite key from ObjectType pattern: "Composite_XXXXXXXX"
                uint ck = 0;
                string ckHex = "00000000";
                if (!string.IsNullOrEmpty(mo.ObjectType) && mo.ObjectType.StartsWith("Composite_", StringComparison.OrdinalIgnoreCase))
                {
                    var hex = mo.ObjectType.Substring("Composite_".Length);
                    if (uint.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var val))
                    {
                        ck = val;
                        ckHex = hex.ToUpperInvariant();
                    }
                }

                var name = ck != 0 ? $"MSUR_CK_{ckHex}" : (mo.ObjectType ?? "MSUR_Composite");
                var obj = new AssembledObject(name, mo.Vertices, mo.Triangles);
                obj.Meta["msur.ck"] = ck.ToString(CultureInfo.InvariantCulture);
                obj.Meta["msur.ck.hex"] = ckHex;
                obj.Meta["msur.objectType"] = mo.ObjectType ?? string.Empty;
                obj.Meta["placement.count"] = mo.PlacementCount.ToString(CultureInfo.InvariantCulture);
                obj.Meta["placement.center"] = string.Format(CultureInfo.InvariantCulture, "{0:F6},{1:F6},{2:F6}", mo.PlacementCenter.X, mo.PlacementCenter.Y, mo.PlacementCenter.Z);
                obj.Meta["msur.surfaceCount"] = mo.SurfaceCount.ToString(CultureInfo.InvariantCulture);
                obj.Meta["vertex.count"] = mo.VertexCount.ToString(CultureInfo.InvariantCulture);
                output.Add(obj);
            }

            return output;
        }

        private static ParpToolbox.Formats.PM4.Pm4Scene? LoadPm4Scene(string sourcePath, Options options)
        {
            if (string.IsNullOrWhiteSpace(sourcePath)) return null;

            try
            {
                if (Directory.Exists(sourcePath))
                {
                    if (options.IncludeAdjacent)
                    {
                        var global = Pm4GlobalTileLoader.LoadRegion(sourcePath, "*.pm4", applyMscnRemap: !options.NoRemap);
                        return Pm4GlobalTileLoader.ToStandardScene(global);
                    }
                    var first = Directory.GetFiles(sourcePath, "*.pm4").FirstOrDefault();
                    if (first == null) return null;
                    var adapter = new Pm4Adapter();
                    return adapter.Load(first);
                }
                if (File.Exists(sourcePath))
                {
                    if (options.IncludeAdjacent)
                    {
                        var dir = Path.GetDirectoryName(sourcePath) ?? ".";
                        var global = Pm4GlobalTileLoader.LoadRegion(dir, "*.pm4", applyMscnRemap: !options.NoRemap);
                        return Pm4GlobalTileLoader.ToStandardScene(global);
                    }
                    var adapter = new Pm4Adapter();
                    return adapter.Load(sourcePath);
                }
            }
            catch (Exception ex)
            {
                System.Console.Error.WriteLine($"[MsurCompositeKeyAssembler] Failed to load Pm4Scene from '{sourcePath}': {ex.Message}");
            }

            return null;
        }
    }
}
