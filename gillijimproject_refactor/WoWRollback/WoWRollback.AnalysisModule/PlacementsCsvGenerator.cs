using System.Globalization;
using System.Text;
using AlphaWdtAnalyzer.Core;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Exports placements (MDDF/MODF) to a normalized CSV consumable by the Viewer.
/// Source is AnalysisIndex (Placements collection).
/// </summary>
public sealed class PlacementsCsvGenerator
{
    public string Generate(AnalysisIndex analysisIndex, string mapName, string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        var outPath = Path.Combine(outputDir, $"{mapName}_placements.csv");

        var sb = new StringBuilder();
        sb.AppendLine("map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set");

        foreach (var p in analysisIndex.Placements)
        {
            var typeStr = p.Type switch
            {
                AssetType.MdxOrM2 => "M2",
                AssetType.Wmo => "WMO",
                _ => "Unknown"
            };

            // Use invariant culture for floats
            string F(float v) => v.ToString("G9", CultureInfo.InvariantCulture);
            string I(int? v) => v.HasValue ? v.Value.ToString(CultureInfo.InvariantCulture) : string.Empty;

            sb.AppendLine(string.Join(",",
                Csv(mapName),
                p.TileX.ToString(CultureInfo.InvariantCulture),
                p.TileY.ToString(CultureInfo.InvariantCulture),
                Csv(typeStr),
                Csv(p.AssetPath ?? string.Empty),
                I(p.UniqueId),
                F(p.WorldX),
                F(p.WorldY),
                F(p.WorldZ),
                F(p.RotationX),
                F(p.RotationY),
                F(p.RotationZ),
                F(p.Scale),
                p.DoodadSet.ToString(CultureInfo.InvariantCulture),
                p.NameSet.ToString(CultureInfo.InvariantCulture)
            ));
        }

        File.WriteAllText(outPath, sb.ToString());
        return outPath;

        static string Csv(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            if (s.Contains('"') || s.Contains(','))
            {
                return '"' + s.Replace("\"", "\"\"") + '"';
            }
            return s;
        }
    }
}
