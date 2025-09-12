using System;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Export;

public sealed class MissingAssetsLogger : IDisposable
{
    private readonly string _path;
    private bool _initialized;
    private StreamWriter? _writer;

    public MissingAssetsLogger(string path)
    {
        _path = path;
    }

    public void Write(MissingAssetRecord rec)
    {
        Ensure();
        _writer!.WriteLine(string.Join(',',
            rec.Type,
            Csv(rec.Original),
            Csv(rec.MapName),
            rec.TileX,
            rec.TileY,
            rec.UniqueId?.ToString() ?? string.Empty));
    }

    private void Ensure()
    {
        if (_initialized) return;
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        _writer = new StreamWriter(_path, append: false);
        _writer.WriteLine("type,original,map,tile_x,tile_y,unique_id");
        _writer.Flush();
        _initialized = true;
    }

    private static string Csv(string s)
    {
        if (s.Contains('"') || s.Contains(','))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }

    public void Dispose()
    {
        _writer?.Dispose();
    }
}

public sealed class MissingAssetRecord
{
    public required string Type { get; init; }
    public required string Original { get; init; }
    public required string MapName { get; init; }
    public required int TileX { get; init; }
    public required int TileY { get; init; }
    public int? UniqueId { get; init; }
}
