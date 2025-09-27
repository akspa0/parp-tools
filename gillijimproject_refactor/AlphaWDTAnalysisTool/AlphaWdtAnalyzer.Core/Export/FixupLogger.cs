using System;
using System.Globalization;
using System.IO;
using System.Collections.Generic;

namespace AlphaWdtAnalyzer.Core.Export;

public sealed class FixupLogger : IDisposable
{
    private readonly string _path;
    private bool _initialized;
    private StreamWriter? _writer;
    private readonly HashSet<string> _seen = new(StringComparer.OrdinalIgnoreCase);
    // Per-tile context and counters
    private string? _curMap;
    private int _curTileX;
    private int _curTileY;
    private int _tileFuzzy;
    private int _tileCapacity;
    private int _tileOverflow;

    public FixupLogger(string path)
    {
        _path = path;
    }

    public void BeginTile(string mapName, int tileX, int tileY)
    {
        _curMap = mapName; _curTileX = tileX; _curTileY = tileY;
        _tileFuzzy = 0; _tileCapacity = 0; _tileOverflow = 0;
    }

    public (int fuzzy, int capacity, int overflow) EndTile()
    {
        var r = (_tileFuzzy, _tileCapacity, _tileOverflow);
        _curMap = null; // clear context after tile finishes
        return r;
    }

    public void Write(FixupRecord rec)
    {
        // Record actionable events:
        // - fuzzy:* (suggested replacements)
        // - overflow_* (too long to fit slot)
        // - capacity_* (fallback chosen due to capacity)
        var m = rec.Method ?? string.Empty;
        if (!(m.StartsWith("fuzzy", StringComparison.OrdinalIgnoreCase)
            || m.StartsWith("overflow", StringComparison.OrdinalIgnoreCase)
            || m.StartsWith("capacity", StringComparison.OrdinalIgnoreCase)))
        {
            return;
        }

        var key = $"{rec.Type}|{rec.Original}|{rec.Resolved}|{rec.Method}";
        if (_seen.Contains(key)) return;
        _seen.Add(key);

        Ensure();
        // Update tile counters if we have an active tile
        if (!string.IsNullOrWhiteSpace(_curMap))
        {
            if (m.StartsWith("fuzzy", StringComparison.OrdinalIgnoreCase)) _tileFuzzy++;
            else if (m.StartsWith("capacity", StringComparison.OrdinalIgnoreCase)) _tileCapacity++;
            else if (m.StartsWith("overflow", StringComparison.OrdinalIgnoreCase)) _tileOverflow++;
        }

        _writer!.WriteLine(string.Join(',',
            rec.Type,
            Csv(rec.Original),
            Csv(rec.Resolved),
            Csv(rec.Method),
            Csv(_curMap),
            _curTileX.ToString(CultureInfo.InvariantCulture),
            _curTileY.ToString(CultureInfo.InvariantCulture)));
    }

    private void Ensure()
    {
        if (_initialized) return;
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        _writer = new StreamWriter(_path, append: false);
        _writer.WriteLine("type,original,resolved,method,map,tile_x,tile_y");
        _writer.Flush();
        _initialized = true;
    }

    private static string Csv(string? s)
    {
        if (string.IsNullOrEmpty(s)) return string.Empty;
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

public sealed class FixupRecord
{
    public required string Type { get; init; }
    public required string Original { get; init; }
    public required string Resolved { get; init; }
    public required string Method { get; init; }
}
