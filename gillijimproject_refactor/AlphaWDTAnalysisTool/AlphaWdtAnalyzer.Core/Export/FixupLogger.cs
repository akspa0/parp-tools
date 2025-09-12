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

    public FixupLogger(string path)
    {
        _path = path;
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
        _writer!.WriteLine(string.Join(',',
            rec.Type,
            Csv(rec.Original),
            Csv(rec.Resolved),
            Csv(rec.Method)));
    }

    private void Ensure()
    {
        if (_initialized) return;
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        _writer = new StreamWriter(_path, append: false);
        _writer.WriteLine("type,original,resolved,method");
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

public sealed class FixupRecord
{
    public required string Type { get; init; }
    public required string Original { get; init; }
    public required string Resolved { get; init; }
    public required string Method { get; init; }
}
