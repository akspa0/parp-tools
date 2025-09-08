using System;
using System.Globalization;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Export;

public sealed class FixupLogger : IDisposable
{
    private readonly string _path;
    private bool _initialized;
    private StreamWriter? _writer;

    public FixupLogger(string path)
    {
        _path = path;
    }

    public void Write(FixupRecord rec)
    {
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
