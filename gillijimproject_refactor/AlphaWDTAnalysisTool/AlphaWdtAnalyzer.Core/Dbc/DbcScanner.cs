using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Dbc;

public sealed class DbcTableResult
{
    public required string Name { get; init; }
    public required string Path { get; init; }
    public required RawDbcTable Table { get; init; }
}

public sealed class DbcScanResult
{
    public List<DbcTableResult> Tables { get; } = new();
}

public static class DbcScanner
{
    public static DbcScanResult Scan(string dbcDir)
    {
        if (!Directory.Exists(dbcDir)) throw new DirectoryNotFoundException(dbcDir);
        var res = new DbcScanResult();
        var files = Directory.EnumerateFiles(dbcDir, "*.dbc", SearchOption.TopDirectoryOnly)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (var file in files)
        {
            try
            {
                var table = RawDbcParser.Parse(file);
                var name = System.IO.Path.GetFileNameWithoutExtension(file);
                res.Tables.Add(new DbcTableResult
                {
                    Name = name,
                    Path = file,
                    Table = table
                });
            }
            catch (Exception)
            {
                // Ignore files we can't parse in this first pass; can log later
            }
        }

        return res;
    }
}
