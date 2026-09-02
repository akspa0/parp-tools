using DBCD;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;

/// <summary>
/// General-purpose DBC/DB2 dump: open any table from a client through DBCD + WoWDBDefs and print
/// its header, columns and rows.
/// </summary>
/// <remarks>
/// Exists because every phase/liquid/lighting question so far has been settled by looking at the
/// table rather than at documentation, and each time that required writing a throwaway probe.
/// <para>
/// <b>Rows are keyed by the ID column, not by <c>DBCDRow.ID</c></b>, which is a positional index for
/// these WDB2 tables: on MoP 5.0.1 <c>LiquidObject</c> loads as keys 1..1244 while the row keyed 42
/// carries <c>ID=316</c>. Keying on the positional value silently resolves the wrong row for every
/// sparse id.
/// </para>
/// </remarks>
internal static class DbcDumpSupport
{
    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client");
        string? tableName = GetOption(args, "--table");
        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(tableName))
        {
            Console.Error.WriteLine("Usage: dbc dump --client <client-dir> --table <name> [--build <version>] [--limit <n>] [--columns a,b,c] [--where col=value] [--id <n>]");
            Environment.ExitCode = 1;
            return;
        }

        string buildVersion = GetOption(args, "--build") ?? "5.0.1.15464";
        string? definitionsDir = GetOption(args, "--defs") ?? ResolveDefinitionsDirectory();
        int limit = int.TryParse(GetOption(args, "--limit"), out int parsedLimit) ? parsedLimit : 40;
        string? columnFilter = GetOption(args, "--columns");
        string? whereClause = GetOption(args, "--where");
        string? singleId = GetOption(args, "--id");

        if (string.IsNullOrWhiteSpace(definitionsDir) || !Directory.Exists(definitionsDir))
        {
            Console.Error.WriteLine($"WoWDBDefs definitions not found (looked for '{definitionsDir ?? "<null>"}'). Pass --defs.");
            Environment.ExitCode = 1;
            return;
        }

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [clientRoot], new ArchiveCatalogBootstrapOptions());

        ArchiveReaderDbcProvider provider = new(archiveCatalog);
        IDBCDStorage storage;
        try
        {
            storage = DbcTableLoader.Load(provider, definitionsDir, buildVersion, tableName);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to open '{tableName}' for build {buildVersion}: {ex.Message}");
            Environment.ExitCode = 1;
            return;
        }

        string? idColumn = DbcTableLoader.DetectColumn(storage, DbcTableLoader.IdColumns);
        var rowsById = new SortedDictionary<int, DBCDRow>();
        foreach (DBCDRow row in storage.Values)
            rowsById[DbcTableLoader.ResolveRowId(row, idColumn)] = row;

        string[] columns = string.IsNullOrWhiteSpace(columnFilter)
            ? storage.AvailableColumns
            : columnFilter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        Console.WriteLine($"table={tableName} build={buildVersion} client={clientRoot}");
        Console.WriteLine($"rows={rowsById.Count} idColumn='{idColumn ?? "<none, using positional>"}' "
            + $"realIdRange={(rowsById.Count == 0 ? "-" : rowsById.Keys.First() + ".." + rowsById.Keys.Last())}");
        Console.WriteLine($"columns: {string.Join(", ", storage.AvailableColumns)}");
        Console.WriteLine();

        (string Column, string Value)? filter = null;
        if (!string.IsNullOrWhiteSpace(whereClause))
        {
            int equals = whereClause.IndexOf('=');
            if (equals > 0)
                filter = (whereClause[..equals].Trim(), whereClause[(equals + 1)..].Trim());
        }

        int printed = 0;
        int matched = 0;
        foreach ((int id, DBCDRow row) in rowsById)
        {
            if (!string.IsNullOrWhiteSpace(singleId) && id.ToString() != singleId)
                continue;

            if (filter is { } f)
            {
                string? actual = ReadField(row, f.Column);
                if (!string.Equals(actual, f.Value, StringComparison.OrdinalIgnoreCase))
                    continue;
            }

            matched++;
            if (printed >= limit)
                continue;

            List<string> parts = [];
            foreach (string column in columns)
                parts.Add($"{column}={ReadField(row, column) ?? "?"}");

            Console.WriteLine($"[{id}] {string.Join(" ", parts)}");
            printed++;
        }

        Console.WriteLine();
        Console.WriteLine(matched > printed
            ? $"matched={matched}, printed={printed} (raise --limit to see the rest)"
            : $"matched={matched}");
    }

    private static string? ReadField(DBCDRow row, string column)
    {
        try
        {
            object value = row[column];
            if (value is Array array)
            {
                List<string> items = [];
                foreach (object? item in array)
                    items.Add(item?.ToString() ?? "null");
                return "[" + string.Join(",", items) + "]";
            }

            return value?.ToString();
        }
        catch
        {
            return null;
        }
    }

    private static string? ResolveDefinitionsDirectory()
    {
        DirectoryInfo? dir = new(AppDomain.CurrentDomain.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "libs", "wowdev", "WoWDBDefs", "definitions");
            if (Directory.Exists(candidate))
                return candidate;

            dir = dir.Parent;
        }

        return null;
    }

    private static string? GetOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }

        return null;
    }
}
