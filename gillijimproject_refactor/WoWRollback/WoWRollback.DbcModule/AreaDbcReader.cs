using System;
using System.Collections.Generic;
using DBCD;
using DBCD.IO;
using DBCD.Providers;

namespace WoWRollback.DbcModule;

/// <summary>
/// Reads AreaTable via DBCD and exposes name/parent dictionaries.
/// </summary>
public sealed class AreaDbcReader
{
    private readonly string _dbdDir;
    private readonly string _locale;

    public AreaDbcReader(string dbdDir, string locale = "enUS")
    {
        _dbdDir = dbdDir ?? throw new ArgumentNullException(nameof(dbdDir));
        _locale = locale;
    }

    public record Result(bool Success, string? ErrorMessage, Dictionary<int, string> NameById, Dictionary<int, int> ParentById);

    public Result ReadAreas(string buildVersion, string dbcDir)
    {
        try
        {
            if (!System.IO.Directory.Exists(dbcDir))
                return new Result(false, $"DBC dir not found: {dbcDir}", new(), new());

            var dbdProvider = new FilesystemDBDProvider(_dbdDir);
            var dbcProvider = new FilesystemDBCProvider(dbcDir, useCache: true);
            var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

            IDBCDStorage storage;
            try
            {
                var locale = Enum.TryParse<Locale>(_locale, ignoreCase: true, out var loc) ? loc : Locale.EnUS;
                storage = dbcd.Load("AreaTable", buildVersion, locale);
            }
            catch
            {
                storage = dbcd.Load("AreaTable", buildVersion, Locale.None);
            }

            var names = new Dictionary<int, string>();
            var parents = new Dictionary<int, int>();

            foreach (var row in storage.Values)
            {
                try
                {
                    var id = GetIntField(row, "ID") ?? 0;
                    if (id <= 0) continue;

                    // Name field varies by build
                    var name = GetField<string>(row, "AreaName")
                               ?? GetField<string>(row, "AreaName_lang")
                               ?? GetField<string>(row, "ZoneName")
                               ?? string.Empty;

                    // Parent field varies by build
                    var parent = GetIntField(row, "ParentAreaID")
                                 ?? GetIntField(row, "ParentAreaId")
                                 ?? GetIntField(row, "ParentAreaNum")
                                 ?? 0;

                    if (!string.IsNullOrWhiteSpace(name)) names[id] = name;
                    parents[id] = parent;
                }
                catch
                {
                    // Skip malformed rows
                }
            }

            return new Result(true, null, names, parents);
        }
        catch (Exception ex)
        {
            return new Result(false, $"Failed to read AreaTable: {ex.Message}", new(), new());
        }
    }

    private static T? GetField<T>(DBCDRow row, string fieldName)
    {
        try
        {
            var value = row[fieldName];
            if (value is T typedValue)
                return typedValue;

            if (typeof(T) == typeof(string))
                return (T)(object)(value?.ToString() ?? "");

            if (value != null)
                return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
        }
        return default;
    }

    private static int? GetIntField(DBCDRow row, string fieldName)
    {
        try
        {
            var value = row[fieldName];
            if (value is int i) return i;
            if (value is long l) return checked((int)l);
            if (value != null)
            {
                var conv = Convert.ChangeType(value, typeof(int));
                if (conv is int ci) return ci;
            }
        }
        catch { }
        return null;
    }
}
