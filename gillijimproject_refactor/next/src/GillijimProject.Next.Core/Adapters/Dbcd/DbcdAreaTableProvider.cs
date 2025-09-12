using DBCD; // wow.tools.local DBCD
using DBCD.Providers;
using System;
using System.IO;

namespace GillijimProject.Next.Core.Adapters.Dbcd;

/// <summary>
/// Provides access to AreaTable and Map DBCD data for Alpha and LK.
/// </summary>
public sealed class DbcdAreaTableProvider
{
    // [PORT] Uses FilesystemDBCProvider (local files) + GithubDBDProvider (definitions).
    // TODO(PORT): Consider allowing a local DBD cache path injection if needed.

    public DbcdAreaTableProvider(
        string alphaDbcPath,
        string lkDbcPath,
        string? alphaBuild = null,
        string? lkBuild = "3.3.5.12340",
        bool cacheDbc = false,
        bool cacheDbd = true)
    {
        AlphaDbcPath = alphaDbcPath;
        LkDbcPath = lkDbcPath;
        AlphaBuild = alphaBuild; // Often unknown for alpha; null lets DBCD pick a matching layout if available
        LkBuild = lkBuild;       // WotLK default build; can be overridden via CLI
        _cacheDbc = cacheDbc;
        _cacheDbd = cacheDbd;
    }

    public string AlphaDbcPath { get; }
    public string LkDbcPath { get; }
    public string? AlphaBuild { get; }
    public string? LkBuild { get; }

    private readonly bool _cacheDbc;
    private readonly bool _cacheDbd;

    private IDBCDStorage? _alphaAreaTable;
    private IDBCDStorage? _lkAreaTable;
    private IDBCDStorage? _alphaMapTable;
    private IDBCDStorage? _lkMapTable;

    private static readonly string[] AlphaBuildCandidates = new[]
    {
        // Known early builds; adjust as needed
        "0.5.5.3494",
        "0.5.3.3368",
        "0.5.3",
    };

    /// <summary>
    /// Ensure both Alpha and LK AreaTable storages are loaded.
    /// </summary>
    public void EnsureLoaded()
    {
        _alphaAreaTable ??= LoadAreaTable(AlphaDbcPath, AlphaBuild, isAlpha: true);
        _lkAreaTable ??= LoadAreaTable(LkDbcPath, LkBuild, isAlpha: false);
        _alphaMapTable ??= LoadMapTable(AlphaDbcPath, AlphaBuild, isAlpha: true);
        _lkMapTable ??= LoadMapTable(LkDbcPath, LkBuild, isAlpha: false);
    }

    public IDBCDStorage GetAlphaAreaTable()
    {
        EnsureLoaded();
        return _alphaAreaTable!;
    }

    public IDBCDStorage GetLkAreaTable()
    {
        EnsureLoaded();
        return _lkAreaTable!;
    }

    public IDBCDStorage GetAlphaMapTable()
    {
        EnsureLoaded();
        return _alphaMapTable!;
    }

    public IDBCDStorage GetLkMapTable()
    {
        EnsureLoaded();
        return _lkMapTable!;
    }

    public int AlphaRowCount => _alphaAreaTable?.Count ?? 0;
    public int LkRowCount => _lkAreaTable?.Count ?? 0;
    public int AlphaMapRowCount => _alphaMapTable?.Count ?? 0;
    public int LkMapRowCount => _lkMapTable?.Count ?? 0;

    private IDBCDStorage LoadAreaTable(string dbcFilePath, string? build, bool isAlpha)
    {
        if (string.IsNullOrWhiteSpace(dbcFilePath))
            throw new IOException("DBC path must not be empty.");

        var dir = Path.GetDirectoryName(dbcFilePath);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
            throw new DirectoryNotFoundException($"DBC directory not found: {dir}");

        var dbcProvider = new FilesystemDBCProvider(dir, _cacheDbc);
        var dbdProvider = new GithubDBDProvider(_cacheDbd);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        // Try requested build first.
        try
        {
            return dbcd.Load("AreaTable", build, Locale.None);
        }
        catch when (isAlpha)
        {
            // Fallthrough to alpha probes below
        }

        if (isAlpha)
        {
            foreach (var candidate in AlphaBuildCandidates)
            {
                try
                {
                    return dbcd.Load("AreaTable", candidate, Locale.None);
                }
                catch
                {
                    // keep trying
                }
            }
        }

        // If we reach here, attempt one last time without build (may still work for some layouts)
        return dbcd.Load("AreaTable", null, Locale.None);
    }

    private IDBCDStorage LoadMapTable(string dbcFilePath, string? build, bool isAlpha)
    {
        if (string.IsNullOrWhiteSpace(dbcFilePath))
            throw new IOException("DBC path must not be empty.");

        var dir = Path.GetDirectoryName(dbcFilePath);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
            throw new DirectoryNotFoundException($"DBC directory not found: {dir}");

        var dbcProvider = new FilesystemDBCProvider(dir, _cacheDbc);
        var dbdProvider = new GithubDBDProvider(_cacheDbd);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        try
        {
            return dbcd.Load("Map", build, Locale.None);
        }
        catch when (isAlpha)
        {
            // Fallthrough to alpha probes below
        }

        if (isAlpha)
        {
            foreach (var candidate in AlphaBuildCandidates)
            {
                try
                {
                    return dbcd.Load("Map", candidate, Locale.None);
                }
                catch
                {
                    // keep trying
                }
            }
        }

        return dbcd.Load("Map", null, Locale.None);
    }
}
