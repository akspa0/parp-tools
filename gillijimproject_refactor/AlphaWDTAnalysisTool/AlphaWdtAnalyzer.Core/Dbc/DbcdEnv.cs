using System.IO;

namespace AlphaWdtAnalyzer.Core.Dbc;

public static class DbcdEnv
{
    public static bool UseDbcdActive
    {
        get
        {
#if USE_DBCD
            return true;
#else
            return false;
#endif
        }
    }

    // Accept either the definitions folder or its parent (containing a 'definitions' child)
    public static string? ResolveDefinitionsDir(string? dbdDir)
    {
        if (string.IsNullOrWhiteSpace(dbdDir)) return null;
        if (Directory.Exists(dbdDir))
        {
            // If this path itself is the definitions folder
            if (File.Exists(Path.Combine(dbdDir, "AreaTable.dbd")) ||
                File.Exists(Path.Combine(dbdDir, "AreaTable.db2.dbd")))
            {
                return dbdDir;
            }
            // If a 'definitions' subdirectory exists, use it
            var def = Path.Combine(dbdDir, "definitions");
            if (Directory.Exists(def)) return def;
        }
        return null;
    }
}
