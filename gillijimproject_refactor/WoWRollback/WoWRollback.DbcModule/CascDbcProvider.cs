using System;
using System.IO;
using DBCD.IO;
using DBCD.Providers;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.DbcModule
{
    public sealed class CascDbcProvider : IDBCProvider
    {
        private readonly IArchiveSource _src;
        public CascDbcProvider(IArchiveSource src) { _src = src; }

        public Stream StreamForTableName(string tableName, string build)
        {
            var cand = new[]
            {
                $"DBFilesClient/{tableName}.db2",
                $"DB2/{tableName}.db2",
                $"db2/{tableName}.db2",
                $"DBFilesClient/{tableName}.dbc",
                $"dbc/{tableName}.dbc"
            };
            foreach (var v in cand)
            {
                if (_src.FileExists(v)) return _src.OpenFile(v);
            }
            throw new FileNotFoundException($"{tableName} not found in CASC (DB2/DBC)");
        }
    }
}
