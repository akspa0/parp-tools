using System;
using System.Collections.Generic;
using System.IO;

namespace WoWRollback.Core.Services.Archive
{
    public interface IArchiveSource : IDisposable
    {
        bool FileExists(string virtualPath);
        Stream OpenFile(string virtualPath);
        IEnumerable<string> EnumerateFiles(string pattern = "*");
    }
}
