using System;
using System.Collections.Generic;

namespace WoWRollback.Core.Services.Assets
{
    public sealed class ListfileCatalog
    {
        private readonly Dictionary<string, ListfileIndex> _byAlias = new(StringComparer.OrdinalIgnoreCase);

        public void Add(string alias, ListfileIndex index)
        {
            if (string.IsNullOrWhiteSpace(alias)) throw new ArgumentException("alias");
            if (index == null) throw new ArgumentNullException(nameof(index));
            _byAlias[alias] = index;
        }

        public bool TryGet(string alias, out ListfileIndex index)
        {
            return _byAlias.TryGetValue(alias, out index!);
        }
    }
}
