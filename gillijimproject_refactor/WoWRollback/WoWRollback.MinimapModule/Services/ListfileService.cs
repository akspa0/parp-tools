using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using WoWRollback.Core.Services.Assets;

namespace WoWRollback.MinimapModule.Services;

public class ListfileService
{
    private readonly ILogger<ListfileService>? _logger;
    private ListfileIndex? _index;

    public ListfileService(ILogger<ListfileService>? logger = null)
    {
        _logger = logger;
    }

    public void Load(string path)
    {
        try 
        {
            _logger?.LogInformation("Loading listfile from {Path}...", path);
            _index = ListfileIndex.Load(path);
            _logger?.LogInformation("Listfile loaded.");
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to load listfile from {Path}", path);
        }
    }

    public string? Resolve(uint fdid)
    {
        if (_index != null && _index.TryGetPathByFdid(fdid, out var path))
        {
            return path;
        }
        return null; // Return null if not found
    }
}
