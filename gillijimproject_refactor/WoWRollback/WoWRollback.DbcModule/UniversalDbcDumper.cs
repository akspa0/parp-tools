using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using DBCD;
using DBCD.IO;
using DBCD.Providers;

namespace WoWRollback.DbcModule;

/// <summary>
/// Dumps ALL DBC files from a directory to JSON format for exploratory analysis.
/// Provides comprehensive data access without needing to re-decode later.
/// </summary>
public sealed class UniversalDbcDumper
{
    private readonly string _dbdDir;
    private readonly string _locale;

    public UniversalDbcDumper(string dbdDir, string locale = "enUS")
    {
        _dbdDir = dbdDir ?? throw new ArgumentNullException(nameof(dbdDir));
        _locale = locale;
    }

    /// <summary>
    /// Dumps all .dbc files from source directory to JSON.
    /// Each DBC becomes a JSON file with metadata and all records.
    /// </summary>
    /// <param name="buildVersion">Build version string (e.g., "0.5.3")</param>
    /// <param name="sourceDbcDir">Directory containing .dbc files</param>
    /// <param name="outputDir">Output directory for JSON files</param>
    /// <returns>Result containing list of dumped files or error message</returns>
    public DumpAllDbcsResult DumpAll(
        string buildVersion,
        string sourceDbcDir,
        string outputDir)
    {
        try
        {
            if (!Directory.Exists(sourceDbcDir))
            {
                return new DumpAllDbcsResult(
                    Success: false,
                    ErrorMessage: $"Source directory not found: {sourceDbcDir}",
                    DumpedFiles: Array.Empty<string>());
            }

            Directory.CreateDirectory(outputDir);

            var dbcFiles = Directory.GetFiles(sourceDbcDir, "*.dbc", SearchOption.TopDirectoryOnly);
            if (dbcFiles.Length == 0)
            {
                return new DumpAllDbcsResult(
                    Success: false,
                    ErrorMessage: $"No .dbc files found in: {sourceDbcDir}",
                    DumpedFiles: Array.Empty<string>());
            }

            var dumpedFiles = new List<string>();
            var errors = new List<string>();

            var dbdProvider = new FilesystemDBDProvider(_dbdDir);
            var dbcProvider = new FilesystemDBCProvider(sourceDbcDir, useCache: true);

            foreach (var dbcPath in dbcFiles)
            {
                var dbcName = Path.GetFileNameWithoutExtension(dbcPath);
                
                try
                {
                    // Load DBC using DBCD library
                    var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);
                    
                    // Try with locale first, fall back to Locale.None if it fails
                    IDBCDStorage storage;
                    try
                    {
                        var locale = Enum.TryParse<Locale>(_locale, ignoreCase: true, out var loc) ? loc : Locale.EnUS;
                        storage = dbcd.Load(dbcName, buildVersion, locale);
                    }
                    catch
                    {
                        storage = dbcd.Load(dbcName, buildVersion, Locale.None);
                    }
                    
                    // Convert to list of dictionaries for JSON serialization
                    var records = new List<Dictionary<string, object?>>();
                    
                    // Get available columns from storage
                    var columns = storage.AvailableColumns;
                    
                    // Cast to specific interface to avoid ambiguity
                    foreach (var kvp in (IEnumerable<KeyValuePair<int, DBCDRow>>)storage)
                    {
                        var row = kvp.Value;
                        var record = new Dictionary<string, object?>();
                        
                        // Add ID as first field
                        record["ID"] = kvp.Key;
                        
                        // Use DBCD's indexer to get actual field values
                        foreach (var column in columns)
                        {
                            try
                            {
                                var value = row[column];
                                
                                // Convert arrays to lists for better JSON serialization
                                if (value != null && value.GetType().IsArray)
                                {
                                    var array = (Array)value;
                                    var list = new List<object?>();
                                    foreach (var item in array)
                                    {
                                        list.Add(item);
                                    }
                                    record[column] = list;
                                }
                                else
                                {
                                    record[column] = value;
                                }
                            }
                            catch
                            {
                                // Skip columns that can't be read
                                record[column] = null;
                            }
                        }
                        
                        records.Add(record);
                    }

                    // Write to JSON
                    var safeVersion = buildVersion.Replace('.', '_');
                    var jsonPath = Path.Combine(outputDir, $"{dbcName}_{safeVersion}.json");
                    
                    var jsonData = new
                    {
                        dbc = dbcName,
                        build = buildVersion,
                        recordCount = records.Count,
                        generatedAt = DateTime.UtcNow.ToString("O"),
                        records = records
                    };
                    
                    var options = new JsonSerializerOptions 
                    { 
                        WriteIndented = true,
                        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never
                    };
                    
                    File.WriteAllText(jsonPath, JsonSerializer.Serialize(jsonData, options));

                    dumpedFiles.Add(jsonPath);
                }
                catch (Exception ex)
                {
                    errors.Add($"{dbcName}: {ex.Message}");
                    // Continue with other DBCs even if one fails
                }
            }

            if (dumpedFiles.Count == 0)
            {
                return new DumpAllDbcsResult(
                    Success: false,
                    ErrorMessage: $"No DBCs successfully dumped. Errors: {string.Join("; ", errors)}",
                    DumpedFiles: Array.Empty<string>());
            }

            var errorSummary = errors.Count > 0 
                ? $" Note: {errors.Count} files failed - {string.Join(", ", errors.Take(3))}{(errors.Count > 3 ? "..." : "")}"
                : null;

            return new DumpAllDbcsResult(
                Success: true,
                ErrorMessage: errorSummary,
                DumpedFiles: dumpedFiles.ToArray());
        }
        catch (Exception ex)
        {
            return new DumpAllDbcsResult(
                Success: false,
                ErrorMessage: $"DBC dump failed: {ex.Message}",
                DumpedFiles: Array.Empty<string>());
        }
    }
}

/// <summary>
/// Result of dumping all DBCs to JSON.
/// </summary>
public sealed record DumpAllDbcsResult(
    bool Success,
    string? ErrorMessage,
    IReadOnlyList<string> DumpedFiles);
