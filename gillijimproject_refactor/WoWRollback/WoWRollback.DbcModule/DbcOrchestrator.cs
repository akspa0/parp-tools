using DBCTool.V2.Cli;

namespace WoWRollback.DbcModule;

/// <summary>
/// High-level orchestrator for DBC operations, wrapping DBCTool.V2 as a library API.
/// Extracts core functionality from CLI commands for programmatic use.
/// </summary>
public sealed class DbcOrchestrator
{
    private readonly string _dbdDir;
    private readonly string _locale;

    /// <summary>
    /// Creates a new DBC orchestrator.
    /// </summary>
    /// <param name="dbdDir">Path to WoWDBDefs definitions directory</param>
    /// <param name="locale">DBC locale (default: enUS)</param>
    public DbcOrchestrator(string dbdDir, string locale = "enUS")
    {
        if (string.IsNullOrWhiteSpace(dbdDir))
            throw new ArgumentException("DBD directory path is required", nameof(dbdDir));

        _dbdDir = dbdDir;
        _locale = locale;
    }

    /// <summary>
    /// Dumps AreaTable.dbc files to CSV for source and target (3.3.5) builds.
    /// </summary>
    /// <param name="srcAlias">Source version alias (e.g., "0.5.3")</param>
    /// <param name="srcDbcDir">Path to source DBC directory</param>
    /// <param name="tgtDbcDir">Path to 3.3.5 DBC directory</param>
    /// <param name="outDir">Output directory for CSV files</param>
    /// <returns>Result containing paths to generated CSVs</returns>
    public DbcDumpResult DumpAreaTables(
        string srcAlias,
        string srcDbcDir,
        string tgtDbcDir,
        string outDir)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(srcAlias))
                throw new ArgumentException("Source alias is required", nameof(srcAlias));
            if (string.IsNullOrWhiteSpace(srcDbcDir))
                throw new ArgumentException("Source DBC directory is required", nameof(srcDbcDir));
            if (string.IsNullOrWhiteSpace(tgtDbcDir))
                throw new ArgumentException("Target DBC directory is required", nameof(tgtDbcDir));
            if (string.IsNullOrWhiteSpace(outDir))
                throw new ArgumentException("Output directory is required", nameof(outDir));

            if (!Directory.Exists(srcDbcDir))
                return new DbcDumpResult(
                    string.Empty,
                    string.Empty,
                    Success: false,
                    ErrorMessage: $"Source DBC directory not found: {srcDbcDir}");

            if (!Directory.Exists(tgtDbcDir))
                return new DbcDumpResult(
                    string.Empty,
                    string.Empty,
                    Success: false,
                    ErrorMessage: $"Target DBC directory not found: {tgtDbcDir}");

            // Create output directory
            Directory.CreateDirectory(outDir);

            // Prepare inputs for DumpAreaCommand
            var inputs = new List<(string build, string dir)>
            {
                (srcAlias, srcDbcDir),
                ("3.3.5", tgtDbcDir)
            };

            // Use existing CLI command logic
            var command = new DumpAreaCommand();
            var exitCode = command.Run(_dbdDir, outDir, _locale, inputs);

            if (exitCode != 0)
            {
                return new DbcDumpResult(
                    string.Empty,
                    string.Empty,
                    Success: false,
                    ErrorMessage: $"DumpAreaCommand returned exit code {exitCode}");
            }

            // Compute expected output paths
            var srcCsvPath = Path.Combine(outDir, srcAlias, "raw", $"AreaTable_{srcAlias.Replace('.', '_')}.csv");
            var tgtCsvPath = Path.Combine(outDir, srcAlias, "raw", "AreaTable_3_3_5.csv");

            return new DbcDumpResult(
                SrcCsvPath: srcCsvPath,
                TgtCsvPath: tgtCsvPath,
                Success: true);
        }
        catch (Exception ex)
        {
            return new DbcDumpResult(
                string.Empty,
                string.Empty,
                Success: false,
                ErrorMessage: $"DBC dump failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Generates crosswalk mapping files between source and 3.3.5 AreaTable/Map.
    /// </summary>
    /// <param name="srcAlias">Source version alias (e.g., "0.5.3")</param>
    /// <param name="srcDbcDir">Path to source DBC directory</param>
    /// <param name="tgtDbcDir">Path to 3.3.5 DBC directory</param>
    /// <param name="outDir">Output directory for crosswalk files</param>
    /// <param name="chainVia060">Whether to chain through 0.6.0 (optional)</param>
    /// <returns>Result containing paths to generated crosswalk artifacts</returns>
    public CrosswalkResult GenerateCrosswalks(
        string srcAlias,
        string srcDbcDir,
        string tgtDbcDir,
        string outDir,
        bool chainVia060 = false)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(srcAlias))
                throw new ArgumentException("Source alias is required", nameof(srcAlias));
            if (string.IsNullOrWhiteSpace(srcDbcDir))
                throw new ArgumentException("Source DBC directory is required", nameof(srcDbcDir));
            if (string.IsNullOrWhiteSpace(tgtDbcDir))
                throw new ArgumentException("Target DBC directory is required", nameof(tgtDbcDir));
            if (string.IsNullOrWhiteSpace(outDir))
                throw new ArgumentException("Output directory is required", nameof(outDir));

            if (!Directory.Exists(srcDbcDir))
                return new CrosswalkResult(
                    string.Empty,
                    string.Empty,
                    string.Empty,
                    Success: false,
                    ErrorMessage: $"Source DBC directory not found: {srcDbcDir}");

            if (!Directory.Exists(tgtDbcDir))
                return new CrosswalkResult(
                    string.Empty,
                    string.Empty,
                    string.Empty,
                    Success: false,
                    ErrorMessage: $"Target DBC directory not found: {tgtDbcDir}");

            // Create output directory
            Directory.CreateDirectory(outDir);

            // Prepare inputs for CompareAreaV2Command
            var inputs = new List<(string build, string dir)>
            {
                (srcAlias, srcDbcDir),
                ("3.3.5", tgtDbcDir)
            };

            // Use existing CLI command logic
            var command = new CompareAreaV2Command();
            var exitCode = command.Run(_dbdDir, outDir, _locale, inputs, chainVia060);

            if (exitCode != 0)
            {
                return new CrosswalkResult(
                    string.Empty,
                    string.Empty,
                    string.Empty,
                    Success: false,
                    ErrorMessage: $"CompareAreaV2Command returned exit code {exitCode}");
            }

            // Compute expected output paths
            var mapsJsonPath = Path.Combine(outDir, srcAlias, "maps.json");
            var crosswalkV2Dir = Path.Combine(outDir, srcAlias, "compare", "v2");
            var crosswalkV3Dir = Path.Combine(outDir, srcAlias, "compare", "v3");

            return new CrosswalkResult(
                MapsJsonPath: mapsJsonPath,
                CrosswalkV2Dir: crosswalkV2Dir,
                CrosswalkV3Dir: crosswalkV3Dir,
                Success: true);
        }
        catch (Exception ex)
        {
            return new CrosswalkResult(
                string.Empty,
                string.Empty,
                string.Empty,
                Success: false,
                ErrorMessage: $"Crosswalk generation failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Dumps ALL DBC files to JSON for comprehensive data access.
    /// Enables exploration of all available data without needing to re-decode.
    /// </summary>
    /// <param name="buildVersion">Build version string (e.g., "0.5.3")</param>
    /// <param name="sourceDbcDir">Directory containing .dbc files</param>
    /// <param name="outputDir">Output directory for JSON files</param>
    /// <returns>Result containing list of dumped files</returns>
    public DumpAllDbcsResult DumpAllDbcs(
        string buildVersion,
        string sourceDbcDir,
        string outputDir)
    {
        // Convert version alias to canonical build format (e.g., "0.5.3" → "0.5.3.3368")
        var canonicalBuild = CanonicalizeBuild(buildVersion);
        
        var dumper = new UniversalDbcDumper(_dbdDir, _locale);
        return dumper.DumpAll(canonicalBuild, sourceDbcDir, outputDir);
    }

    /// <summary>
    /// Converts version alias to canonical DBCD build format.
    /// </summary>
    private static string CanonicalizeBuild(string alias)
    {
        return alias switch
        {
            "0.5.3" => "0.5.3.3368",
            "0.5.5" => "0.5.5.3494",
            "0.6.0" => "0.6.0.3592",
            "3.3.5" => "3.3.5.12340",
            _ => alias
        };
    }
}
