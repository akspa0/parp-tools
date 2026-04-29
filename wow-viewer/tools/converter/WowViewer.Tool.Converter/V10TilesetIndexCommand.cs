using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using SereniaBLPLib;
using WowViewer.Core.Blp;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Files;

namespace WowViewer.Tool.Converter;

public static class V10TilesetIndexCommand
{
	private static readonly Dictionary<string, string> ZoneAbbreviations = new(StringComparer.OrdinalIgnoreCase)
	{
		["BT"] = "BlackTemple", ["IC"] = "Icecrown", ["ND"] = "Northrend",
		["GH"] = "Ghostlands", ["DH"] = "Dragonblight", ["ZD"] = "Zul'Drak",
		["ITK"] = "Icecrown", ["SR"] = "StonetalonRidge", ["SM"] = "ScarletMonastery",
		["HM"] = "HillsbradFoothills", ["NG"] = "Nagrand", ["SA"] = "SilithusArea",
		["UDM"] = "Undermine", ["DUR"] = "Durotar", ["ELW"] = "Elwynn",
		["ESW"] = "EasternWeald", ["VS"] = "VioletStand", ["OG"] = "Orgrimmar",
		["CAN"] = "Caverns", ["CAV"] = "Caverns", ["DG"] = "Dagger",
		["ED"] = "ElwynnDirt", ["GSL"] = "GrizzlyHills", ["HGL"] = "HowlingFjord",
		["TI"] = "Tirisfal", ["AR"] = "Arathi", ["NR"] = "Northrend",
		["RIV"] = "Rivendare", ["WAR"] = "Warfront", ["NAJ"] = "Naj'entu",
		["SWA"] = "Swamp", ["UND"] = "Undercity", ["JF"] = "Jintha'Alor",
		["7SR"] = "StormwindRock", ["8SWA"] = "Swamp", ["8RIV"] = "Rivendare",
		["8UND"] = "Undercity", ["8WAR"] = "Warfront", ["8NAJ"] = "Naj'entu",
		["SP"] = "StormPeaks", ["SB"] = "SholazarBasin",
		["HFjords"] = "HowlingFjord", ["HF"] = "HowlingFjord", ["ZM"] = "Zangarmarsh",
	};

	private static readonly string[] TilesetPathPrefixes =
	[
		"World\\Art\\Tileset\\",
		"World\\Textures\\Terrain\\",
		"Tileset\\",
		"textures\\terrain\\",
	];

	private static readonly Dictionary<string, string> LegacyNameAliases = new(StringComparer.OrdinalIgnoreCase)
	{
		{ "WW_", "WF_" },
		{ "Westwood", "Westfall" },
	};

	private static readonly HashSet<string> KnownTypeHints = new(StringComparer.OrdinalIgnoreCase)
	{
		"grass", "dirt", "rock", "sand", "soil", "stone", "mud", "snow", "ice",
		"gravel", "clay", "cobble", "brick", "wood", "leaf", "flower", "moss",
		"slime", "lava", "ash", "bone", "fur", "skin", "scale", "metal",
		"shadow", "light", "glow", "crystal", "water", "coral", "reef",
	};

	public static void Run(string[] args)
	{
		string? clientRoot = GetOption(args, "--client-root", "-c");
		string? era = GetOption(args, "--era", "-e");
		string? outputDir = GetOption(args, "--output-dir", "-o");
		bool skipMpq = HasFlag(args, "--no-mpq");
		int limit = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;
		int blpSampleSize = GetIntOption(args, "--blp-sample", "-s") ?? 500;

		if (string.IsNullOrWhiteSpace(clientRoot))
		{
			Console.Error.WriteLine("Error: --client-root <path> is required.");
			Environment.ExitCode = 1;
			return;
		}

		clientRoot = ResolveClientRoot(clientRoot);
		if (!Directory.Exists(clientRoot))
		{
			Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
			Environment.ExitCode = 1;
			return;
		}

		string eraTag = string.IsNullOrWhiteSpace(era)
			? Path.GetFileName(clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
			: era.Trim();

		string outputRoot = string.IsNullOrWhiteSpace(outputDir)
			? Path.Combine(Environment.CurrentDirectory, "output", "ml-training", "v10_tileset_database")
			: Path.GetFullPath(outputDir);

		Directory.CreateDirectory(outputRoot);

		List<TilesetEntry> entries = [];
		int scanned = 0;
		int errors = 0;
		int mpqEntries = 0;
		int looseEntries = 0;

		entries = ScanLooseFiles(clientRoot, eraTag, limit, out int looseScanned, out int looseErrors);
		looseEntries = entries.Count;
		scanned = looseScanned;
		errors = looseErrors;

		if (entries.Count < limit && !skipMpq)
		{
			List<TilesetEntry> mpqEntriesList = ScanMpqArchives(clientRoot, eraTag, limit - entries.Count, blpSampleSize, out int mpqScanned, out int mpqErrors);
			HashSet<string> existingPaths = new(entries.Select(e => e.RelativePath), StringComparer.OrdinalIgnoreCase);
			foreach (TilesetEntry entry in mpqEntriesList)
			{
				if (existingPaths.Add(entry.RelativePath))
					entries.Add(entry);
			}
			mpqEntries = mpqEntriesList.Count;
			scanned += mpqScanned;
			errors += mpqErrors;
		}

		TilesetDatabase database = new(
			SchemaVersion: "v10-tileset-database.v1",
			GeneratedAtUtc: DateTimeOffset.UtcNow,
			ClientRoot: clientRoot,
			EraTag: eraTag,
			ScannedFileCount: scanned,
			ErrorCount: errors,
			LooseFileCount: looseEntries,
			MpqFileCount: mpqEntries,
			Entries: entries.OrderBy(e => e.RelativePath, StringComparer.OrdinalIgnoreCase).ToList());

		string dbPath = Path.Combine(outputRoot, $"tileset_database_{SanitizeFileName(eraTag)}.json");
		File.WriteAllText(dbPath, JsonSerializer.Serialize(database, CreateJsonOptions()));

		HashSet<string> designKits = entries.Select(e => e.DesignKit).Where(k => !string.IsNullOrWhiteSpace(k)).Distinct(StringComparer.OrdinalIgnoreCase).ToHashSet();
		HashSet<string> textureTypes = entries.Select(e => e.TypeHint).Where(t => !string.IsNullOrWhiteSpace(t)).Distinct(StringComparer.OrdinalIgnoreCase).ToHashSet();

		Console.WriteLine("WowViewer.Tool.Converter index-tilesets report");
		Console.WriteLine($"ClientRoot: {clientRoot}");
		Console.WriteLine($"EraTag: {eraTag}");
		Console.WriteLine($"OutputDir: {outputRoot}");
		Console.WriteLine($"Scanned: {scanned}");
		Console.WriteLine($"Indexed: {entries.Count}");
		Console.WriteLine($"  Loose: {looseEntries}");
		Console.WriteLine($"  MPQ: {mpqEntries}");
		Console.WriteLine($"Errors: {errors}");
		Console.WriteLine($"DesignKits: {designKits.Count}");
		Console.WriteLine($"TextureTypes: {textureTypes.Count}");
		Console.WriteLine($"Wrote {dbPath}");
	}

	private static string ResolveClientRoot(string path)
	{
		string fullPath = Path.GetFullPath(path);
		string wowSubdir = Path.Combine(fullPath, "World of Warcraft");
		if (Directory.Exists(wowSubdir))
			return wowSubdir;
		return fullPath;
	}

	private static List<TilesetEntry> ScanLooseFiles(string clientRoot, string eraTag, int limit, out int scanned, out int errors)
	{
		List<TilesetEntry> entries = [];
		scanned = 0;
		errors = 0;

		foreach (string scanRoot in TilesetPathPrefixes)
		{
			string fullPath = Path.Combine(clientRoot, scanRoot.Replace('\\', Path.DirectorySeparatorChar).TrimEnd(Path.DirectorySeparatorChar));
			if (!Directory.Exists(fullPath))
				continue;

			foreach (string blpPath in Directory.EnumerateFiles(fullPath, "*.blp", SearchOption.AllDirectories).OrderBy(p => p, StringComparer.OrdinalIgnoreCase))
			{
				if (entries.Count >= limit)
					break;

				scanned++;
				string relativePath = Path.GetRelativePath(clientRoot, blpPath).Replace(Path.DirectorySeparatorChar, '\\');

				try
				{
					BlpSummary summary = BlpSummaryReader.Read(blpPath);
					TilesetEntry entry = BuildTilesetEntryFromSummary(relativePath, blpPath, summary, eraTag, scanRoot);
					entries.Add(entry);
				}
				catch (Exception ex)
				{
					errors++;
					Console.Error.WriteLine($"Warning: failed to read {relativePath}: {ex.Message}");
				}
			}

			if (entries.Count >= limit)
				break;
		}

		return entries;
	}

	private static List<TilesetEntry> ScanMpqArchives(string clientRoot, string eraTag, int limit, int blpSampleSize, out int scanned, out int errors)
	{
		List<TilesetEntry> entries = [];
		scanned = 0;
		errors = 0;

		IArchiveCatalog archiveCatalog = CreateArchiveCatalog(clientRoot);
		try
		{
			IReadOnlyList<string> allFiles = archiveCatalog.GetAllKnownFiles();
			List<string> listfilePaths = [];

			if (allFiles.Count == 0)
			{
				listfilePaths = archiveCatalog.ExtractInternalListfiles()
					.Where(f => f.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
					.ToList();
			}
			else
			{
				listfilePaths = allFiles
					.Where(f => f.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
					.ToList();
			}

			List<string> tilesetBlps = listfilePaths
				.Where(f => TilesetPathPrefixes.Any(prefix => f.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
				.OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
				.ToList();

			scanned = tilesetBlps.Count;

			HashSet<string> sampledKits = new(StringComparer.OrdinalIgnoreCase);
			Dictionary<string, (int Width, int Height, string Format, string Compression, byte AlphaDepth, int MipCount)> dimensionCache = new(StringComparer.OrdinalIgnoreCase);

			foreach (string blpPath in tilesetBlps)
			{
				if (entries.Count >= limit)
					break;

				string normalizedPath = blpPath.Replace('/', '\\');
				string designKit = ExtractDesignKitFromMpqPath(normalizedPath);
				string zoneName = ExtractZoneNameFromMpqPath(normalizedPath);

				bool shouldSample = dimensionCache.Count < blpSampleSize
					|| !sampledKits.Contains(designKit);

				(int width, int height, string format, string compression, byte alphaDepth, int mipCount) = (0, 0, "unknown", "unknown", (byte)0, 0);

				if (shouldSample && archiveCatalog.FileExists(normalizedPath))
				{
					try
					{
						byte[]? blpBytes = archiveCatalog.ReadFile(normalizedPath);
						if (blpBytes is { Length: > 0 })
						{
							using MemoryStream stream = new(blpBytes, writable: false);
							BlpSummary summary = BlpSummaryReader.Read(stream, normalizedPath);
							width = summary.Width;
							height = summary.Height;
							format = summary.Format.ToString();
							compression = summary.Compression.ToString();
							alphaDepth = summary.AlphaDepthBits;
							mipCount = summary.InBoundsMipLevelCount;
							dimensionCache[normalizedPath] = (width, height, format, compression, alphaDepth, mipCount);
							if (!string.IsNullOrWhiteSpace(designKit))
								sampledKits.Add(designKit);
						}
					}
					catch
					{
						errors++;
					}
				}
				else if (dimensionCache.TryGetValue(normalizedPath, out var cached))
				{
					(width, height, format, compression, alphaDepth, mipCount) = cached;
				}

				string fileName = Path.GetFileNameWithoutExtension(normalizedPath);
				string fileNameLower = fileName.ToLowerInvariant();
				(string prefix, string baseName, string suffix) = ParseTextureName(fileName);
				string typeHint = InferTypeHint(baseName, fileNameLower);
				bool isSpecularVariant = suffix.Equals("s", StringComparison.OrdinalIgnoreCase)
					|| fileNameLower.EndsWith("_s", StringComparison.OrdinalIgnoreCase);
				string normalizedLegacyPath = NormalizeLegacyPath(normalizedPath);
				string normalizedZone = NormalizeLegacyZone(zoneName);

				entries.Add(new TilesetEntry(
					RelativePath: normalizedPath,
					NormalizedPath: normalizedLegacyPath,
					FolderPath: Path.GetDirectoryName(normalizedPath) ?? "",
					FileName: fileName,
					FileNameLower: fileNameLower,
					DesignKit: designKit,
					ZoneName: zoneName,
					NormalizedZone: normalizedZone,
					EraTag: eraTag,
					Prefix: prefix,
					BaseName: baseName,
					Suffix: suffix,
					TypeHint: typeHint,
					Width: width,
					Height: height,
					Format: format,
					Compression: compression,
					AlphaDepthBits: alphaDepth,
					HasAlpha: alphaDepth > 0,
					IsSpecularVariant: isSpecularVariant,
					MipLevelCount: mipCount,
					Sha256: ""));
			}
		}
		finally
		{
			archiveCatalog.Dispose();
		}

		return entries;
	}

	private static IArchiveCatalog CreateArchiveCatalog(string clientRoot)
	{
		IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
		ArchiveCatalogBootstrapper.Bootstrap(
			archiveCatalog,
			BuildLegacySearchRoots(clientRoot),
			new ArchiveCatalogBootstrapOptions(ExternalListfilePath: ResolveLegacyListfilePath()));
		if (archiveCatalog is MpqArchiveCatalog mpqCatalog)
			mpqCatalog.ScanMapMpqArchives(clientRoot);
		return archiveCatalog;
	}

	private static IReadOnlyList<string> BuildLegacySearchRoots(string clientRoot)
	{
		List<string> roots = [];
		string dataRoot = Path.Combine(clientRoot, "Data");
		if (Directory.Exists(dataRoot))
			roots.Add(dataRoot);
		if (!string.Equals(clientRoot, dataRoot, StringComparison.OrdinalIgnoreCase))
			roots.Add(clientRoot);
		return roots.Count > 0 ? roots : [clientRoot];
	}

	private static string? ResolveLegacyListfilePath()
	{
		string[] candidates =
		[
			Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv"),
			Path.Combine(AppContext.BaseDirectory, "community-listfile-withcapitals.csv"),
			"community-listfile-withcapitals.csv",
			"listfile.csv",
		];
		foreach (string candidate in candidates)
		{
			if (File.Exists(candidate))
				return candidate;
		}
		return null;
	}

	private static TilesetEntry BuildTilesetEntryFromSummary(string relativePath, string absolutePath, BlpSummary summary, string eraTag, string scanRoot)
	{
		string folderPath = Path.GetDirectoryName(relativePath) ?? "";
		string fileName = Path.GetFileNameWithoutExtension(relativePath);
		string fileNameLower = fileName.ToLowerInvariant();

		string designKit = ExtractDesignKit(folderPath, scanRoot);
		string zoneName = ExtractZoneName(folderPath);
		(string prefix, string baseName, string suffix) = ParseTextureName(fileName);
		string typeHint = InferTypeHint(baseName, fileNameLower);
		bool hasAlpha = summary.AlphaDepthBits > 0;
		bool isSpecularVariant = suffix.Equals("s", StringComparison.OrdinalIgnoreCase)
			|| fileNameLower.EndsWith("_s", StringComparison.OrdinalIgnoreCase);

		string normalizedPath = NormalizeLegacyPath(relativePath);
		string normalizedZone = NormalizeLegacyZone(zoneName);

		return new TilesetEntry(
			RelativePath: relativePath,
			NormalizedPath: normalizedPath,
			FolderPath: folderPath,
			FileName: fileName,
			FileNameLower: fileNameLower,
			DesignKit: designKit,
			ZoneName: zoneName,
			NormalizedZone: normalizedZone,
			EraTag: eraTag,
			Prefix: prefix,
			BaseName: baseName,
			Suffix: suffix,
			TypeHint: typeHint,
			Width: summary.Width,
			Height: summary.Height,
			Format: summary.Format.ToString(),
			Compression: summary.Compression.ToString(),
			AlphaDepthBits: summary.AlphaDepthBits,
			HasAlpha: hasAlpha,
			IsSpecularVariant: isSpecularVariant,
			MipLevelCount: summary.InBoundsMipLevelCount,
			Sha256: ComputeFileSha256(absolutePath));
	}

	private static string ExtractDesignKit(string folderPath, string scanRoot)
	{
		string normalized = folderPath.Replace('/', '\\');
		string[] parts = normalized.Split('\\', StringSplitOptions.RemoveEmptyEntries);
		int scanRootIndex = -1;
		string[] scanRootParts = scanRoot.Split('\\', StringSplitOptions.RemoveEmptyEntries);
		for (int i = 0; i <= parts.Length - scanRootParts.Length; i++)
		{
			bool match = true;
			for (int j = 0; j < scanRootParts.Length; j++)
			{
				if (!string.Equals(parts[i + j], scanRootParts[j], StringComparison.OrdinalIgnoreCase))
				{
					match = false;
					break;
				}
			}
			if (match)
			{
				scanRootIndex = i + scanRootParts.Length;
				break;
			}
		}
		if (scanRootIndex < 0 || scanRootIndex >= parts.Length)
			return string.Empty;
		return ExpandZoneAbbreviation(StripCopyOfPrefix(parts[scanRootIndex]));
	}

	private static string ExtractDesignKitFromMpqPath(string normalizedPath)
	{
		string[] parts = normalizedPath.Split('\\', StringSplitOptions.RemoveEmptyEntries);
		for (int i = 0; i < parts.Length - 1; i++)
		{
			if (string.Equals(parts[i], "Tileset", StringComparison.OrdinalIgnoreCase)
				|| (string.Equals(parts[i], "Art", StringComparison.OrdinalIgnoreCase) && i + 1 < parts.Length && string.Equals(parts[i + 1], "Tileset", StringComparison.OrdinalIgnoreCase)))
			{
				int kitIndex = string.Equals(parts[i], "Tileset", StringComparison.OrdinalIgnoreCase) ? i + 1 : i + 2;
				if (kitIndex < parts.Length - 1)
					return ExpandZoneAbbreviation(StripCopyOfPrefix(parts[kitIndex]));
			}
		}
		return string.Empty;
	}

	private static string ExtractZoneName(string folderPath)
	{
		string normalized = folderPath.Replace('/', '\\');
		string[] parts = normalized.Split('\\', StringSplitOptions.RemoveEmptyEntries);
		if (parts.Length < 2)
			return string.Empty;
		return ExpandZoneAbbreviation(StripCopyOfPrefix(parts[^1]));
	}

	private static string ExtractZoneNameFromMpqPath(string normalizedPath)
	{
		string[] parts = normalizedPath.Split('\\', StringSplitOptions.RemoveEmptyEntries);
		if (parts.Length < 2)
			return string.Empty;
		return ExpandZoneAbbreviation(StripCopyOfPrefix(parts[^2]));
	}

	private static string ExpandZoneAbbreviation(string raw)
	{
		if (ZoneAbbreviations.TryGetValue(raw, out string? expanded))
			return expanded;
		return raw;
	}

	private static string StripCopyOfPrefix(string name)
	{
		if (name.StartsWith("Copy of ", StringComparison.OrdinalIgnoreCase))
			return name.Substring(8);
		if (name.StartsWith("Copyof", StringComparison.OrdinalIgnoreCase))
			return name.Substring(6);
		return name;
	}

	private static (string Prefix, string BaseName, string Suffix) ParseTextureName(string fileName)
	{
		string baseName = fileName;
		string prefix = "";
		string suffix = "";

		int underscoreIndex = fileName.IndexOf('_');
		if (underscoreIndex > 0 && underscoreIndex < fileName.Length - 1)
		{
			prefix = fileName[..underscoreIndex];
			baseName = fileName[(underscoreIndex + 1)..];
		}

		if (baseName.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
		{
			suffix = "s";
			baseName = baseName[..^2];
		}
		else if (baseName.EndsWith("alpha", StringComparison.OrdinalIgnoreCase))
		{
			suffix = "alpha";
			baseName = baseName[..^5];
		}

		return (prefix, baseName, suffix);
	}

	private static string InferTypeHint(string baseName, string fileNameLower)
	{
		foreach (string typeHint in KnownTypeHints)
		{
			if (fileNameLower.Contains(typeHint, StringComparison.OrdinalIgnoreCase))
				return typeHint;
		}
		return string.Empty;
	}

	private static string NormalizeLegacyPath(string path)
	{
		string result = path;
		foreach ((string oldPrefix, string newPrefix) in LegacyNameAliases)
		{
			if (result.StartsWith(oldPrefix, StringComparison.OrdinalIgnoreCase))
				result = newPrefix + result[oldPrefix.Length..];
		}
		return result;
	}

	private static string NormalizeLegacyZone(string zoneName)
	{
		string result = ExpandZoneAbbreviation(StripCopyOfPrefix(zoneName));
		foreach ((string oldName, string newName) in LegacyNameAliases)
		{
			if (result.Contains(oldName.TrimEnd('_'), StringComparison.OrdinalIgnoreCase))
				result = result.Replace(oldName.TrimEnd('_'), newName.TrimEnd('_'), StringComparison.OrdinalIgnoreCase);
		}
		return result;
	}

	private static string ComputeFileSha256(string path)
	{
		using FileStream stream = File.OpenRead(path);
		byte[] hash = SHA256.HashData(stream);
		return Convert.ToHexString(hash);
	}

	private static string? GetOption(string[] args, string longName, string shortName)
	{
		for (int i = 0; i < args.Length - 1; i++)
		{
			if (string.Equals(args[i], longName, StringComparison.OrdinalIgnoreCase)
				|| string.Equals(args[i], shortName, StringComparison.OrdinalIgnoreCase))
				return args[i + 1];
		}
		return null;
	}

	private static int? GetIntOption(string[] args, string longName, string shortName)
	{
		string? value = GetOption(args, longName, shortName);
		if (string.IsNullOrWhiteSpace(value))
			return null;
		return int.TryParse(value, out int parsed) ? parsed : null;
	}

	private static bool HasFlag(IEnumerable<string> args, string name)
	{
		return args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
	}

	private static string SanitizeFileName(string value)
	{
		StringBuilder builder = new(value.Length);
		foreach (char c in value)
			builder.Append(char.IsLetterOrDigit(c) ? c : '_');
		return builder.ToString().Trim('_');
	}

	private static JsonSerializerOptions CreateJsonOptions()
	{
		JsonSerializerOptions options = new() { WriteIndented = true };
		options.Converters.Add(new JsonStringEnumConverter());
		return options;
	}
}

public sealed record TilesetEntry(
	[property: JsonPropertyName("relative_path")] string RelativePath,
	[property: JsonPropertyName("normalized_path")] string NormalizedPath,
	[property: JsonPropertyName("folder_path")] string FolderPath,
	[property: JsonPropertyName("file_name")] string FileName,
	[property: JsonPropertyName("file_name_lower")] string FileNameLower,
	[property: JsonPropertyName("design_kit")] string DesignKit,
	[property: JsonPropertyName("zone_name")] string ZoneName,
	[property: JsonPropertyName("normalized_zone")] string NormalizedZone,
	[property: JsonPropertyName("era_tag")] string EraTag,
	[property: JsonPropertyName("prefix")] string Prefix,
	[property: JsonPropertyName("base_name")] string BaseName,
	[property: JsonPropertyName("suffix")] string Suffix,
	[property: JsonPropertyName("type_hint")] string TypeHint,
	[property: JsonPropertyName("width")] int Width,
	[property: JsonPropertyName("height")] int Height,
	[property: JsonPropertyName("format")] string Format,
	[property: JsonPropertyName("compression")] string Compression,
	[property: JsonPropertyName("alpha_depth_bits")] byte AlphaDepthBits,
	[property: JsonPropertyName("has_alpha")] bool HasAlpha,
	[property: JsonPropertyName("is_specular_variant")] bool IsSpecularVariant,
	[property: JsonPropertyName("mip_level_count")] int MipLevelCount,
	[property: JsonPropertyName("sha256")] string Sha256);

public sealed record TilesetDatabase(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_at_utc")] DateTimeOffset GeneratedAtUtc,
	[property: JsonPropertyName("client_root")] string ClientRoot,
	[property: JsonPropertyName("era_tag")] string EraTag,
	[property: JsonPropertyName("scanned_file_count")] int ScannedFileCount,
	[property: JsonPropertyName("error_count")] int ErrorCount,
	[property: JsonPropertyName("loose_file_count")] int LooseFileCount,
	[property: JsonPropertyName("mpq_file_count")] int MpqFileCount,
	[property: JsonPropertyName("entries")] List<TilesetEntry> Entries);
