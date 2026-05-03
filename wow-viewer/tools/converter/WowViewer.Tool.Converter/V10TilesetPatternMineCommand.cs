using System.Drawing;
using System.Text.Json;
using System.Text.Json.Serialization;
using SereniaBLPLib;
using WowViewer.Core.Datasets;
using WowViewer.Core.IO.Files;

namespace WowViewer.Tool.Converter;

public static class V10TilesetPatternMineCommand
{
	public static void Run(string[] args)
	{
		string? input = GetOption(args, "--input", "-i");
		string? outputDir = GetOption(args, "--output-dir", "-o");
		int limit = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;
		int mipLevel = GetIntOption(args, "--mip", "-m") ?? 3;
		bool writePreviews = HasFlag(args, "--write-previews");

		if (string.IsNullOrWhiteSpace(input))
		{
			Console.Error.WriteLine("Error: --input <merged_tileset_index.json> is required.");
			Environment.ExitCode = 1;
			return;
		}

		if (!File.Exists(input))
		{
			Console.Error.WriteLine($"Error: input file not found: {input}");
			Environment.ExitCode = 1;
			return;
		}

		string outputRoot = string.IsNullOrWhiteSpace(outputDir)
			? Path.Combine(Environment.CurrentDirectory, "output", "ml-training", "v10_tileset_patterns")
			: Path.GetFullPath(outputDir);

		Directory.CreateDirectory(outputRoot);

		using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(input));
		JsonElement docRoot = doc.RootElement;

		List<string> clientRoots = [];
		foreach (JsonElement cr in docRoot.GetProperty("client_roots").EnumerateArray())
			clientRoots.Add(cr.GetString() ?? "");

		List<JsonElement> entryElements = [];
		foreach (JsonElement e in docRoot.GetProperty("entries").EnumerateArray())
			entryElements.Add(e);

		Console.WriteLine($"Loaded {entryElements.Count} tileset entries");
		Console.WriteLine($"Client roots: {string.Join(", ", clientRoots)}");

		HashSet<string> processedNames = new(StringComparer.OrdinalIgnoreCase);

		List<IArchiveCatalog> catalogs = [];
		try
		{
			foreach (string root in clientRoots)
				catalogs.Add(CreateArchiveCatalog(root));

			Dictionary<string, PatternResult> patterns = new(StringComparer.OrdinalIgnoreCase);
			int processed = 0;
			int errors = 0;
			int skipped = 0;
			int skippedDuplicate = 0;
			int skippedSpecular = 0;

			foreach (JsonElement entry in entryElements)
			{
				if (processed >= limit)
					break;

				string relPath = entry.GetProperty("relative_path").GetString() ?? "";
				string fileName = entry.GetProperty("file_name").GetString() ?? "";
				string designKit = entry.GetProperty("design_kit").GetString() ?? "";
				string eraTag = entry.GetProperty("era_tag").GetString() ?? "";
				string typeHint = entry.GetProperty("type_hint").GetString() ?? "";

				if (fileName.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
				{
					skippedSpecular++;
					continue;
				}

				if (!processedNames.Add(fileName))
				{
					skippedDuplicate++;
					continue;
				}

				try
				{
					(byte[] rgba, int w, int h) = DecodeBlp(relPath, clientRoots, catalogs, mipLevel);
					if (rgba is null || w < 16 || h < 16)
					{
						skipped++;
						continue;
					}

					PatternStamp stamp = PatternMiner.AnalyzeTexture(rgba, w, h, fileName);

					string baseKey = $"{designKit}_{typeHint}_{fileName}";

					if (!patterns.TryGetValue(baseKey, out PatternResult? existing) || stamp.PeriodicityScore > existing.Stamp.PeriodicityScore)
					{
						patterns[baseKey] = new PatternResult(
							Stamp: stamp,
							SourceEntries: [relPath],
							DesignKit: designKit,
							EraTag: eraTag);
					}

					processed++;

					if (processed % 100 == 0)
						Console.WriteLine($"  Processed {processed} textures...");
				}
				catch (Exception ex)
				{
					errors++;
					if (errors <= 10)
						Console.Error.WriteLine($"  Error processing {relPath}: {ex.Message}");
				}
			}

			string reportPath = Path.Combine(outputRoot, "pattern_library.json");
			File.WriteAllText(reportPath, JsonSerializer.Serialize(new PatternLibraryReport(
				SchemaVersion: "v10-tileset-patterns.v3",
				GeneratedAtUtc: DateTimeOffset.UtcNow,
				TotalProcessed: processed,
				TotalErrors: errors,
				TotalSkipped: skipped,
				Patterns: patterns.Values.OrderByDescending(p => p.Stamp.PeriodicityScore).ToList()), CreateJsonOptions()));

			Console.WriteLine("");
			Console.WriteLine("WowViewer.Tool.Converter mine-tileset-patterns report");
			Console.WriteLine($"Input: {input}");
			Console.WriteLine($"Output: {reportPath}");
			Console.WriteLine($"Processed: {processed}");
			Console.WriteLine($"Errors: {errors}");
			Console.WriteLine($"Skipped: {skipped} ({skippedSpecular} specular, {skippedDuplicate} duplicate names)");
			Console.WriteLine($"Pattern clusters: {patterns.Count}");
			Console.WriteLine("");

			Console.WriteLine("=== Top Patterns (by periodicity score) ===");
			foreach (PatternResult p in patterns.Values.OrderByDescending(p => p.Stamp.PeriodicityScore).Take(20))
			{
				Console.WriteLine($"  {p.DesignKit}/{p.Stamp.TextureName}: tile={p.Stamp.TileSizeX}x{p.Stamp.TileSizeY} scale={p.Stamp.PatternScaleHint} tint={p.Stamp.MeanColorHex} hue={p.Stamp.MeanHueDegrees:F1} chroma={ShortHash(p.Stamp.ChromaSignatureHash)} detail={ShortHash(p.Stamp.ChromaDetailSignatureHash)} detailEnergy={p.Stamp.ChromaDetailEnergy:F2} score={p.Stamp.PeriodicityScore:F4}");
			}
		}
		finally
		{
			foreach (IArchiveCatalog catalog in catalogs)
				catalog.Dispose();
		}
	}

	private static (byte[]? Rgba, int Width, int Height) DecodeBlp(string relativePath, List<string> clientRoots, List<IArchiveCatalog> catalogs, int mipLevel)
	{
		string normalizedRel = relativePath.Replace('\\', '/');

		for (int i = 0; i < clientRoots.Count; i++)
		{
			string loosePath = Path.Combine(clientRoots[i], relativePath.Replace('\\', Path.DirectorySeparatorChar));
			if (File.Exists(loosePath))
			{
				(byte[] rgba, int w, int h) = WowViewer.Core.IO.Blp.BlpPixelDecoder.DecodeRgbaWithDimensions(loosePath, mipLevel);
				return (rgba, w, h);
			}

			if (i < catalogs.Count && catalogs[i].FileExists(normalizedRel))
			{
				byte[]? blpBytes = catalogs[i].ReadFile(normalizedRel);
				if (blpBytes is { Length: > 0 })
				{
					using MemoryStream ms = new(blpBytes, writable: false);
					using BlpFile blp = new(ms);
					using Bitmap bmp = blp.GetBitmap(mipLevel);
					int w = bmp.Width;
					int h = bmp.Height;
					byte[] rgba = new byte[w * h * 4];
					for (int y = 0; y < h; y++)
						for (int x = 0; x < w; x++)
						{
							System.Drawing.Color c = bmp.GetPixel(x, y);
							int idx = (y * w + x) * 4;
							rgba[idx] = c.R; rgba[idx + 1] = c.G; rgba[idx + 2] = c.B; rgba[idx + 3] = c.A;
						}
					return (rgba, w, h);
				}
			}
		}

		return (null, 0, 0);
	}

	private static IArchiveCatalog CreateArchiveCatalog(string clientRoot)
	{
		IArchiveCatalog catalog = new MpqArchiveCatalogFactory().Create();
		ArchiveCatalogBootstrapper.Bootstrap(catalog, [clientRoot, Path.Combine(clientRoot, "Data")], new ArchiveCatalogBootstrapOptions(ExternalListfilePath: null));
		if (catalog is MpqArchiveCatalog mpqCatalog)
			mpqCatalog.ScanMapMpqArchives(clientRoot);
		return catalog;
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

	private static bool HasFlag(string[] args, string name)
	{
		return args.Any(a => string.Equals(a, name, StringComparison.OrdinalIgnoreCase));
	}

	private static string ShortHash(string value)
	{
		if (string.IsNullOrWhiteSpace(value))
			return "";
		return value.Length <= 12 ? value : value[..12];
	}

	private static JsonSerializerOptions CreateJsonOptions()
	{
		return new JsonSerializerOptions { WriteIndented = true, Converters = { new JsonStringEnumConverter() } };
	}
}

public sealed record PatternResult(
	[property: JsonPropertyName("stamp")] PatternStamp Stamp,
	[property: JsonPropertyName("source_entries")] List<string> SourceEntries,
	[property: JsonPropertyName("design_kit")] string DesignKit,
	[property: JsonPropertyName("era_tag")] string EraTag);

public sealed record PatternLibraryReport(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_at_utc")] DateTimeOffset GeneratedAtUtc,
	[property: JsonPropertyName("total_processed")] int TotalProcessed,
	[property: JsonPropertyName("total_errors")] int TotalErrors,
	[property: JsonPropertyName("total_skipped")] int TotalSkipped,
	[property: JsonPropertyName("patterns")] List<PatternResult> Patterns);
