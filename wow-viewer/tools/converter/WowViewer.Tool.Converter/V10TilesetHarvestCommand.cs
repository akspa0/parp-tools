using System.Drawing;
using System.Drawing.Imaging;
using System.Text.Json;
using System.Text.Json.Serialization;
using SereniaBLPLib;
using WowViewer.Core.IO.Files;

namespace WowViewer.Tool.Converter;

public static class V10TilesetHarvestCommand
{
	public static void Run(string[] args)
	{
		string? input = GetOption(args, "--input", "-i");
		string? outputDir = GetOption(args, "--output-dir", "-o");
		int limit = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;

		if (string.IsNullOrWhiteSpace(input))
		{
			Console.Error.WriteLine("Error: --input <merged_tileset_index.json> is required.");
			Environment.ExitCode = 1;
			return;
		}

		string outputRoot = string.IsNullOrWhiteSpace(outputDir)
			? Path.Combine(Environment.CurrentDirectory, "output", "ml-training", "v10_tileset_pngs")
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

		Console.WriteLine($"Source entries: {entryElements.Count}");

		HashSet<string> uniqueNames = new(StringComparer.OrdinalIgnoreCase);
		List<(string Name, string RelPath, string DesignKit, string EraTag)> toHarvest = [];

		foreach (JsonElement entry in entryElements)
		{
			if (toHarvest.Count >= limit)
				break;

			string name = entry.GetProperty("file_name").GetString() ?? "";
			string relPath = entry.GetProperty("relative_path").GetString() ?? "";
			string designKit = entry.GetProperty("design_kit").GetString() ?? "";
			string eraTag = entry.GetProperty("era_tag").GetString() ?? "";

			if (name.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
				continue;

			if (uniqueNames.Add(name))
				toHarvest.Add((name, relPath, designKit, eraTag));
		}

		Console.WriteLine($"Unique non-specular textures to harvest: {toHarvest.Count}");

		List<IArchiveCatalog> catalogs = [];
		try
		{
			foreach (string cr in clientRoots)
				catalogs.Add(CreateArchiveCatalog(cr));

			int harvested = 0;
			int errors = 0;
			List<HarvestEntry> manifest = [];

			foreach ((string name, string relPath, string designKit, string eraTag) in toHarvest)
			{
				string safeKit = SanitizeFileName(designKit);
				string kitDir = Path.Combine(outputRoot, safeKit);
				Directory.CreateDirectory(kitDir);

				string pngPath = Path.Combine(kitDir, $"{name}.png");

				if (File.Exists(pngPath))
				{
					manifest.Add(new HarvestEntry(Name: name, PngPath: pngPath, DesignKit: designKit, RelativePath: relPath, EraTag: eraTag));
					harvested++;
					continue;
				}

				bool decoded = TryDecodeBlpToPng(relPath, clientRoots, catalogs, pngPath);
				if (decoded)
				{
					manifest.Add(new HarvestEntry(Name: name, PngPath: pngPath, DesignKit: designKit, RelativePath: relPath, EraTag: eraTag));
					harvested++;
				}
				else
				{
					errors++;
				}

				if (harvested % 200 == 0)
					Console.WriteLine($"  Harvested {harvested}/{toHarvest.Count}...");
			}

			string manifestPath = Path.Combine(outputRoot, "harvest_manifest.json");
			File.WriteAllText(manifestPath, JsonSerializer.Serialize(new HarvestManifest(
				SchemaVersion: "v10-tileset-harvest.v1",
				GeneratedAtUtc: DateTimeOffset.UtcNow,
				TotalHarvested: harvested,
				TotalErrors: errors,
				Entries: manifest), new JsonSerializerOptions { WriteIndented = true }));

			Console.WriteLine("");
			Console.WriteLine($"Harvest complete: {harvested} exported, {errors} errors");
			Console.WriteLine($"Manifest: {manifestPath}");
		}
		finally
		{
			foreach (IArchiveCatalog catalog in catalogs)
				catalog.Dispose();
		}
	}

	private static bool TryDecodeBlpToPng(string relPath, List<string> clientRoots, List<IArchiveCatalog> catalogs, string outputPath)
	{
		string normalizedRel = relPath.Replace('\\', '/');

		for (int i = 0; i < clientRoots.Count; i++)
		{
			string loosePath = Path.Combine(clientRoots[i], relPath.Replace('\\', Path.DirectorySeparatorChar));
			if (File.Exists(loosePath))
			{
				using FileStream fs = File.OpenRead(loosePath);
				using BlpFile blp = new(fs);
				using System.Drawing.Bitmap bmp = blp.GetBitmap(0);
				bmp.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);
				return true;
			}

			if (i < catalogs.Count && catalogs[i].FileExists(normalizedRel))
			{
				byte[]? blpBytes = catalogs[i].ReadFile(normalizedRel);
				if (blpBytes is { Length: > 0 })
				{
					using MemoryStream ms = new(blpBytes, writable: false);
					using BlpFile blp = new(ms);
					using System.Drawing.Bitmap bmp = blp.GetBitmap(0);
					bmp.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);
					return true;
				}
			}
		}

		return false;
	}

	private static IArchiveCatalog CreateArchiveCatalog(string clientRoot)
	{
		IArchiveCatalog catalog = new MpqArchiveCatalogFactory().Create();
		ArchiveCatalogBootstrapper.Bootstrap(catalog, [clientRoot, Path.Combine(clientRoot, "Data")], new ArchiveCatalogBootstrapOptions(ExternalListfilePath: null));
		if (catalog is MpqArchiveCatalog mpqCatalog)
			mpqCatalog.ScanMapMpqArchives(clientRoot);
		return catalog;
	}

	private static string SanitizeFileName(string value)
	{
		System.Text.StringBuilder builder = new(value.Length);
		foreach (char c in value)
			builder.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');
		return builder.ToString().Trim('_');
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
		return int.TryParse(value, out int parsed) ? parsed : null;
	}
}

public sealed record HarvestEntry(
	[property: JsonPropertyName("name")] string Name,
	[property: JsonPropertyName("png_path")] string PngPath,
	[property: JsonPropertyName("design_kit")] string DesignKit,
	[property: JsonPropertyName("relative_path")] string RelativePath,
	[property: JsonPropertyName("era_tag")] string EraTag);

public sealed record HarvestManifest(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_at_utc")] DateTimeOffset GeneratedAtUtc,
	[property: JsonPropertyName("total_harvested")] int TotalHarvested,
	[property: JsonPropertyName("total_errors")] int TotalErrors,
	[property: JsonPropertyName("entries")] List<HarvestEntry> Entries);
