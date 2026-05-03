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

		List<ArchiveCatalogSession> sessions = [];
		Dictionary<string, ArchiveCatalogSession> sessionsByEraTag = new(StringComparer.OrdinalIgnoreCase);
		try
		{
			foreach (string cr in clientRoots)
			{
				ArchiveCatalogSession session = V10TilesetArchiveReader.GetOrCreateSession(cr);
				sessions.Add(session);
				foreach (string eraTagKey in GetEraTagKeysForClientRoot(cr))
				{
					sessionsByEraTag.TryAdd(eraTagKey, session);
				}
			}

			int harvested = 0;
			int errors = 0;
			int preferredSessionHits = 0;
			int fallbackSessionHits = 0;
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

				DecodeResult decodeResult = TryDecodeBlpToPng(relPath, eraTag, sessionsByEraTag, sessions, pngPath);
				if (decodeResult != DecodeResult.NotFound)
				{
					if (decodeResult == DecodeResult.PreferredSession)
						preferredSessionHits++;
					else
						fallbackSessionHits++;

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
			Console.WriteLine($"Preferred era-session hits: {preferredSessionHits}");
			Console.WriteLine($"Fallback session hits: {fallbackSessionHits}");
			Console.WriteLine($"Manifest: {manifestPath}");
		}
		finally
		{
		}
	}

	private static DecodeResult TryDecodeBlpToPng(
		string relPath,
		string eraTag,
		IReadOnlyDictionary<string, ArchiveCatalogSession> sessionsByEraTag,
		List<ArchiveCatalogSession> sessions,
		string outputPath)
	{
		ArchiveCatalogSession? preferredSession = null;
		if (!string.IsNullOrWhiteSpace(eraTag) && sessionsByEraTag.TryGetValue(eraTag.Trim(), out preferredSession))
		{
			if (TryDecodeBlpToPngFromSession(preferredSession, relPath, outputPath))
				return DecodeResult.PreferredSession;
		}

		foreach (ArchiveCatalogSession session in sessions)
		{
			if (ReferenceEquals(session, preferredSession))
				continue;

			if (TryDecodeBlpToPngFromSession(session, relPath, outputPath))
				return DecodeResult.FallbackSession;
		}

		return DecodeResult.NotFound;
	}

	private static bool TryDecodeBlpToPngFromSession(ArchiveCatalogSession session, string relPath, string outputPath)
	{
		byte[]? blpBytes = V10TilesetArchiveReader.TryReadVirtualFile(session, relPath);
		if (blpBytes is not { Length: > 0 })
			return false;

		using MemoryStream ms = new(blpBytes, writable: false);
		using BlpFile blp = new(ms);
		using System.Drawing.Bitmap bmp = blp.GetBitmap(0);
		bmp.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);
		return true;
	}

	private static IEnumerable<string> GetEraTagKeysForClientRoot(string clientRoot)
	{
		if (string.IsNullOrWhiteSpace(clientRoot))
			yield break;

		string? current = Path.GetFullPath(clientRoot);
		for (int depth = 0; depth < 3 && !string.IsNullOrWhiteSpace(current); depth++)
		{
			string candidate = Path.GetFileName(current.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
			if (!string.IsNullOrWhiteSpace(candidate))
			{
				yield return candidate;
				yield return candidate.Replace('_', '.');
			}

			current = Path.GetDirectoryName(current);
		}
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

	private enum DecodeResult
	{
		NotFound,
		PreferredSession,
		FallbackSession,
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
