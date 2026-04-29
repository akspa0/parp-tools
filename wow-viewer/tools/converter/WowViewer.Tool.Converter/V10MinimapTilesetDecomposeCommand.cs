using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Datasets;

namespace WowViewer.Tool.Converter;

public static class V10MinimapTilesetDecomposeCommand
{
	public static void Run(string[] args)
	{
		string? patternLibraryPath = GetOption(args, "--pattern-library", "-p");
		string? minimapPath = GetOption(args, "--minimap", "-m");
		string? outputDir = GetOption(args, "--output-dir", "-o");
		int gridSize = GetIntOption(args, "--grid-size", "-g") ?? 16;
		int maxCandidates = GetIntOption(args, "--max-candidates", "-c") ?? 3;
		int patternLimit = GetIntOption(args, "--limit-patterns", "-n") ?? int.MaxValue;

		if (string.IsNullOrWhiteSpace(patternLibraryPath))
		{
			Console.Error.WriteLine("Error: --pattern-library <pattern_library.json> is required.");
			Environment.ExitCode = 1;
			return;
		}

		if (string.IsNullOrWhiteSpace(minimapPath))
		{
			Console.Error.WriteLine("Error: --minimap <minimap.png> is required.");
			Environment.ExitCode = 1;
			return;
		}

		if (!File.Exists(patternLibraryPath))
		{
			Console.Error.WriteLine($"Error: pattern library not found: {patternLibraryPath}");
			Environment.ExitCode = 1;
			return;
		}

		if (!File.Exists(minimapPath))
		{
			Console.Error.WriteLine($"Error: minimap not found: {minimapPath}");
			Environment.ExitCode = 1;
			return;
		}

		string outputRoot = string.IsNullOrWhiteSpace(outputDir)
			? Path.Combine(Environment.CurrentDirectory, "output", "ml-training", "v10_minimap_decomposition")
			: Path.GetFullPath(outputDir);
		Directory.CreateDirectory(outputRoot);

		try
		{
			PatternLibraryReport library = JsonSerializer.Deserialize<PatternLibraryReport>(
				File.ReadAllText(patternLibraryPath),
				CreateJsonOptions()) ?? throw new InvalidOperationException("Pattern library JSON was empty.");

			if (!string.Equals(library.SchemaVersion, "v10-tileset-patterns.v3", StringComparison.OrdinalIgnoreCase))
				Console.Error.WriteLine($"Warning: expected v10-tileset-patterns.v3, found {library.SchemaVersion}.");

			List<TilesetPatternCandidate> patterns = library.Patterns
				.Where(p => p.Stamp.ChromaMipSignature.Length > 0 && p.Stamp.ChromaDetailSignature.Length > 0)
				.OrderByDescending(p => p.Stamp.PeriodicityScore)
				.Take(patternLimit)
				.Select((p, i) => new TilesetPatternCandidate(
					Id: $"{i:D5}:{p.DesignKit}:{p.Stamp.TextureName}",
					TextureName: p.Stamp.TextureName,
					DesignKit: p.DesignKit,
					EraTag: p.EraTag,
					Stamp: p.Stamp))
				.ToList();

			if (patterns.Count == 0)
				throw new InvalidOperationException("Pattern library does not contain usable v3 color/detail signatures.");

			using Image<Rgba32> image = Image.Load<Rgba32>(minimapPath);
			byte[] rgba = ToRgbaBytes(image);
			MinimapTilesetDecomposition decomposition = MinimapTilesetPatternMatcher.Decompose(
				rgba,
				image.Width,
				image.Height,
				patterns,
				gridSize,
				maxCandidates);

			string jsonPath = Path.Combine(outputRoot, "minimap_tileset_decomposition.json");
			File.WriteAllText(jsonPath, JsonSerializer.Serialize(new MinimapDecompositionReport(
				SchemaVersion: "v10-minimap-tileset-decomposition.v1",
				GeneratedAtUtc: DateTimeOffset.UtcNow,
				PatternLibraryPath: Path.GetFullPath(patternLibraryPath),
				PatternLibrarySchemaVersion: library.SchemaVersion,
				MinimapPath: Path.GetFullPath(minimapPath),
				PatternCandidates: patterns.Count,
				GridSize: gridSize,
				MaxCandidatesPerCell: maxCandidates,
				Decomposition: decomposition), CreateJsonOptions()));

			WritePreviewImages(outputRoot, rgba, image.Width, image.Height, decomposition);

			double avgTopScore = decomposition.Cells
				.Where(c => c.Candidates.Count > 0)
				.Select(c => c.Candidates[0].Score)
				.DefaultIfEmpty(0.0)
				.Average();
			int distinctTop = decomposition.Cells
				.Where(c => c.Candidates.Count > 0)
				.Select(c => c.Candidates[0].CandidateId)
				.Distinct(StringComparer.OrdinalIgnoreCase)
				.Count();

			Console.WriteLine("WowViewer.Tool.Converter decompose-minimap-tilesets report");
			Console.WriteLine($"Pattern library: {patternLibraryPath}");
			Console.WriteLine($"Minimap: {minimapPath}");
			Console.WriteLine($"Output: {jsonPath}");
			Console.WriteLine($"Grid: {gridSize}x{gridSize}");
			Console.WriteLine($"Pattern candidates: {patterns.Count}");
			Console.WriteLine($"Cells: {decomposition.Cells.Count}");
			Console.WriteLine($"Distinct top candidates: {distinctTop}");
			Console.WriteLine($"Average top score: {avgTopScore:F4}");
			Console.WriteLine($"Preview: {Path.Combine(outputRoot, "best_match_mean.png")}");
			Console.WriteLine($"Residual: {Path.Combine(outputRoot, "residual_to_best_mean.png")}");
			Console.WriteLine($"Confidence: {Path.Combine(outputRoot, "confidence.png")}");
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error decomposing minimap: {ex.Message}");
			Environment.ExitCode = 1;
		}
	}

	private static byte[] ToRgbaBytes(Image<Rgba32> image)
	{
		Rgba32[] pixels = new Rgba32[image.Width * image.Height];
		image.CopyPixelDataTo(pixels);
		byte[] rgba = new byte[pixels.Length * 4];
		for (int i = 0; i < pixels.Length; i++)
		{
			int idx = i * 4;
			rgba[idx + 0] = pixels[i].R;
			rgba[idx + 1] = pixels[i].G;
			rgba[idx + 2] = pixels[i].B;
			rgba[idx + 3] = pixels[i].A;
		}

		return rgba;
	}

	private static void WritePreviewImages(string outputRoot, byte[] sourceRgba, int width, int height, MinimapTilesetDecomposition decomposition)
	{
		using Image<Rgba32> bestMean = new(width, height);
		using Image<Rgba32> residual = new(width, height);
		using Image<Rgba32> confidence = new(width, height);

		foreach (MinimapTilesetCell cell in decomposition.Cells)
		{
			MinimapTilesetMatch? best = cell.Candidates.FirstOrDefault();
			Rgba32 meanColor = best is null ? new Rgba32(0, 0, 0) : ParseHexColor(best.MeanColorHex);
			byte confidenceByte = best is null ? (byte)0 : (byte)Math.Clamp((int)Math.Round(best.Score * 255.0), 0, 255);
			Rgba32 confidenceColor = new(confidenceByte, confidenceByte, confidenceByte);

			for (int y = cell.PixelY; y < cell.PixelY + cell.Height; y++)
			{
				for (int x = cell.PixelX; x < cell.PixelX + cell.Width; x++)
				{
					int sourceIdx = ((y * width) + x) * 4;
					Rgba32 source = new(sourceRgba[sourceIdx + 0], sourceRgba[sourceIdx + 1], sourceRgba[sourceIdx + 2]);
					bestMean[x, y] = meanColor;
					residual[x, y] = new Rgba32(
						(byte)Math.Abs(source.R - meanColor.R),
						(byte)Math.Abs(source.G - meanColor.G),
						(byte)Math.Abs(source.B - meanColor.B));
					confidence[x, y] = confidenceColor;
				}
			}
		}

		bestMean.SaveAsPng(Path.Combine(outputRoot, "best_match_mean.png"));
		residual.SaveAsPng(Path.Combine(outputRoot, "residual_to_best_mean.png"));
		confidence.SaveAsPng(Path.Combine(outputRoot, "confidence.png"));
	}

	private static Rgba32 ParseHexColor(string hex)
	{
		if (hex.Length != 7 || hex[0] != '#')
			return new Rgba32(0, 0, 0);

		return new Rgba32(
			Convert.ToByte(hex.Substring(1, 2), 16),
			Convert.ToByte(hex.Substring(3, 2), 16),
			Convert.ToByte(hex.Substring(5, 2), 16));
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

	private static JsonSerializerOptions CreateJsonOptions()
	{
		return new JsonSerializerOptions { WriteIndented = true, PropertyNameCaseInsensitive = true, Converters = { new JsonStringEnumConverter() } };
	}
}

public sealed record MinimapDecompositionReport(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_at_utc")] DateTimeOffset GeneratedAtUtc,
	[property: JsonPropertyName("pattern_library_path")] string PatternLibraryPath,
	[property: JsonPropertyName("pattern_library_schema_version")] string PatternLibrarySchemaVersion,
	[property: JsonPropertyName("minimap_path")] string MinimapPath,
	[property: JsonPropertyName("pattern_candidates")] int PatternCandidates,
	[property: JsonPropertyName("grid_size")] int GridSize,
	[property: JsonPropertyName("max_candidates_per_cell")] int MaxCandidatesPerCell,
	[property: JsonPropertyName("decomposition")] MinimapTilesetDecomposition Decomposition);
